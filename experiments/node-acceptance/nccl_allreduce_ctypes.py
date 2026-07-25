"""Multi-node NCCL all-reduce, no torch: ctypes on the env-v5 image's own libs (UAT A2, v5-native).

Env v5's Gen-AI task image ships CUDA 12.9 + libnccl.so.2 (2.29.7) but no torch (survey runs
491958602140255, 58718485399690). This probe does what allreduce_probe.py did via torch:
- one process per node (AIR runs `command` once per node with NODE_RANK set)
- each process drives its node's GPUs: one NCCL comm per local GPU via grouped ncclCommInitRank
- ncclUniqueId bootstrap over raw TCP on MASTER_ADDR:MASTER_PORT (torchrun isn't using it)
- correctness: in-place all-reduce of ones == world_size on every GPU (assert-gated)
- bandwidth: 256MB fp32 x10, algbw/busbw like allreduce_probe.py (smoke-grade, not nccl-tests)
- receipt: node 0 logs MLflow params/metrics (client API); sentinel MULTINODE_NCCL_V5_OK

Success = run state SUCCESS + receipt sentinel + world_size == NUM_NODES*LOCAL_WORLD_SIZE.
"""
import ctypes
import ctypes.util
import os
import signal
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import pynvml  # native on v5; vendored fallback for v4

NCCL_FLOAT32 = 7
NCCL_SUM = 0
H2D, D2H = 1, 2
COUNT = 64 * 1024 * 1024  # 256MB fp32
ITERS = 10


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def ck(rc, what):
    assert rc == 0, f"{what} failed with status {rc}"


def bootstrap_unique_id(node_rank, master_addr, master_port, num_nodes, nccl):
    """Node 0 generates the id and serves it; peers fetch it. 128 raw bytes over TCP."""
    uid = NcclUniqueId()
    if node_rank == 0:
        ck(nccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", master_port))
        srv.listen(num_nodes)
        payload = bytes(bytearray(uid.internal))
        for _ in range(num_nodes - 1):
            conn, _addr = srv.accept()
            conn.sendall(payload)
            conn.close()
        srv.close()
    else:
        deadline = time.time() + 300
        data = None
        while time.time() < deadline:
            try:
                s = socket.create_connection((master_addr, master_port), timeout=5)
                data = b""
                while len(data) < 128:
                    chunk = s.recv(128 - len(data))
                    if not chunk:
                        break
                    data += chunk
                s.close()
                if len(data) == 128:
                    break
            except OSError:
                time.sleep(2)
        assert data and len(data) == 128, "failed to fetch ncclUniqueId from node 0"
        ctypes.memmove(uid.internal, data, 128)
    return uid


def log_receipt(params=None, metrics=None):
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        for k, v in (params or {}).items():
            client.log_param(run_id, k, v)
        for k, v in (metrics or {}).items():
            client.log_metric(run_id, k, v)
    except Exception as e:
        print(f"receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def main():
    signal.alarm(1500)  # hard fail-fast: never hang to job timeout
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("POD_RANK", "0")))
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    pynvml.nvmlInit()
    local = int(os.environ.get("LOCAL_WORLD_SIZE", pynvml.nvmlDeviceGetCount()))
    world = num_nodes * local
    uuids = []
    for i in range(local):
        u = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
        uuids.append(u.decode() if isinstance(u, bytes) else u)
    print(f"NODE {node_rank}/{num_nodes} local={local} world={world} "
          f"host={socket.gethostname()} uuids={','.join(u[-12:] for u in uuids)}", flush=True)

    cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
    nccl = ctypes.CDLL("libnccl.so.2")
    ver = ctypes.c_int()
    nccl.ncclGetVersion(ctypes.byref(ver))

    uid = bootstrap_unique_id(node_rank, master_addr, master_port, num_nodes, nccl)
    print(f"NODE {node_rank} uniqueId ready (nccl={ver.value})", flush=True)

    # one comm per local GPU, global ranks node_rank*local + i, grouped init
    comms = [ctypes.c_void_p() for _ in range(local)]
    nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                      NcclUniqueId, ctypes.c_int]
    ck(nccl.ncclGroupStart(), "ncclGroupStart(init)")
    for i in range(local):
        ck(cudart.cudaSetDevice(i), "cudaSetDevice")
        ck(nccl.ncclCommInitRank(ctypes.byref(comms[i]), world, uid, node_rank * local + i),
           "ncclCommInitRank")
    ck(nccl.ncclGroupEnd(), "ncclGroupEnd(init)")
    print(f"NODE {node_rank} comms initialized", flush=True)

    nbytes = COUNT * 4
    ones = struct.pack("<f", 1.0) * COUNT
    bufs = []
    for i in range(local):
        ck(cudart.cudaSetDevice(i), "cudaSetDevice")
        p = ctypes.c_void_p()
        ck(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), "cudaMalloc")
        ck(cudart.cudaMemcpy(p, ones, ctypes.c_size_t(nbytes), H2D), "cudaMemcpy H2D")
        bufs.append(p)
    del ones

    def allreduce_all():
        ck(nccl.ncclGroupStart(), "ncclGroupStart")
        for i in range(local):
            ck(cudart.cudaSetDevice(i), "cudaSetDevice")
            ck(nccl.ncclAllReduce(bufs[i], bufs[i], ctypes.c_size_t(COUNT), NCCL_FLOAT32,
                                  NCCL_SUM, comms[i], None), "ncclAllReduce")
        ck(nccl.ncclGroupEnd(), "ncclGroupEnd")

    def sync_all():
        for i in range(local):
            ck(cudart.cudaSetDevice(i), "cudaSetDevice")
            ck(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    # correctness: ones all-reduced once -> every element == world
    allreduce_all()
    sync_all()
    out = (ctypes.c_float * 8)()
    for i in range(local):
        ck(cudart.cudaSetDevice(i), "cudaSetDevice")
        ck(cudart.cudaMemcpy(out, bufs[i], ctypes.c_size_t(8 * 4), D2H), "cudaMemcpy D2H")
        vals = list(out)
        assert all(v == float(world) for v in vals), \
            f"gpu{i}: allreduce wrong, got {vals[:3]} want {world}"
    print(f"NODE {node_rank} CORRECTNESS_OK all elements == {world}", flush=True)

    # bandwidth (values keep growing world^k — irrelevant for timing; fp32 inf is fine)
    for _ in range(3):
        allreduce_all()
    sync_all()
    t0 = time.time()
    for _ in range(ITERS):
        allreduce_all()
    sync_all()
    dt = (time.time() - t0) / ITERS
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    print(f"NODE {node_rank} all_reduce 256MB x{ITERS}: {dt*1000:.1f} ms/iter, "
          f"algbw {algbw:.1f} GB/s, busbw ~{busbw:.1f} GB/s", flush=True)

    for c in comms:
        nccl.ncclCommDestroy(c)

    if node_rank == 0:
        log_receipt(
            params={
                "probe_sentinel": "MULTINODE_NCCL_V5_OK",
                "world_size": world,
                "num_nodes": num_nodes,
                "local_world_size": local,
                "nccl_version": ver.value,
                "backend": "ctypes-nccl-v5",
                "node0_gpu_uuids": ",".join(uuids),
            },
            metrics={"allreduce_256mb_ms": dt * 1000, "algbw_gbps": algbw, "busbw_gbps": busbw},
        )
        print("MULTINODE_NCCL_V5_OK", flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
