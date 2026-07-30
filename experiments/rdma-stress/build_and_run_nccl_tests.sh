#!/usr/bin/env bash
# M4: the defensible benchmark — build nccl-tests from vendored source with the image's
# own toolchain (no egress), then run all_reduce_perf. Single-process: multi-GPU via -g;
# MULTI-NODE REQUIRES MPI — this script reports whether mpirun exists (expected absent on
# the AIR image → M4 stays intra-node/NVLink-grade until MPI or a launcher shim exists).
# Env knobs: GPUS (default: all), MINBYTES/MAXBYTES (default 8..4G).
set -euo pipefail

SRC="$CODE_SOURCE_PATH/experiments/rdma-stress/nccl-tests-src"
echo "M4 toolchain: nvcc=$(command -v nvcc || echo MISSING) mpirun=$(command -v mpirun || echo MISSING)"
ls /usr/include/nccl.h 2>/dev/null || ls /usr/local/cuda*/include/nccl.h 2>/dev/null \
  || echo "M4 nccl.h: NOT FOUND in standard locations (build will tell)"

BUILD=/tmp/nccl-tests-build
cp -r "$SRC" "$BUILD"
cd "$BUILD"
make -j"$(nproc)" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" 2>&1 | tail -5

NGPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
echo "M4 running all_reduce_perf -g $NGPUS"
./build/all_reduce_perf -b "${MINBYTES:-8}" -e "${MAXBYTES:-4G}" -f 2 -g "$NGPUS"
echo "M4_NCCL_TESTS_DONE"
