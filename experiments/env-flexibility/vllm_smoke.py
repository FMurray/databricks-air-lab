"""vLLM in the standard env (UAT W6) — de-risks the inference-stack-in-GA-env question.

The finding is binary and either answer is valuable: vLLM initializes on this GPU in env v4
(torch/CUDA compat) and generates at a measured rate, or the pip install / init failure IS the
result (→ Workspace Base Environments is plan B). Model deliberately tiny (opt-125m): this
tests the stack, not the model.
"""
import os
import time

MODEL = os.environ.get("VLLM_MODEL", "facebook/opt-125m")

from vllm import LLM, SamplingParams  # import failure = the finding; let it raise loudly

prompts = [
    "The reserved GPU pool is",
    "Serverless training works by",
    "The acceptance test passed because",
]

t0 = time.time()
llm = LLM(model=MODEL, gpu_memory_utilization=0.5)
init_s = time.time() - t0
print(f"RESULT vllm_init_seconds={init_s:.1f} model={MODEL}")

t0 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=128, temperature=0.8))
gen_s = time.time() - t0
n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
assert all(o.outputs[0].text.strip() for o in outs), "empty generation output"
print(f"RESULT vllm_generate=PASS tokens={n_tokens} seconds={gen_s:.2f} "
      f"tokens_per_sec={n_tokens/gen_s:.1f}")
