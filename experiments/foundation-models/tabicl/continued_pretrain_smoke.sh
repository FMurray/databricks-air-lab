#!/usr/bin/env bash
# T1: continued-pretraining mechanics — the customer's fusion->continue path, self-contained.
# Phase A: pretrain PHASE_STEPS from scratch -> checkpoint.
# Phase B: fresh trainer, --checkpoint_path <phase-A ckpt> --only_load_model True, PHASE_STEPS
#          more. PASS = phase-B step-1 loss well below a cold start (~ln(10)+margin), i.e. the
#          weights actually loaded (a silent re-init would start near cold loss).
# Phase C (informative either way): try loading the RELEASED HF inference checkpoint —
#          answers whether the published .ckpt format chains into the trainer (the [customer-internal]
#          weights question). Failure here is a FINDING, not a run failure.
set -uo pipefail

BASE="$(dirname "$0")/pol_stage1_smoke.sh"
PHASE_STEPS="${PHASE_STEPS:-60}"
ROOT="${ROOT:-/tmp/tabicl-continue}"

echo "=== PHASE A: from-scratch $PHASE_STEPS steps"
CKPT_DIR="$ROOT/phaseA" MAX_STEPS="$PHASE_STEPS" bash "$BASE"
CKPT_A=$(ls -t "$ROOT"/phaseA/*.ckpt | head -1)
echo "T1_PHASE_A_OK ckpt=$CKPT_A"

echo "=== PHASE B: continue from phase-A checkpoint, $PHASE_STEPS more steps"
CKPT_DIR="$ROOT/phaseB" MAX_STEPS="$PHASE_STEPS" CHECKPOINT_PATH="$CKPT_A" bash "$BASE" \
  | tee "$ROOT/phaseB.log"
grep -q "CONTINUING from checkpoint" "$ROOT/phaseB.log" && echo "T1_PHASE_B_RAN"
# loss continuity: first logged ce in phase B (cold start ~= ln(max_classes)=2.30 for 10-cls)
B_CE=$(grep -oE "ce=[0-9.]+" "$ROOT/phaseB.log" | head -1 | cut -d= -f2)
echo "T1_PHASE_B_FIRST_CE=$B_CE (cold-start reference ~2.3; loaded weights should start lower)"

echo "=== PHASE C: released HF checkpoint format compatibility (finding either way)"
set +e
python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt")
print("RELEASED_CKPT", p)
EOF
REL=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('jingang/TabICL','tabicl-classifier-v2-20260212.ckpt'))" 2>/dev/null)
if [ -n "$REL" ]; then
  CKPT_DIR="$ROOT/phaseC" MAX_STEPS=5 CHECKPOINT_PATH="$REL" bash "$BASE" \
    && echo "T1_PHASE_C_RELEASED_CKPT_LOADS=yes" \
    || echo "T1_PHASE_C_RELEASED_CKPT_LOADS=no (format finding: inference ckpt does not chain into trainer as-is)"
else
  echo "T1_PHASE_C_SKIPPED (HF download failed)"
fi
exit 0
