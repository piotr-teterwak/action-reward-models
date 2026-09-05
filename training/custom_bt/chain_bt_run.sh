set -uo pipefail
cd /projectnb/ivc-ml/piotrt/browser_agents/browser-environment
L=output/prm_offline_pilot/train_full/reward_smoke2.log
while qstat -j 5986111 >/dev/null 2>&1; do sleep 45; done
echo "[bt-chain] smoke2 ended $(date)"
if grep -q "\[reward\] done" "$L" && ! grep -qi "OutOfMemory\|Traceback" "$L"; then
  echo "[bt-chain] SMOKE PASSED — launching full all-linear BT run"
  qsub -N reward_bt_al -o output/prm_offline_pilot/train_full/train_reward_bt_alllinear.log \
    -v "OBJECTIVE=bt,TRAIN_DATA=output/prm_offline_pilot/train_full/reward_pairs_15k.jsonl,TRAIN_OUT=output/prm_offline_pilot/train_full/reward_bt_alllinear,LORA_TARGETS=all-linear,LORA_EXCLUDE=.*visual.*,LORA_R=16,LORA_ALPHA=32,EPOCHS=2,SAVE_STEPS=100,BS=1,GA=16" \
    scripts/run_train_reward.qsub
else
  echo "[bt-chain] SMOKE FAILED — not launching. tail:"
  grep -iE "loss|OutOfMemory|Traceback|Error|done|s/it" "$L" | tail -8
fi
echo "[bt-chain] DONE"
