# Judged perspective reasoning

`build_perspective_judged_metadata.py` sends all good/bad/pair check texts in
each metadata row to `gpt-5.6-terra` in one text-only request. It writes
`train.jsonl` and
`test.jsonl` under a sibling directory ending in `_judge` (or `--output-dir`),
and resumes from completed rows. The API labels the polarity of each check,
not whether the annotation is factually correct:

- `1`: the check concludes that the relationship is coherent/correct;
- `0`: the check concludes that the relationship is inconsistent/incorrect.

The image-level label is deterministic (`good=1`, `bad=0`).

Example on the training server:

```bash
cd /data/bagel/repo/Bagel
export EVAL_KEY_ENVS='KEY_0,KEY_1,KEY_2,KEY_3,KEY_4,KEY_5,KEY_6,KEY_7,KEY_8,KEY_9,KEY_20,KEY_21,KEY_22,KEY_23,KEY_24,KEY_25,KEY_26,KEY_27,KEY_28,KEY_29'
/data/bagel/conda/envs/bagel/bin/python -u scripts/build_perspective_judged_metadata.py \
  --data-root /data/bagel/data/perspective_combined_train1400_test20_gptimage2_20260920 \
  --output-dir /data/bagel/data/perspective_combined_train1400_test20_gptimage2_20260920_judge \
  --model gpt-5.6-terra --workers 20 \
  --base-url https://svip.xuedingmao.com/v1
```

For training, keep `BAGEL_REASON_HEATMAP_DATA_DIR` pointed at the original
image root and point `BAGEL_REASON_HEATMAP_METADATA_PATH` at the generated
`..._judge/train.jsonl`. Set `BAGEL_PERSPECTIVE_MULTITASK_REASON=1` and
`BAGEL_PERSPECTIVE_JUDGMENT=1`.

The text objective is now:

```text
Ltext = ce_weight * (Lreason + judgment_ce_weight * Ljudgment)
Ljudgment = 0.5 * Lcheck + 0.5 * Lglobal
```

`Lreason` is the CE over the original `<think>...</think>` explanation.
`Lcheck` averages token CE across the check verdicts; `Lglobal` is the
image-level verdict CE. With `--ce_weight 0.25 --judgment_ce_weight 1`,
the effective coefficients are reason 0.25, checks 0.125, global 0.125.
