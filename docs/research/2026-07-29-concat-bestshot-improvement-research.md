# Concat Bestshot Improvement Research

Date: 2026-07-29
Branch: `feature/trajectory-pretrain-qformer-concat-bestshot-bf16`

## 1. Runtime truth hien tai

Current bestshot setup is already a strong baseline:

| Axis | Current value |
| --- | --- |
| Mode | `concat + Q-Former + trajectory` |
| Data | `alter_only=true`, `test_alter`, `num_frames=1` |
| Precision | bf16 full-weight, `quantization.enabled=false` |
| LoRA | `r=32`, `alpha=32`, `dropout=0.05`, targets: `wqkv/wo/w1/w2/w3` |
| Trajectory | `d_traj=384`, `num_layers=4`, `ffn_dim=768`, `dropout=0.10` |
| Response | `structured_json` |
| Metric target | `instruction` |
| Objective | teacher-forced token CE over the full response JSON |
| Decode | `num_beams=3`, `repetition_penalty=1.3`, `max_new_tokens=512`, deterministic |

This explains why the current method improves over earlier setups: it fixed several big issues already: bf16 instead of 4-bit, higher LoRA rank, structured supervision, and instruction-field metrics. The remaining gap is likely not one tiny config bug; it is more likely a combination of missing visual context, objective/metric mismatch, and limited trainable capacity at the multimodal seam.

## 2. Evidence tu research

| Topic | Evidence | Implication for this repo |
| --- | --- | --- |
| LoRA capacity | LoRA freezes the base model and injects trainable low-rank updates; rank/target modules control adaptation capacity. Source: [LoRA paper](https://arxiv.org/abs/2106.09685), [HF PEFT LoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora). | `r=32` is reasonable, but `r=64` is a plausible next capacity step if current outputs are close but not expressive enough. |
| Decode matters | HF docs state decoding strategy directly controls generated text behavior. Source: [HF generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies), [HF generation parameters](https://huggingface.co/docs/transformers/en/main_classes/text_generation). | Before retraining, test decode presets. This is cheap and may unlock better ROUGE/GPTScore from the same checkpoint. |
| Label smoothing | PyTorch CE supports `label_smoothing` as regularization. Source: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). | Use mild smoothing only if overfit persists; too much can weaken exact structured output learning. |
| VLM instruction tuning | LLaVA shows multimodal instruction tuning can strongly improve image-conditioned instruction following. Source: [LLaVA paper](https://arxiv.org/abs/2304.08485). | Response/prompt format matters; structured target likely helps because the task needs stable fields and final instruction. |
| Connector/seam tuning | PEFT study on MLLMs reports connector-layer tuning is often important, not just LLM-side PEFT. Source: [ACL Findings PEFT for MLLMs](https://aclanthology.org/2024.findings-acl.598.pdf). | The bridge/Q-Former seam is a high-value capacity target. Freezing too much may cap gains. |
| Multi-frame/video context | Video-LLaVA and LongVLM support the idea that multiple visual frames/segments help temporal understanding. Sources: [Video-LLaVA](https://arxiv.org/abs/2311.10122), [LongVLM](https://arxiv.org/abs/2404.03384). | Since `cebc853` 3-frame was strong, `3-frame + concat trajectory` is the most promising next run. |

## 3. Vi sao hien tai chi tang nhe

### Rat kha nang

1. **Only one frame is the main bottleneck.** Current concat sees the final frame plus trajectory tokens. The old 3-frame setup gave the model more raw visual evidence: appearance, direction changes, relative movement, and context that tracker features may compress away.

2. **Loss is not aligned with the final metric.** Training CE is computed over the whole JSON, but metric only scores `instruction`. The model can get better at boilerplate JSON/location/weather/scene while the final instruction improves only slightly.

3. **Decode is fixed and may be suboptimal.** `num_beams=3` and `repetition_penalty=1.3` can push the model toward safe/template outputs. If outputs look repetitive or overly generic, decode is a cheap suspect.

### Co the

4. **LoRA r32 may still under-adapt for this domain.** Because the base model and vision encoder are mostly frozen, more LoRA capacity (`r64`) or better target coverage may improve domain-specific navigation phrasing.

5. **Q-Former/bridge is too frozen.** Current config keeps Q-Former frozen and only trains projections/LoRA/trajectory. If the trajectory-image-language seam is the bottleneck, partially unfreezing bridge/Q-Former components may help.

6. **Trajectory branch capacity may be enough for tracker features but not semantics.** Going from `384/4/768` to `512/6/1536` may help, but this is less likely than adding frames because tracker input itself may lack appearance/background.

### It kha nang / rui ro cao

7. **More epochs alone.** You already saw early epochs can be strongest, so more epochs without changing supervision/capacity can overfit style.

8. **Huge LoRA rank jump to r128 immediately.** It may help but has higher overfit and VRAM risk. `r64` is the cleaner first step.

9. **Complex RL/MRT/DPO objective.** It could align with metrics, but it is expensive and unstable for this repo right now. Not first tier.

## 4. Shortlist huong nen uu tien

| Rank | Direction | Why | Risk |
| --- | --- | --- | --- |
| 1 | `3-frame + structured_json + concat trajectory + bf16 + r32` | Combines the strongest old signal with current bestshot improvements. | More VRAM/time; must fix placeholder/`num_patches_list` contract carefully. |
| 2 | Decode sweep on existing checkpoints | No retrain, fastest way to test if current model is under-decoded. | May improve metric without improving real understanding. |
| 3 | Instruction-weighted loss | Directly aligns training with the scored field. | Needs careful label masking around JSON; can weaken other fields if too aggressive. |
| 4 | LoRA `r64` | More adaptation capacity with controlled risk. | More trainable params, possible overfit. |
| 5 | Partial bridge/Q-Former tuning | Evidence suggests connector tuning matters in MLLMs. | More implementation risk and possible instability. |
| 6 | Larger trajectory branch `512/6/1536` | More capacity for tracker tokens. | If tracker signal is information-limited, this will not solve the main issue. |
| 7 | Data strategy: hard-example oversampling | Could improve rare hazards/directions. | Can distort distribution and overfit templates. |

## 5. GPU run matrix de xuat

Run these in order. Do not mix too many changes in one run unless marked as a bundle.

### Tier 0: no retrain

Use the best checkpoint already produced and test decode presets:

| Preset | Decode |
| --- | --- |
| current | `num_beams=3`, `repetition_penalty=1.3`, `max_new_tokens=512` |
| concise beam | `num_beams=4`, `length_penalty=0.7`, `repetition_penalty=1.1`, `max_new_tokens=192` |
| less template | `num_beams=1`, `do_sample=false`, `repetition_penalty=1.05`, `max_new_tokens=192` |
| stronger lexical | `num_beams=5`, `length_penalty=0.8`, `repetition_penalty=1.15`, `max_new_tokens=192` |

Decision: if decode alone gives a clear boost, keep training setup and only add configurable decode.

### Tier 1: highest-value training run

`3-frame + structured_json + concat trajectory + bf16 + LoRA r32`.

Keep:
- same pretrain concat checkpoint
- same response format and metric target
- same trajectory architecture
- same LoRA r32

Change:
- dataset returns 3 frames using a config-controlled rule
- prompt has 3 image placeholders
- collate/infer pass correct per-frame patch counts

This is the best first training run because it targets the most likely missing information rather than just increasing parameters.

### Tier 2: capacity run

`1-frame + structured_json + concat + bf16 + LoRA r64`.

Keep everything else fixed. If this beats r32 without earlier overfit, it suggests adaptation capacity was limiting.

### Tier 3: objective-aligned run

`instruction-weighted loss`.

Implementation idea:
- still train full structured JSON
- multiply loss for tokens inside `"instruction"` by `2.0`
- keep normal CE for other JSON fields

This directly addresses the mismatch where metrics score only `instruction`.

### Tier 4: bridge/seam run

Partial bridge/Q-Former tuning:
- keep Q-Former encoder mostly frozen
- allow query tokens and final projection seam to train
- use lower LR than LoRA/trajectory

This is more invasive, but justified if Tier 1/2 plateau.

### Tier 5: larger trajectory run

`d_traj=512`, `num_layers=6`, `ffn_dim=1536`, dropout `0.10`.

Only run this after confirming trajectory tokens are the bottleneck. It is less promising than multi-frame because it cannot recover visual information missing from tracker features.

## 6. Ket luan chot

The strongest next bet is **not** simply more epochs or a bigger trajectory MLP. The most likely real improvement is:

1. first test decode presets on existing checkpoints;
2. then train `3-frame + concat trajectory + structured_json + bf16 + r32`;
3. then try LoRA `r64`;
4. then add instruction-weighted loss.

If only one GPU-expensive run is allowed, choose **Tier 1: 3-frame concat bestshot**. It has the best reason to outperform the current setup because it adds missing raw visual context while preserving everything that already improved the bestshot branch.
