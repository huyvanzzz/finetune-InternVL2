# Concat Bestshot Cebc853 Prompt Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the current `concat` best-shot runtime (`bf16`, LoRA `r=32`, 2 GPU, pretrain checkpoint handoff) while switching supervision toward the `audit-cebc853` prompt and structured output contract.

**Architecture:** Do not change model capacity, trajectory architecture, Q-Former bridge, or pretrain checkpoint loading. Add a separate `concat` config/notebook that uses the `audit-cebc853` structured JSON output style and prompt wording, while preserving one-frame visual input plus trajectory concat.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, PEFT LoRA, Accelerate, YAML configs, pytest source/config tests.

## Global Constraints

- Target worktree: `.worktrees/trajectory-pretrain-qformer-concat-bestshot-bf16`.
- Keep `model.quantization.enabled: false`.
- Keep `training.bf16: true`, `training.fp16: false`.
- Keep LoRA `r: 32`, `alpha: 32`, `dropout: 0.05`.
- Keep `trajectory.fusion_mode: "concat"`.
- Keep trajectory architecture `d_traj=384`, `num_layers=4`, `ffn_dim=768`, `dropout=0.10`.
- Keep `data.alter_only: true`.
- Keep `data.num_frames: 1`; do not port the hard-coded 3-frame image path from `audit-cebc853` in this task.
- Keep manual test path through `scripts/test_infer.py`; do not restore auto-test inside distributed training.
- Do not change checkpoint contract: `--pretrain_checkpoint` starts a new finetune run, `--checkpoint` resumes a finetune checkpoint.

---

### Task 1: Lock Source Truth From `audit-cebc853`

**Files:**
- Read: `.worktrees/audit-cebc853/wad_dataset.py`
- Read: `.worktrees/audit-cebc853/preprocessing.py`
- Read: `.worktrees/audit-cebc853/scripts/test_infer.py`
- Modify: none

**Interfaces:**
- Consumes: `audit-cebc853` prompt and output behavior.
- Produces: exact source constants for Task 2 and Task 3.

- [ ] **Step 1: Record the `audit-cebc853` prompt body**

Use this exact structured prompt body as the source truth:

```text
Analyze: location, weather, traffic, scene -> then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location", "weather", and "traffic".
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction".
```

For alter-only samples, use this exact format instruction:

```text
Format response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>
```

- [ ] **Step 2: Record the output contract**

The output target must be:

```python
answer = f"<answer>{ground_truth_dict.to_json()}</answer>"
```

In the current branch, this is already represented by:

```python
format_ground_truth(sample, response_format="structured_json")
```

- [ ] **Step 3: Record the deliberate non-port**

Do not copy this `audit-cebc853` line:

```python
question = f"<image><image><image>\n{text_content}"
```

The current `concat` branch still uses one final frame, so the correct question prefix remains:

```python
question = f"<image>\n{text_content}"
```

### Task 2: Add A Separate Structured `concat` Config

**Files:**
- Create: `internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml`
- Test: `tests/test_finetune_alter_only.py`

**Interfaces:**
- Consumes: current `internvl_config_traj_concat_bestshot_bf16_2gpu.yaml`.
- Produces: a new config for a clean run without modifying the existing best-shot direct-text config.

- [ ] **Step 1: Write failing config test**

Add this test to `tests/test_finetune_alter_only.py`:

```python
def test_concat_bestshot_cebc853_output_config_uses_structured_output_and_keeps_bf16_r32():
    cfg = yaml.safe_load((ROOT / "internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml").read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == "concat"
    assert cfg["trajectory"]["d_traj"] == 384
    assert cfg["trajectory"]["num_layers"] == 4
    assert cfg["trajectory"]["ffn_dim"] == 768
    assert cfg["trajectory"]["dropout"] == pytest.approx(0.10)
    assert cfg["data"]["alter_only"] is True
    assert cfg["data"]["num_frames"] == 1
    assert cfg["data"]["response_format"] == "structured_json"
    assert cfg["model"]["quantization"]["enabled"] is False
    assert cfg["model"]["lora"]["r"] == 32
    assert cfg["training"]["bf16"] is True
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["gradient_accumulation_steps"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_concat_bestshot_cebc853_output_config_uses_structured_output_and_keeps_bf16_r32 -q
```

Expected:

```text
FAILED ... FileNotFoundError
```

- [ ] **Step 3: Create the config**

Create `internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml` by copying `internvl_config_traj_concat_bestshot_bf16_2gpu.yaml` and changing only these values:

```yaml
experiment:
  name: "wad_internvl3_training_traj_concat_bestshot_bf16_2gpu_cebc853_output"
  description: "InternVL2-2B concat bestshot bf16 r32 with cebc853-style structured JSON supervision"
  tags: ["internvl2", "navigation", "mllm", "trajectory", "concat", "bf16", "2gpu", "bestshot", "structured_json", "cebc853_output"]

data:
  response_format: "structured_json"

training:
  output_dir: "./outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu_cebc853_output"
```

All other values should remain identical to the current best-shot config.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_concat_bestshot_cebc853_output_config_uses_structured_output_and_keeps_bf16_r32 -q
```

Expected:

```text
1 passed
```

### Task 3: Make Structured Prompt Match `audit-cebc853` Exactly

**Files:**
- Modify: `wad_dataset.py`
- Test: `tests/test_finetune_alter_only.py`

**Interfaces:**
- Consumes: `response_format="structured_json"` from Task 2.
- Produces: stable `audit-cebc853` prompt body for dataset samples.

- [ ] **Step 1: Write failing prompt test**

Add this test to `tests/test_finetune_alter_only.py`:

```python
def test_structured_prompt_matches_cebc853_body_but_keeps_single_image_placeholder():
    dataset = wad_dataset.WADDatasetForInternVL(
        metadata_dataset={"train": [{"frame_path": "dummy", "alter": "go forward safely"}]},
        frame_index={},
        bbox_by_folder={},
        trajectory_source=None,
        split="train",
        response_format="structured_json",
    )

    text_content = dataset._build_text_content({"alter": "go forward safely"})
    question = dataset._build_question(text_content)

    assert question.startswith("<image>\n")
    assert "<image><image><image>" not in question
    assert "Analyze: location, weather, traffic, scene -> then give instruction." in question
    assert '1. Perception: Extract "location", "weather", and "traffic".' in question
    assert '2. Comprehension: Synthesize details into the "scene".' in question
    assert '3. Decision: Formulate the final "instruction".' in question
    assert '<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>' in question
```

- [ ] **Step 2: Run test to verify current mismatch if arrow encoding differs**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_structured_prompt_matches_cebc853_body_but_keeps_single_image_placeholder -q
```

Expected before implementation:

```text
FAILED ... assert 'Analyze: location, weather, traffic, scene -> then give instruction.' in question
```

- [ ] **Step 3: Update `_build_text_content`**

In `wad_dataset.py`, update only the `structured_json` branch of `_build_text_content` to use ASCII `->` and the exact `audit-cebc853` wording:

```python
text_content = """

Analyze: location, weather, traffic, scene -> then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location", "weather", and "traffic".
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction"."""
```

Do not change the `direct_text` branch.

- [ ] **Step 4: Run prompt test**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_structured_prompt_matches_cebc853_body_but_keeps_single_image_placeholder -q
```

Expected:

```text
1 passed
```

### Task 4: Verify Structured Output And Metric Target

**Files:**
- Modify: `tests/test_finetune_alter_only.py`
- Read: `preprocessing.py`
- Read: `scripts/test_infer.py`

**Interfaces:**
- Consumes: structured config from Task 2.
- Produces: confidence that train target and test metric both use the `instruction` field.

- [ ] **Step 1: Add output contract test**

Add this test:

```python
def test_structured_output_contract_uses_answer_json_and_instruction_metric():
    sample = {
        "area_type": "Road",
        "weather_condition": "Sunny",
        "traffic_flow_rating": "High",
        "summary": "A busy road with people nearby.",
        "alter": "Please slow down and keep to the left.",
    }

    from preprocessing import format_ground_truth

    answer = format_ground_truth(sample, "structured_json")
    assert answer.startswith("<answer>{")
    assert answer.endswith("}</answer>")
    assert '"location": "road"' in answer
    assert '"weather": "sunny"' in answer
    assert '"traffic": "high"' in answer
    assert '"instruction": "Please slow down and keep to the left."' in answer

    infer_source = (ROOT / "scripts" / "test_infer.py").read_text(encoding="utf-8")
    assert 'metric_target_field = "raw_text" if response_format == "direct_text" else "instruction"' in infer_source
```

- [ ] **Step 2: Run output contract test**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_structured_output_contract_uses_answer_json_and_instruction_metric -q
```

Expected:

```text
1 passed
```

### Task 5: Add A Notebook For The Structured `concat` Run

**Files:**
- Create: `run_qformer_concat_bestshot_bf16_2gpu_cebc853_output.ipynb`
- Modify: `tests/test_finetune_alter_only.py`

**Interfaces:**
- Consumes: config from Task 2.
- Produces: one runnable notebook for server training and manual epoch testing.

- [ ] **Step 1: Write failing notebook test**

Add this test:

```python
def test_concat_bestshot_cebc853_output_notebook_uses_new_config_and_pretrain_checkpoint():
    notebook = json.loads((ROOT / "run_qformer_concat_bestshot_bf16_2gpu_cebc853_output.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    train_cell = "".join(notebook["cells"][7]["source"])
    infer_cell = "".join(notebook["cells"][8]["source"])

    assert 'TARGET_BRANCH = "feature/trajectory-pretrain-qformer-concat-bestshot-bf16"' in cell0
    assert 'CONFIG_PATH = "internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml"' in cell0
    assert 'PRETRAIN_CHECKPOINT = ""' in train_cell
    assert 'cmd += ["--pretrain_checkpoint", PRETRAIN_CHECKPOINT]' in train_cell
    assert 'accelerate", "launch", "--num_processes", "2"' in train_cell
    assert '"--split", "test_alter"' in infer_cell
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_concat_bestshot_cebc853_output_notebook_uses_new_config_and_pretrain_checkpoint -q
```

Expected:

```text
FAILED ... FileNotFoundError
```

- [ ] **Step 3: Create notebook**

Copy `run_qformer_concat_bestshot_bf16_2gpu.ipynb` to `run_qformer_concat_bestshot_bf16_2gpu_cebc853_output.ipynb`, then change only:

```python
CONFIG_PATH = "internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml"
```

Keep:

```python
TARGET_BRANCH = "feature/trajectory-pretrain-qformer-concat-bestshot-bf16"
PRETRAIN_CHECKPOINT = ""
accelerate launch --num_processes 2
--split test_alter
```

- [ ] **Step 4: Run notebook test**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py::test_concat_bestshot_cebc853_output_notebook_uses_new_config_and_pretrain_checkpoint -q
```

Expected:

```text
1 passed
```

### Task 6: Full Verification

**Files:**
- Verify: `train.py`
- Verify: `wad_dataset.py`
- Verify: `scripts/test_infer.py`
- Verify: `preprocessing.py`
- Verify: `tests/test_finetune_alter_only.py`

**Interfaces:**
- Consumes: Tasks 2-5.
- Produces: a run-ready branch for the new structured `concat` case.

- [ ] **Step 1: Static compile**

Run:

```bash
python -m py_compile train.py wad_dataset.py scripts/test_infer.py preprocessing.py tests/test_finetune_alter_only.py
```

Expected:

```text
no output, exit code 0
```

- [ ] **Step 2: Focused tests**

Run:

```bash
python -m pytest tests/test_finetune_alter_only.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Runtime smoke command**

On server, start a new run from the concat pretrain checkpoint:

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml \
  --pretrain_checkpoint huyvanzzz/pretrain_concat
```

Expected startup log must show:

```text
trajectory_mode=concat
alter_only=True
loss_mode=cross_entropy
quantization_enabled=False
bf16=True
LoRA r=32
```

- [ ] **Step 4: Manual test command**

After an epoch checkpoint exists, run:

```bash
python scripts/test_infer.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu_cebc853_output.yaml \
  --checkpoint outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu_cebc853_output/<RUN_ID>/epoch_1 \
  --split test_alter \
  --output_file results/concat_bestshot_cebc853_output_<RUN_ID>_epoch1_test_alter.json
```

Expected:

```text
Computing Metrics (ROUGE, TF-IDF) on 'instruction'...
```

### Task 7: Document Run Difference

**Files:**
- Create: `docs/research/2026-07-29-concat-bestshot-cebc853-output-case.md`

**Interfaces:**
- Consumes: final implementation behavior.
- Produces: a short note so later comparisons do not mix this case with direct-text bestshot.

- [ ] **Step 1: Create research note**

Write this exact structure:

```markdown
# Concat Bestshot Cebc853 Output Case

## What Stays The Same

- bf16 full-weight runtime
- LoRA r=32
- 2 GPU Accelerate training
- concat trajectory fusion
- trajectory 384/4/768/dropout=0.10
- alter-only data
- pretrain checkpoint contract

## What Changes

- `data.response_format` changes from `direct_text` to `structured_json`.
- The training target changes from raw instruction text to `<answer>{... "instruction": ...}</answer>`.
- Test metrics extract and score the `instruction` field.
- The structured prompt body follows the `audit-cebc853` prompt wording.

## What Does Not Change

- This case does not port 3-frame image input.
- This case does not restore auto-test during training.
- This case does not change learning rate, batch size, dropout, or loss.
```

- [ ] **Step 2: Verify note exists**

Run:

```bash
test -f docs/research/2026-07-29-concat-bestshot-cebc853-output-case.md
```

Expected:

```text
exit code 0
```

---

## Self-Review

- Spec coverage: The plan keeps `bf16/r32`, ports `audit-cebc853` prompt/output, keeps concat and one-frame trajectory path, and avoids auto-test in train.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: The plan reuses existing `response_format`, `format_ground_truth`, `WADDatasetForInternVL`, and `scripts/test_infer.py` contracts.
