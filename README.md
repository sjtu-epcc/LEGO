# LEGO
LEGO: Supporting LLM-enhanced Games with One Gaming GPU [HPCA 2026]

# README (Simplified Reproduction Guide)

This project contains four major components required to reproduce the experiments in LEGO.
Because real-world environments (UE4 version, GPU count, model paths, dataset paths) differ among users, **the provided scripts are templates** — users are expected to **modify paths and configuration parameters inside each `.sh` script** according to their system.

Below we outline what each directory does and how to run the experiments described in Section VII of the paper.

---

# A.5 Evaluation and Expected Results

## A.5.1 Repository Layout

```
LEGO/
├── games/        # Rendering traces collected from three representative games
├── scheduler/    # Custom scheduling logic replacing UE4 engine components
├── adaptor/      # Training and evaluation of the adaptor module
└── fighting/     # Two-model competition environment
```

**Important:**
The `.sh` files inside each directory are **templates**, not plug-and-play executables.
Users must modify:

* model paths
* dataset paths
* GPU settings
* UE4 installation paths
* environment variables

before executing.

---

## A.5.2 Section VII-B — UE4 Scheduling Experiments

1. Install Unreal Engine 4 (UE4).
2. Replace the scheduling components with those in:

```
scheduler/
```

3. We provide:

```
configure_scheduler.sh
```

This script **automates code replacement inside UE4**, but users must modify the UE4 installation path inside the script.

4. To simulate game workloads:

```
run_scheduler.sh
```

This launches multiple co-location cases (game + LLM).
Again, **users must edit the script** to point to their UE4 project paths.

5. All experimental results are saved locally as `.csv`.
   Each entry logs:

* rendering frame execution time
* LLM inference time

These CSVs can be used to compute:

* 99th-percentile FPS
* 99th-percentile APM

as shown in Section VII-B.

---

## A.5.3 Section VII-C — Heatmap + Adaptor Training + Accuracy Evaluation

### Step 1: Construct heatmaps

In `adaptor/`, we provide:

```
draw_heatmap.sh
```

This script calls:

```
struct_heatmap.py
```

for both Llama3-8B and Mistral-7B.

Users must update:

* `--model_path`

inside the script.

### Step 2: Train one adaptor per resource configuration

We provide:

```
train_adaptor.sh
```

This script uses LLaMA-Factory to train multiple adaptor variants.
Users must:

* modify model directory
* install dependencies
* adjust YAML config files
* ensure correct GPU layout

### Step 3: Evaluate accuracy

Run:

```
run_test.sh
```

This evaluates MMLU / ARC-C / SQuAD2.0, producing results comparable to Table IV.

All scripts require path modifications.

---

## A.5.4 Section VII-D — Two-Model Fighting Experiment

1. Install the fighting environment:

```
configure_fighting.sh
```

Need to set up Diambra and prepares ROM files accodring to instructions.
Users must modify environment paths inside the script.

2. Start the LLM policy servers:

```
run_fighting.sh
```

This launches two local LLM inference endpoints (one per GPU).
Users must change:

* `MODEL_PATH`
* GPU indices
* ports

3. The fighting loop records performance metrics into CSV files.

> **Note:** This experiment requires multiple GPUs so both models can act simultaneously.

---

# A.6 Implementation Notes

* All experiments use **Unreal Engine 4 + DirectX 12**.
* Model inference uses a modified frontend of **LLaMA.cpp**, linked via dynamic library.
* LEGO does **not** depend on a specific UE4 version or LLaMA.cpp version.
* Layer-skipping adaptation and temporal scheduling logic are **fully portable** across engines and LLM frameworks.
* Users may integrate LEGO with other engines or transformer runtimes, as long as:

  * rendering frame boundaries are exposed
  * inference subtasks can be scheduled at layer granularity

---

# Final Reminder for Users

The `.sh` scripts provided in the repository **are not plug-and-play**.
They serve as **templates** that demonstrate the **correct sequence of steps**, but users **must manually adjust**:

* filesystem paths
* UE4 installation locations
* dataset/model directories
* GPU configuration
* Python environment names
* YAML training configs

before running.

This reflects real experimental environments, where paths, hardware, and game engines differ across users.
