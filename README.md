# H100 GPT-2 Large GA Timing Test

This repository measures the GPU time for GPT-2 Large training steps with
different gradient accumulation values. It is meant to be easy to run on a new
H100 machine or cluster.

You do not need any dataset, HuggingFace token, or SimAI code. The script builds
GPT-2 Large from configuration and uses synthetic input tokens.

## What I Need From You

Please run the commands below on an H100 GPU machine and send back:

1. The final output tarball, for example:

   ```text
   h100_profile_<host>_<date>.tar.gz
   ```

2. If the tarball is too large, send these files instead:

   ```text
   h100_profile_*/summary.csv
   h100_profile_*/tc_profile_*.json
   h100_profile_*/telemetry/*.query.csv
   h100_profile_*/telemetry/*.environment.txt
   h100_profile_*/telemetry/*.nvidia_smi_q_before.txt
   h100_profile_*/telemetry/*.nvidia_smi_q_after.txt
   h100_profile_*/logs/*.out
   h100_profile_*/logs/*.err
   ```

3. Also copy-paste the final terminal output from the summary command.

## Step 0: Get Onto a GPU Node

Run this on a compute node with an H100 GPU, not on a login node.

First check that the machine can see a GPU:

```bash
nvidia-smi -L
```

Expected output should mention something like:

```text
NVIDIA H100
```

If it does not show an H100, stop and tell me what it printed.

## Step 1: Download the Test Code

```bash
git clone https://github.com/YutongJau/h100-ga-profile.git
cd h100-ga-profile
```

## Step 2: Check Python and PyTorch

Run:

```bash
python3 - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch error:", repr(e))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("transformers error:", repr(e))
PY
```

If `torch` or `transformers` is missing, install them in your environment. One
simple option is:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers
```

Then run the Python check above again.

Important:

```text
cuda available True
gpu NVIDIA H100 ...
```

If CUDA is not available, stop and send me the error/output.

## Step 3: Run a Short Smoke Test

This checks that the benchmark works. It is not the final measurement.

```bash
PYTHON=python3 \
RUN_TAG=smoke \
OUT_DIR=$PWD/h100_profile_smoke \
STEPS=5 \
WARMUP=2 \
bash run_with_telemetry.sh
```

Then summarize it:

```bash
python3 summarize_results.py h100_profile_smoke
```

If this fails, send me:

```text
h100_profile_smoke/logs/profile_smoke.out
h100_profile_smoke/logs/profile_smoke.err
h100_profile_smoke/telemetry/smoke.environment.txt
```

If it succeeds, continue.

## Step 4: Run the Real Measurement

This runs 5 repeats. It may take a while.

Copy and paste this whole block:

```bash
OUT_DIR=$PWD/h100_profile_$(hostname)_$(date +%Y%m%d_%H%M)

for i in 01 02 03 04 05; do
  echo "=== repeat $i ==="
  PYTHON=python3 \
  RUN_TAG=r${i} \
  OUT_DIR=$OUT_DIR \
  STEPS=100 \
  WARMUP=20 \
  bash run_with_telemetry.sh
done

python3 summarize_results.py "$OUT_DIR" | tee "$OUT_DIR/summary.csv"
tar -czf "${OUT_DIR}.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "DONE"
echo "Please send this file:"
echo "${OUT_DIR}.tar.gz"
```

Please send me the `.tar.gz` file printed at the end.

## Step 5: Quick Sanity Check Before Sending

After the real measurement finishes, look at the summary output.

It should have rows like:

```text
run,host,gpu,torch,cuda,GA1_ms,GA2_ms,GA4_ms,T_fwdbwd_ms,...
r01,...
r02,...
r03,...
r04,...
r05,...
```

Please tell me if you see any of these problems:

```text
gpu is not H100
cuda available is False
active_throttle_reasons is not empty
SM clock looks very low
temperature is extremely high
any repeat failed before writing tc_profile_*.json
```

Do not delete failed runs; include them in the tarball so I can debug.

## If You Use SLURM Instead

If your cluster uses SLURM and you do not have an interactive GPU shell, edit:

```text
submit_slurm_h100_template.sh
```

At minimum, adjust any required cluster-specific lines such as partition,
account, modules, or virtualenv activation. Then run:

```bash
mkdir -p logs
sbatch submit_slurm_h100_template.sh
```

After the job finishes, run:

```bash
python3 summarize_results.py h100_profile_outputs_* | tee slurm_summary.csv
tar -czf h100_profile_slurm_outputs.tar.gz h100_profile_outputs_* logs slurm_summary.csv
```

Send:

```text
h100_profile_slurm_outputs.tar.gz
```

## What the Scripts Produce

Each repeat writes:

```text
tc_profile_<run>.json
logs/profile_<run>.out
logs/profile_<run>.err
telemetry/<run>.environment.txt
telemetry/<run>.query.csv
telemetry/<run>.dmon.log
telemetry/<run>.nvidia_smi_q_before.txt
telemetry/<run>.nvidia_smi_q_after.txt
```

The most important final numbers are:

```text
GA1_ms
GA2_ms
GA4_ms
T_fwdbwd_ms
T_fixed_ms
temperature
power
SM clock
memory clock
P-state
active throttle reasons
```

## Notes

- Please run this on an H100 if possible.
- Please do not run other heavy GPU jobs on the same GPU during the test.
- Please do not edit the Python script unless it fails and you need to tell me
  what changed.
- The smoke test is only for checking setup. The real measurement is the 5-repeat
  run in Step 4.
