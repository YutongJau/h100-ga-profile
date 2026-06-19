# H100 GPT-2 Large GA Timing Test

Hi, thank you so much for helping me run this. This is a small,
self-contained timing test for GPT-2 Large training steps with different
gradient accumulation values.

I tried to make the instructions below copy-pasteable. You do not need any
SimAI code, dataset, or HuggingFace token. The script builds GPT-2 Large from
configuration and uses synthetic input tokens.

If anything looks weird or fails, please do not spend ages debugging it. Just
send me the output/logs and I can figure it out from there. You are already
saving me a lot here.

## What I Need From You

Could you please run the commands below on an H100 GPU machine and send back:

1. The final output tarball, for example:

   ```text
   h100_profile_<host>_<date>.tar.gz
   ```

2. If the tarball is too large, these files are enough:

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

That is all I need. The tarball is easiest if it works.

## Step 0: Get Onto an H100 GPU Node

Please run this on a compute node with an H100 GPU, not on a login node.

First, check that the machine can see the GPU:

```bash
nvidia-smi -L
```

The output should mention something like:

```text
NVIDIA H100
```

If it does not mention H100, no worries. Please send me what it printed and do
not bother running the rest yet.

## Step 1: Download the Test Code

```bash
git clone https://github.com/YutongJau/h100-ga-profile.git
cd h100-ga-profile
```

## Step 2: Check Python and PyTorch

Please run this quick environment check:

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

What I am hoping to see is roughly:

```text
cuda available True
gpu NVIDIA H100 ...
```

If `torch` or `transformers` is missing, one simple install option is:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers
```

Then run the Python check above again.

If CUDA is still not available, please just send me the output. Please do not
fight the cluster for too long.

## Step 3: Run a Short Smoke Test

This is just a tiny test to make sure the benchmark can run. It is not the real
measurement.

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

If this fails, please send me these files and I will debug it:

```text
h100_profile_smoke/logs/profile_smoke.out
h100_profile_smoke/logs/profile_smoke.err
h100_profile_smoke/telemetry/smoke.environment.txt
```

If it succeeds, you are through the annoying setup part.

## Step 4: Run the Real Measurement

This runs 5 repeats and may take a while. Thank you, truly.

Please copy and paste this whole block:

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

After the real measurement finishes, the summary output should have rows like:

```text
run,host,gpu,torch,cuda,GA1_ms,GA2_ms,GA4_ms,T_fwdbwd_ms,...
r01,...
r02,...
r03,...
r04,...
r05,...
```

Could you please tell me if you notice any of these:

```text
gpu is not H100
cuda available is False
active_throttle_reasons is not empty
SM clock looks very low
temperature is extremely high
any repeat failed before writing tc_profile_*.json
```

Please keep failed runs too. Failed logs are useful, and I would rather receive
too much information than accidentally miss the clue.

## If You Use SLURM Instead

If the cluster uses SLURM and you do not have an interactive GPU shell, please
edit:

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

Please send:

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

The most important final numbers for me are:

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

## Tiny Notes

- Please run this on an H100 if possible.
- If possible, please do not run other heavy GPU jobs on the same GPU during the
  test.
- Please do not edit the Python script unless it fails and you need to change
  something to make it run. If you do change anything, just tell me what changed.
- The smoke test is only for setup. The real measurement is the 5-repeat run in
  Step 4.
- Thank you again. I owe you one.
