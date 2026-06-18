# Portable H100 PyTorch GA Timing Profile

This directory is a self-contained benchmark kit for re-measuring GPT-2 Large
GA timing on an external H100 machine or cluster. It does not depend on the
SimAI Python package and does not download model weights.

## Files

```text
profile_tc_portable.py         Standalone PyTorch GPT-2 Large GA profiler
run_with_telemetry.sh          Runs one profile with nvidia-smi telemetry
submit_slurm_h100_template.sh  SLURM template for batch submission
summarize_results.py           Summarizes JSON results and telemetry CSV logs
make_tarball.sh                Creates a tarball for transfer
```

## Environment Requirements

Minimum:

```text
python
torch with CUDA support
transformers
nvidia-smi
one visible NVIDIA GPU
```

The benchmark constructs GPT-2 Large from config, so no HuggingFace token or
internet access is required.

## Quick Smoke Test

Run this inside an allocated GPU session:

```bash
PYTHON=python3 \
RUN_TAG=smoke \
OUT_DIR=$PWD/h100_profile_smoke \
STEPS=5 \
WARMUP=2 \
bash run_with_telemetry.sh
```

Expected: it writes one JSON file under `h100_profile_smoke/` plus logs and
telemetry files. Do not use smoke-test timing as a calibration value.

## Formal Repeat

Run at least 3 repeats, preferably 4-6:

```bash
for i in 01 02 03 04; do
  PYTHON=python3 \
  RUN_TAG=r${i} \
  OUT_DIR=$PWD/h100_profile_oxford_$(date +%Y%m%d) \
  STEPS=100 \
  WARMUP=20 \
  bash run_with_telemetry.sh
done
```

For SLURM, edit cluster-specific module/account/partition lines in:

```text
submit_slurm_h100_template.sh
```

Then submit, for example:

```bash
sbatch submit_slurm_h100_template.sh
```

## Summarize Results

```bash
python3 summarize_results.py h100_profile_oxford_YYYYMMDD
```

The summary includes:

```text
GA1/GA2/GA4 median step time
T_fwdbwd slope
T_fixed intercept
temperature min/median/max
power min/median/max
SM clock min/median/max
memory clock min/median/max
P-state
active throttle reasons
```

## Rejection Rules

Do not use a repeat as a paper-facing calibration run if any of the following
hold:

```text
GPU is not an H100 when the purpose is H100 calibration
PyTorch cannot see CUDA
active throttle reasons are present during the timed run
hardware or software thermal slowdown appears in nvidia-smi -q
SM clock is persistently much lower than the cluster's normal H100 application clock
the process used CPU fallback or failed before writing a complete JSON file
```

If Oxford differs materially from UCL, keep both as separate calibration
records until telemetry explains the difference. Do not overwrite the UCL
profile value without preserving provenance.

## Transfer

Create a tarball:

```bash
bash make_tarball.sh
```

Copy the resulting `h100_profile_portable_20260618.tar.gz` to the target
machine, extract it, and run the smoke test first.

## Optional GitHub Transfer

This directory is also a standalone git repository. To publish it, create an
empty private GitHub repository, for example:

```text
h100-ga-profile
```

Then push from this directory:

```bash
git remote add origin git@github.com:YOUR_USERNAME/h100-ga-profile.git
git push -u origin main
```

On the target machine:

```bash
git clone git@github.com:YOUR_USERNAME/h100-ga-profile.git
cd h100-ga-profile
```

Then run the smoke test before formal repeats.
