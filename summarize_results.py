#!/usr/bin/env python3
"""Summarize portable H100/A100 profile JSON files and telemetry CSV logs."""

import argparse
import csv
import glob
import json
import os
import statistics


def parse_float_with_unit(value):
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def telemetry_summary(csv_path):
    if not os.path.exists(csv_path):
        return {}
    temps = []
    powers = []
    power_limits = []
    sm_clocks = []
    mem_clocks = []
    pstates = set()
    throttle_reasons = set()
    row_count = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            normalized = {k.strip(): v for k, v in row.items()}
            temp = parse_float_with_unit(normalized.get("temperature.gpu"))
            power = parse_float_with_unit(normalized.get("power.draw [W]"))
            limit = parse_float_with_unit(normalized.get("power.limit [W]"))
            sm_clock = parse_float_with_unit(normalized.get("clocks.sm [MHz]"))
            mem_clock = parse_float_with_unit(normalized.get("clocks.mem [MHz]"))
            pstate = (normalized.get("pstate") or "").strip()
            reason = (normalized.get("clocks_throttle_reasons.active") or "").strip()

            if temp is not None:
                temps.append(temp)
            if power is not None:
                powers.append(power)
            if limit is not None:
                power_limits.append(limit)
            if sm_clock is not None:
                sm_clocks.append(sm_clock)
            if mem_clock is not None:
                mem_clocks.append(mem_clock)
            if pstate:
                pstates.add(pstate)
            if reason and reason.upper() not in {"N/A", "NONE", "NOT ACTIVE"}:
                throttle_reasons.add(reason)

    def stats(values):
        if not values:
            return ""
        return f"{min(values):.0f}/{statistics.median(values):.0f}/{max(values):.0f}"

    return {
        "telemetry_rows": row_count,
        "temp_min_med_max_C": stats(temps),
        "power_min_med_max_W": stats(powers),
        "power_limit_med_W": f"{statistics.median(power_limits):.0f}" if power_limits else "",
        "sm_clock_min_med_max_MHz": stats(sm_clocks),
        "mem_clock_min_med_max_MHz": stats(mem_clocks),
        "pstates": "|".join(sorted(pstates)),
        "active_throttle_reasons": "|".join(sorted(throttle_reasons)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Output directories or JSON files")
    args = parser.parse_args()

    json_paths = []
    for path in args.paths:
        if os.path.isdir(path):
            json_paths.extend(sorted(glob.glob(os.path.join(path, "tc_profile*.json"))))
        else:
            json_paths.append(path)

    rows = []
    for json_path in sorted(set(json_paths)):
        with open(json_path) as f:
            data = json.load(f)
        base = os.path.basename(json_path)
        run_tag = base
        for prefix in ("tc_profile_h100_telemetry_", "tc_profile_h100_", "tc_profile_"):
            if run_tag.startswith(prefix):
                run_tag = run_tag[len(prefix) :]
        if run_tag.endswith(".json"):
            run_tag = run_tag[:-5]

        ga = data.get("ga_results", {})
        telem_csv = os.path.join(
            os.path.dirname(json_path),
            "telemetry",
            f"{run_tag}.query.csv",
        )
        if not os.path.exists(telem_csv):
            telem_csv = os.path.join(
                os.path.dirname(json_path),
                "telemetry",
                f"profH100tel_{run_tag}.query.csv",
            )

        row = {
            "run": run_tag,
            "host": data.get("host", ""),
            "gpu": data.get("gpu", ""),
            "torch": data.get("torch", ""),
            "cuda": data.get("torch_cuda", ""),
            "GA1_ms": ga.get("1", {}).get("median_ms", ""),
            "GA2_ms": ga.get("2", {}).get("median_ms", ""),
            "GA4_ms": ga.get("4", {}).get("median_ms", ""),
            "T_fwdbwd_ms": data.get("T_fwdbwd_ms", ""),
            "T_fixed_ms": data.get("T_fixed_ms", ""),
            "json": json_path,
        }
        row.update(telemetry_summary(telem_csv))
        rows.append(row)

    headers = [
        "run",
        "host",
        "gpu",
        "torch",
        "cuda",
        "GA1_ms",
        "GA2_ms",
        "GA4_ms",
        "T_fwdbwd_ms",
        "T_fixed_ms",
        "telemetry_rows",
        "temp_min_med_max_C",
        "power_min_med_max_W",
        "power_limit_med_W",
        "sm_clock_min_med_max_MHz",
        "mem_clock_min_med_max_MHz",
        "pstates",
        "active_throttle_reasons",
        "json",
    ]
    print(",".join(headers))
    for row in rows:
        print(",".join(str(row.get(h, "")) for h in headers))

    t_vals = [
        float(row["T_fwdbwd_ms"])
        for row in rows
        if row.get("T_fwdbwd_ms") not in ("", None)
    ]
    if t_vals:
        print()
        print(
            "T_fwdbwd aggregate: "
            f"n={len(t_vals)} "
            f"mean={statistics.mean(t_vals):.2f} "
            f"median={statistics.median(t_vals):.2f} "
            f"min={min(t_vals):.2f} "
            f"max={max(t_vals):.2f}"
        )


if __name__ == "__main__":
    main()
