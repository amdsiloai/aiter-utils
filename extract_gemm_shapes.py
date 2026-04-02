#!/usr/bin/env python3
"""
Extract GEMM Shapes (Tuned & Untuned)

Scans docker container log files for AITER GEMM shapes and writes
deduplicated results to CSVs, grouped into tuned vs. untuned.

Untuned shapes come from lines like:
    shape is M:6314, N:2048, K:2048, not found tuned config in
    /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default config!

Tuned shapes come from the startup tuning table like:
    (EngineCore pid=661)     256    16 7168  256      ck   8   0   2.97 a8w8_blockscale_...   19.79  697.21  0.0

Usage:
    python extract_gemm_shapes.py ./logs
    python extract_gemm_shapes.py ./logs --gemm-type a8w8_blockscale -o shapes_dir
"""

import csv
import re
import argparse
from pathlib import Path


TUNED_COLS = [
    "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK",
    "us", "kernelName", "tflops", "bw", "errRatio",
]


def extract_gemm_shapes(
    log_dir: str,
    gemm_type: str = "a8w8_blockscale",
    output_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Extract unique tuned and untuned GEMM shapes from log files.

    Args:
        log_dir: Directory containing log files (searched recursively)
        gemm_type: GEMM type to filter for
        output_dir: Directory to write output CSVs. If None, uses log_dir.
        verbose: Print progress

    Returns:
        Dict with "tuned" and "untuned" lists of shapes
    """
    untuned_re = re.compile(
        r"shape is M:(\d+),\s*N:(\d+),\s*K:(\d+),\s*not found tuned config"
    )

    # Tuned table row: cu_num M N K libtype kernelId splitK us kernelName tflops bw errRatio
    tuned_re = re.compile(
        r"\s+(\d+)"           # cu_num
        r"\s+(\d+)"           # M
        r"\s+(\d+)"           # N
        r"\s+(\d+)"           # K
        r"\s+(\w+)"           # libtype
        r"\s+(\d+)"           # kernelId
        r"\s+(\d+)"           # splitK
        r"\s+([\d.]+)"        # us
        r"\s+(\S+)"           # kernelName
        r"\s+([\d.]+)"        # tflops
        r"\s+([\d.]+)"        # bw
        r"\s+([\d.]+)"        # errRatio
    )

    log_path = Path(log_dir)
    log_files = list(log_path.rglob("*.log"))

    if verbose:
        print(f"Scanning {len(log_files)} log file(s) for {gemm_type} GEMM shapes...")

    all_untuned = set()
    all_tuned = {}  # (M, N, K) -> best row dict (by lowest us)
    files_with_untuned = 0
    files_with_tuned = 0

    for log_file in log_files:
        file_untuned = set()
        file_tuned_count = 0
        try:
            with open(log_file, "r", errors="replace") as f:
                for line in f:
                    if gemm_type not in line:
                        continue

                    m = untuned_re.search(line)
                    if m:
                        file_untuned.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                        continue

                    m = tuned_re.search(line)
                    if m:
                        row = {
                            "cu_num": int(m.group(1)),
                            "M": int(m.group(2)),
                            "N": int(m.group(3)),
                            "K": int(m.group(4)),
                            "libtype": m.group(5),
                            "kernelId": int(m.group(6)),
                            "splitK": int(m.group(7)),
                            "us": float(m.group(8)),
                            "kernelName": m.group(9),
                            "tflops": float(m.group(10)),
                            "bw": float(m.group(11)),
                            "errRatio": float(m.group(12)),
                        }
                        key = (row["M"], row["N"], row["K"])
                        if key not in all_tuned or row["us"] < all_tuned[key]["us"]:
                            all_tuned[key] = row
                        file_tuned_count += 1

        except Exception as e:
            if verbose:
                print(f"  Warning: could not read {log_file.name}: {e}")
            continue

        if file_untuned:
            files_with_untuned += 1
            all_untuned.update(file_untuned)
        if file_tuned_count:
            files_with_tuned += 1

        if verbose and (file_untuned or file_tuned_count):
            rel = log_file.relative_to(log_path)
            parts = []
            if file_untuned:
                parts.append(f"{len(file_untuned)} untuned")
            if file_tuned_count:
                parts.append(f"{file_tuned_count} tuned")
            print(f"  {rel}: {', '.join(parts)}")

    out = Path(output_dir) if output_dir else log_path
    out.mkdir(parents=True, exist_ok=True)

    sorted_tuned = sorted(all_tuned.values(), key=lambda r: (r["M"], r["N"], r["K"]))
    sorted_untuned = sorted(all_untuned)

    tuned_only_keys = set(all_tuned.keys())
    untuned_only_keys = all_untuned - tuned_only_keys
    overlap_keys = all_untuned & tuned_only_keys

    if verbose:
        print(f"\nTuned:   {len(sorted_tuned)} unique (M,N,K) from {files_with_tuned} file(s)")
        print(f"Untuned: {len(sorted_untuned)} unique (M,N,K) from {files_with_untuned} file(s)")
        if overlap_keys:
            print(f"Overlap: {len(overlap_keys)} shapes appear in both tuned and untuned")

    # Write tuned CSV with full kernel details
    if sorted_tuned:
        tuned_csv = out / f"{gemm_type}_tuned_shapes.csv"
        with open(tuned_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TUNED_COLS)
            writer.writeheader()
            writer.writerows(sorted_tuned)
        if verbose:
            print(f"\nTuned  -> {tuned_csv}")

    # Write untuned CSV (M, N, K only)
    if sorted_untuned:
        untuned_csv = out / f"{gemm_type}_untuned_shapes.csv"
        with open(untuned_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["M", "N", "K"])
            for m, n, k in sorted_untuned:
                writer.writerow([m, n, k])
        if verbose:
            print(f"Untuned -> {untuned_csv}")

    # Write combined summary (M, N, K, status)
    all_keys = tuned_only_keys | all_untuned
    combined = []
    for key in sorted(all_keys):
        m, n, k = key
        if key in tuned_only_keys and key in all_untuned:
            status = "both"
        elif key in tuned_only_keys:
            status = "tuned"
        else:
            status = "untuned"
        combined.append({"M": m, "N": n, "K": k, "status": status})

    if combined:
        combined_csv = out / f"{gemm_type}_all_shapes.csv"
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["M", "N", "K", "status"])
            writer.writeheader()
            writer.writerows(combined)
        if verbose:
            status_counts = {}
            for row in combined:
                status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            summary_parts = [f"{v} {k}" for k, v in sorted(status_counts.items())]
            print(f"All    -> {combined_csv}  ({', '.join(summary_parts)})")

    return {
        "tuned": sorted_tuned,
        "untuned": sorted_untuned,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract tuned and untuned GEMM shapes from docker container logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_gemm_shapes.py ./logs
  python extract_gemm_shapes.py ./logs --gemm-type a8w8_blockscale
  python extract_gemm_shapes.py ./logs -o ./shapes
        """,
    )

    parser.add_argument(
        "log_dir",
        help="Directory containing log files (searched recursively)",
    )
    parser.add_argument(
        "--gemm-type",
        "-g",
        default="a8w8_blockscale",
        help='GEMM type to extract (default: "a8w8_blockscale")',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for CSVs (default: same as log_dir)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    extract_gemm_shapes(
        log_dir=args.log_dir,
        gemm_type=args.gemm_type,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    exit(main())
