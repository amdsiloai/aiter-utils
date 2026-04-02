#!/usr/bin/env python3
"""
Extract Untuned GEMM Shapes

Scans docker container log files for AITER GEMM shapes that fell back
to default config and writes the unique (M, N, K) tuples to a CSV.

Usage:
    python extract_gemm_shapes.py ./logs
    python extract_gemm_shapes.py ./logs --gemm-type a8w8_blockscale -o shapes.csv
"""

import csv
import re
import argparse
from pathlib import Path


def extract_gemm_shapes(
    log_dir: str,
    gemm_type: str = "a8w8_blockscale",
    output_csv: str = None,
    verbose: bool = True,
) -> list:
    """
    Extract unique untuned GEMM shapes from log files.

    Parses lines like:
        shape is M:6314, N:2048, K:2048, not found tuned config in
        /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default config!

    Args:
        log_dir: Directory containing log files (searched recursively)
        gemm_type: GEMM type to filter for (matches the csv filename in the log)
        output_csv: Path to write deduplicated shapes CSV. If None, uses
            {log_dir}/{gemm_type}_untuned_shapes.csv
        verbose: Print progress

    Returns:
        Sorted list of unique (M, N, K) tuples
    """
    shape_re = re.compile(
        r"shape is M:(\d+),\s*N:(\d+),\s*K:(\d+),\s*not found tuned config"
    )

    log_path = Path(log_dir)
    log_files = list(log_path.rglob("*.log"))

    if verbose:
        print(f"Scanning {len(log_files)} log file(s) for untuned {gemm_type} GEMM shapes...")

    all_shapes = set()
    files_with_matches = 0

    for log_file in log_files:
        file_shapes = set()
        try:
            with open(log_file, "r", errors="replace") as f:
                for line in f:
                    if gemm_type not in line:
                        continue
                    m = shape_re.search(line)
                    if m:
                        shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        file_shapes.add(shape)
        except Exception as e:
            if verbose:
                print(f"  Warning: could not read {log_file.name}: {e}")
            continue

        if file_shapes:
            files_with_matches += 1
            all_shapes.update(file_shapes)
            if verbose:
                print(f"  {log_file.relative_to(log_path)}: {len(file_shapes)} unique shapes")

    sorted_shapes = sorted(all_shapes)

    if not sorted_shapes:
        if verbose:
            print(f"  No untuned {gemm_type} GEMM shapes found.")
        return []

    if output_csv is None:
        output_csv = str(log_path / f"{gemm_type}_untuned_shapes.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K"])
        for m, n, k in sorted_shapes:
            writer.writerow([m, n, k])

    if verbose:
        print(f"\n{len(sorted_shapes)} unique shapes from {files_with_matches} file(s)")
        print(f"Written to: {output_csv}")

    return sorted_shapes


def main():
    parser = argparse.ArgumentParser(
        description="Extract untuned GEMM shapes from docker container logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_gemm_shapes.py ./logs
  python extract_gemm_shapes.py ./logs --gemm-type a8w8_blockscale
  python extract_gemm_shapes.py ./logs -o untuned_shapes.csv
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
        "--output-csv",
        "-o",
        default=None,
        help="Output CSV path (default: {log_dir}/{gemm_type}_untuned_shapes.csv)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    extract_gemm_shapes(
        log_dir=args.log_dir,
        gemm_type=args.gemm_type,
        output_csv=args.output_csv,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    exit(main())
