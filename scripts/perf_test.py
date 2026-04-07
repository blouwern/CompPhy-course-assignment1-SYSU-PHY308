#!/usr/bin/env python3
"""Automate performance tests for MPIMatrixMultiply_bsp.

Runs each matrix size multiple times, parses output timing lines, and writes CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


TIME_PATTERNS = {
    "MPI": re.compile(r"^\[Time taken\]\s+MPI:\s+([0-9.]+)\s+seconds\s*$"),
    "Rough simple sequential": re.compile(
        r"^\[Time taken\]\s+Rough simple sequential:\s+([0-9.]+)\s+seconds\s*$"
    ),
    "CBLAS": re.compile(r"^\[Time taken\]\s+CBLAS:\s+([0-9.]+)\s+seconds\s*$"),
}


@dataclass
class RunResult:
    mpi: Optional[float]
    seq: Optional[float]
    cblas: Optional[float]


@dataclass
class SizeResult:
    nrows_a: int
    ncols_a: int
    nrows_b: int
    ncols_b: int
    runs: List[RunResult]

    def mean(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        mpi = _mean_optional(r.mpi for r in self.runs)
        seq = _mean_optional(r.seq for r in self.runs)
        cblas = _mean_optional(r.cblas for r in self.runs)
        return mpi, seq, cblas


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    total = 0.0
    count = 0
    for value in values:
        if value is None:
            continue
        total += value
        count += 1
    if count == 0:
        return None
    return total / count


def parse_timings(stdout: str) -> RunResult:
    values = {"MPI": None, "Rough simple sequential": None, "CBLAS": None}
    for line in stdout.splitlines():
        for key, pat in TIME_PATTERNS.items():
            match = pat.match(line.strip())
            if match:
                values[key] = float(match.group(1))
    return RunResult(mpi=values["MPI"], seq=values["Rough simple sequential"], cblas=values["CBLAS"])


def run_once(mpirun: str, exe_path: str, np: int, size: Tuple[int, int, int, int]) -> RunResult:
    args = [mpirun, "-np", str(np), exe_path] + [str(x) for x in size]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed (exit {code}): {cmd}\nstdout:\n{out}\nstderr:\n{err}".format(
                code=completed.returncode,
                cmd=" ".join(args),
                out=completed.stdout,
                err=completed.stderr,
            )
        )
    return parse_timings(completed.stdout)


def parse_sizes(size_args: Iterable[str]) -> List[Tuple[int, int, int, int]]:
    sizes: List[Tuple[int, int, int, int]] = []
    for s in size_args:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"Invalid size '{s}'. Expected 4 comma-separated ints.")
        sizes.append(tuple(int(p) for p in parts))
    return sizes


def write_csv(path: str, results: List[SizeResult]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_matrixA_row",
                "n_matrixA_col",
                "n_matrixB_row",
                "n_matrixB_col",
                "runs",
                "mpi_avg_sec",
                "seq_avg_sec",
                "cblas_avg_sec",
            ]
        )
        for res in results:
            mpi, seq, cblas = res.mean()
            writer.writerow(
                [
                    res.nrows_a,
                    res.ncols_a,
                    res.nrows_b,
                    res.ncols_b,
                    len(res.runs),
                    "" if mpi is None else f"{mpi:.6f}",
                    "" if seq is None else f"{seq:.6f}",
                    "" if cblas is None else f"{cblas:.6f}",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance test for MPIMatrixMultiply_bsp")
    parser.add_argument(
        "--exe",
        default="build/MPIMatrixMultiply_bsp",
        help="Path to MPIMatrixMultiply_bsp",
    )
    parser.add_argument("--mpirun", default="mpirun", help="mpirun command")
    parser.add_argument("--np", type=int, help="MPI process count")
    parser.add_argument("--runs", type=int, default=5, help="Runs per size")
    parser.add_argument(
        "--out",
        default="report/perf_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--size",
        action="append",
        required=True,
        help="Matrix size as 'nArow,nAcol,nBrow,nBcol'. Can repeat.",
    )

    args = parser.parse_args()
    sizes = parse_sizes(args.size)

    results: List[SizeResult] = []
    for size in sizes:
        runs: List[RunResult] = []
        for _ in range(args.runs):
            runs.append(run_once(args.mpirun, args.exe, args.np, size))
        size_result = SizeResult(*size, runs=runs)
        results.append(size_result)
        mpi, seq, cblas = size_result.mean()
        print(
            "Size {size}: MPI={mpi}, Seq={seq}, CBLAS={cblas}".format(
                size=",".join(str(x) for x in size),
                mpi="" if mpi is None else f"{mpi:.6f}",
                seq="" if seq is None else f"{seq:.6f}",
                cblas="" if cblas is None else f"{cblas:.6f}",
            )
        )

    write_csv(args.out, results)
    print(f"Wrote {len(results)} size entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
