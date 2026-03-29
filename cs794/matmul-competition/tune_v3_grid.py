#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "v3_vectorized_loads.cu"
DEFAULT_OUT_DIR = ROOT / "autotune_v3"
VEC = 4


@dataclass(frozen=True)
class Config:
    tile: int
    reg_tile: int

    @property
    def block_dim(self) -> int:
        return self.tile // self.reg_tile

    @property
    def threads_per_block(self) -> int:
        return self.block_dim * self.block_dim

    @property
    def loads_per_thread(self) -> int:
        return (self.tile * (self.tile // VEC)) // self.threads_per_block

    @property
    def shared_bytes(self) -> int:
        return 2 * self.tile * self.tile * 4

    @property
    def slug(self) -> str:
        return f"tile{self.tile}_r{self.reg_tile}"


@dataclass
class RunResult:
    config: Config
    source_file: str
    avg_ms: float | None
    gflops: float | None
    correctness_ok: bool
    status: str
    output_log: str


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def build_configs(
    tile_values: list[int],
    reg_values: list[int],
    max_threads: int,
    max_shared_kb: int,
) -> list[Config]:
    configs: list[Config] = []
    for tile in tile_values:
        for reg_tile in reg_values:
            cfg = Config(tile=tile, reg_tile=reg_tile)
            if is_valid(cfg, max_threads=max_threads, max_shared_kb=max_shared_kb):
                configs.append(cfg)
    return configs


def is_valid(cfg: Config, max_threads: int, max_shared_kb: int) -> bool:
    if cfg.tile <= 0 or cfg.reg_tile <= 0:
        return False
    if cfg.tile % VEC != 0:
        return False
    if cfg.reg_tile % VEC != 0:
        return False
    if cfg.tile % cfg.reg_tile != 0:
        return False
    if cfg.threads_per_block > max_threads:
        return False
    if cfg.shared_bytes > max_shared_kb * 1024:
        return False
    total_float4_loads = cfg.tile * (cfg.tile // VEC)
    if total_float4_loads % cfg.threads_per_block != 0:
        return False
    return True


def replace_const(source: str, name: str, value: str) -> str:
    pattern = rf"^\s*constexpr int {name} = .*?;\s*(?://.*)?$"
    updated, count = re.subn(
        pattern, f"constexpr int {name} = {value};", source, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"Could not uniquely replace constant '{name}'.")
    return updated


def render_variant(source: str, cfg: Config) -> str:
    rendered = source
    rendered = replace_const(rendered, "TILE", str(cfg.tile))
    rendered = replace_const(rendered, "R", str(cfg.reg_tile))
    rendered = replace_const(rendered, "LOADS_PER_THREAD", str(cfg.loads_per_thread))

    return rendered


def parse_run_output(output: str) -> tuple[bool, float | None, float | None]:
    correctness_ok = "All correctness tests PASSED." in output
    avg_match = re.search(r"Avg kernel time:\s+([0-9.]+)\s+ms", output)
    gflops_match = re.search(r"GFLOPS:\s+([0-9.]+)", output)
    avg_ms = float(avg_match.group(1)) if avg_match else None
    gflops = float(gflops_match.group(1)) if gflops_match else None
    return correctness_ok, avg_ms, gflops


def run_config(
    cfg: Config,
    source_text: str,
    variants_dir: Path,
    logs_dir: Path,
    run_script: Path,
) -> RunResult:
    variant_name = f"v3_grid_{cfg.slug}.cu"
    variant_path = variants_dir / variant_name
    launch_path = ROOT / variant_name
    rendered = render_variant(source_text, cfg)
    variant_path.write_text(rendered)
    launch_path.write_text(rendered)

    cmd = [str(run_script), variant_name, "ncu"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        combined_output = completed.stdout + completed.stderr
    finally:
        launch_path.unlink(missing_ok=True)

    log_path = logs_dir / f"{variant_path.stem}.log"
    log_path.write_text(combined_output)

    correctness_ok, avg_ms, gflops = parse_run_output(combined_output)
    status = (
        "ok"
        if completed.returncode == 0 and correctness_ok and avg_ms is not None
        else "failed"
    )
    return RunResult(
        config=cfg,
        source_file=str(variant_path.relative_to(ROOT)),
        avg_ms=avg_ms,
        gflops=gflops,
        correctness_ok=correctness_ok,
        status=status,
        output_log=str(log_path.relative_to(ROOT)),
    )


def save_results(results: list[RunResult], out_dir: Path) -> None:
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    json_payload = []
    for result in results:
        row = asdict(result)
        row["config"] = asdict(result.config)
        json_payload.append(row)
    json_path.write_text(json.dumps(json_payload, indent=2))

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "avg_ms",
                "gflops",
                "correctness_ok",
                "tile",
                "reg_tile",
                "block_dim",
                "threads_per_block",
                "loads_per_thread",
                "shared_bytes",
                "source_file",
                "output_log",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.status,
                    result.avg_ms,
                    result.gflops,
                    result.correctness_ok,
                    result.config.tile,
                    result.config.reg_tile,
                    result.config.block_dim,
                    result.config.threads_per_block,
                    result.config.loads_per_thread,
                    result.config.shared_bytes,
                    result.source_file,
                    result.output_log,
                ]
            )


def print_summary(results: list[RunResult], top_k: int) -> None:
    successful = [r for r in results if r.status == "ok" and r.avg_ms is not None]
    failed = [r for r in results if r.status != "ok"]
    successful.sort(
        key=lambda row: row.avg_ms if row.avg_ms is not None else float("inf")
    )

    print(
        f"Completed {len(results)} configs: {len(successful)} succeeded, {len(failed)} failed."
    )
    if not successful:
        return

    print("")
    print(f"Top {min(top_k, len(successful))} configs by Avg kernel time:")
    for index, result in enumerate(successful[:top_k], start=1):
        cfg = result.config
        print(
            f"{index:2d}. {result.avg_ms:.3f} ms  {result.gflops:.2f} GFLOPS  "
            f"TILE={cfg.tile} R={cfg.reg_tile} "
            f"block={cfg.block_dim}x{cfg.block_dim} "
            f"threads={cfg.threads_per_block}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid search v3 vectorized-load hyperparameters via ./run.sh <file> ncu."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Kernel source to use as the template. Defaults to v3_vectorized_loads.cu.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated variants, logs, and result tables.",
    )
    parser.add_argument(
        "--tile",
        type=parse_int_list,
        default=parse_int_list("32,64,96"),
        help="Comma-separated TILE candidates for the square NxN benchmark.",
    )
    parser.add_argument(
        "--reg-tile",
        type=parse_int_list,
        default=parse_int_list("4,8,12,16"),
        help="Comma-separated R candidates. Must be multiples of 4 for float4 stores.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1024,
        help="Maximum threads per block to allow in the search.",
    )
    parser.add_argument(
        "--max-shared-kb",
        type=int,
        default=96,
        help="Maximum dynamic shared memory per block, in KiB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of valid configs to run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best configs to print at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the valid config list without launching benchmarks.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.exists():
        print(f"Template source not found: {source_path}", file=sys.stderr)
        return 1

    source_text = source_path.read_text()
    run_script = ROOT / "run.sh"
    if not run_script.exists():
        print(f"run.sh not found at {run_script}", file=sys.stderr)
        return 1

    configs = build_configs(
        tile_values=args.tile,
        reg_values=args.reg_tile,
        max_threads=args.max_threads,
        max_shared_kb=args.max_shared_kb,
    )
    configs.sort(
        key=lambda cfg: (
            cfg.tile,
            cfg.reg_tile,
            cfg.threads_per_block,
        )
    )
    if args.limit > 0:
        configs = configs[: args.limit]

    if not configs:
        print("No valid configs found for the requested search space.", file=sys.stderr)
        return 1

    print(f"Found {len(configs)} valid configs.")
    for cfg in configs:
        print(
            f"  {cfg.slug}: block={cfg.block_dim}x{cfg.block_dim}, "
            f"threads={cfg.threads_per_block}, loads/thread={cfg.loads_per_thread}, "
            f"shared={cfg.shared_bytes // 1024}KiB"
        )
    if args.dry_run:
        return 0

    out_dir = args.out_dir.resolve()
    variants_dir = out_dir / "variants"
    logs_dir = out_dir / "logs"
    variants_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for index, cfg in enumerate(configs, start=1):
        print("")
        print(f"[{index}/{len(configs)}] Running {cfg.slug}")
        result = run_config(cfg, source_text, variants_dir, logs_dir, run_script)
        results.append(result)
        if result.status == "ok":
            print(f"  -> {result.avg_ms:.3f} ms, {result.gflops:.2f} GFLOPS")
        else:
            print(f"  -> failed (see {result.output_log})")
        save_results(results, out_dir)

    print("")
    print_summary(results, args.top_k)
    print(f"\nSaved detailed results under {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
