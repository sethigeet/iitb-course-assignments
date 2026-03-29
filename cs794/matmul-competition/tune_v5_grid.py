#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "v5_3d_tiling.cu"
DEFAULT_OUT_DIR = ROOT / "autotune_v5"

TILE = 64
# The current kernel assumes float4 vectorization in N, so TN stays fixed at 4.
TN = 4
VEC = 4
WARP_SIZE = 32

PADDED_A_FASTPATH = """      if (a_row < N && a_col + (VEC - 1) < N) {
        const float4 a_vec =
            reinterpret_cast<const float4 *>(&A[a_row * N + a_col])[0];
        As[innerRow * A_TILE_STRIDE + col4 + 0] = a_vec.x;
        As[innerRow * A_TILE_STRIDE + col4 + 1] = a_vec.y;
        As[innerRow * A_TILE_STRIDE + col4 + 2] = a_vec.z;
        As[innerRow * A_TILE_STRIDE + col4 + 3] = a_vec.w;
      } else {"""

UNPADDED_A_FASTPATH = """      if (a_row < N && a_col + (VEC - 1) < N) {
        reinterpret_cast<float4 *>(&As[innerRow * A_TILE_STRIDE + col4])[0] =
            reinterpret_cast<const float4 *>(&A[a_row * N + a_col])[0];
      } else {"""


@dataclass(frozen=True)
class Config:
    thread_tile_m: int
    thread_tile_n: int
    warp_tile_m: int
    warp_tile_n: int
    block_dim_x: int
    pad_a_tile: bool

    @property
    def warps_m(self) -> int:
        return TILE // self.warp_tile_m

    @property
    def warps_n(self) -> int:
        return TILE // self.warp_tile_n

    @property
    def threads_per_block(self) -> int:
        return self.warps_m * self.warps_n * WARP_SIZE

    @property
    def block_dim_y(self) -> int:
        return self.threads_per_block // self.block_dim_x

    @property
    def a_tile_stride(self) -> int:
        return TILE + 1 if self.pad_a_tile else TILE

    @property
    def loads_per_thread(self) -> int:
        return (TILE * (TILE // VEC)) // self.threads_per_block

    @property
    def warp_aspect_ratio(self) -> float:
        return max(self.warp_tile_m, self.warp_tile_n) / min(
            self.warp_tile_m, self.warp_tile_n
        )

    @property
    def slug(self) -> str:
        pad = "padA1" if self.pad_a_tile else "padA0"
        return (
            f"tm{self.thread_tile_m}_tn{self.thread_tile_n}"
            f"_wm{self.warp_tile_m}_wn{self.warp_tile_n}"
            f"_bx{self.block_dim_x}_{pad}"
        )


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


def parse_bool_list(raw: str) -> list[bool]:
    mapping = {
        "0": False,
        "1": True,
        "false": False,
        "true": True,
        "no": False,
        "yes": True,
    }
    values = []
    for token in raw.split(","):
        key = token.strip().lower()
        if key not in mapping:
            raise argparse.ArgumentTypeError(
                f"Unsupported boolean token '{token}'. Use 0/1 or false/true."
            )
        values.append(mapping[key])
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one boolean value.")
    return values


def build_configs(
    tm_values: Iterable[int],
    warp_m_values: Iterable[int],
    block_x_values: Iterable[int],
    pad_values: Iterable[bool],
    max_warp_aspect: float,
) -> list[Config]:
    configs: list[Config] = []
    for tm, warp_m, block_x, pad_a_tile in itertools.product(
        tm_values, warp_m_values, block_x_values, pad_values
    ):
        warp_area = WARP_SIZE * tm * TN
        if warp_area % warp_m != 0:
            continue

        warp_n = warp_area // warp_m
        cfg = Config(
            thread_tile_m=tm,
            thread_tile_n=TN,
            warp_tile_m=warp_m,
            warp_tile_n=warp_n,
            block_dim_x=block_x,
            pad_a_tile=pad_a_tile,
        )
        if is_valid(cfg) and cfg.warp_aspect_ratio <= max_warp_aspect:
            configs.append(cfg)
    return configs


def is_valid(cfg: Config) -> bool:
    if TILE % cfg.thread_tile_m != 0:
        return False
    if cfg.thread_tile_n != TN:
        return False
    if cfg.warp_tile_m <= 0 or cfg.warp_tile_n <= 0:
        return False
    if TILE % cfg.warp_tile_m != 0 or TILE % cfg.warp_tile_n != 0:
        return False
    if cfg.warp_tile_m % cfg.thread_tile_m != 0:
        return False
    if cfg.warp_tile_n % cfg.thread_tile_n != 0:
        return False
    if (
        cfg.warp_tile_m * cfg.warp_tile_n
        != WARP_SIZE * cfg.thread_tile_m * cfg.thread_tile_n
    ):
        return False
    if cfg.threads_per_block != 256:
        return False
    if cfg.block_dim_x <= 0 or cfg.block_dim_x > cfg.threads_per_block:
        return False
    if cfg.threads_per_block % cfg.block_dim_x != 0:
        return False
    total_float4_loads = TILE * (TILE // VEC)
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
    rendered = replace_const(rendered, "THREAD_TILE_M", str(cfg.thread_tile_m))
    rendered = replace_const(rendered, "THREAD_TILE_N", str(cfg.thread_tile_n))
    rendered = replace_const(rendered, "WARP_TILE_M", str(cfg.warp_tile_m))
    rendered = replace_const(rendered, "WARP_TILE_N", str(cfg.warp_tile_n))
    rendered = replace_const(rendered, "BLOCK_DIM_X", str(cfg.block_dim_x))
    rendered = replace_const(rendered, "BLOCK_DIM_Y", "THREADS_PER_BLOCK / BLOCK_DIM_X")
    rendered = replace_const(rendered, "A_TILE_STRIDE", str(cfg.a_tile_stride))
    rendered = replace_const(rendered, "LOADS_PER_THREAD", str(cfg.loads_per_thread))

    if PADDED_A_FASTPATH in rendered:
        current_a_path = PADDED_A_FASTPATH
    elif UNPADDED_A_FASTPATH in rendered:
        current_a_path = UNPADDED_A_FASTPATH
    else:
        raise ValueError("Could not locate the A-tile fast path block.")

    target_a_path = PADDED_A_FASTPATH if cfg.pad_a_tile else UNPADDED_A_FASTPATH
    rendered = rendered.replace(current_a_path, target_a_path, 1)
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
    variant_name = f"v5_grid_{cfg.slug}.cu"
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
                "thread_tile_m",
                "thread_tile_n",
                "warp_tile_m",
                "warp_tile_n",
                "warp_aspect_ratio",
                "threads_per_block",
                "block_dim_x",
                "block_dim_y",
                "pad_a_tile",
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
                    result.config.thread_tile_m,
                    result.config.thread_tile_n,
                    result.config.warp_tile_m,
                    result.config.warp_tile_n,
                    f"{result.config.warp_aspect_ratio:.3f}",
                    result.config.threads_per_block,
                    result.config.block_dim_x,
                    result.config.block_dim_y,
                    int(result.config.pad_a_tile),
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
            f"TM={cfg.thread_tile_m} TN={cfg.thread_tile_n} "
            f"WM={cfg.warp_tile_m} WN={cfg.warp_tile_n} "
            f"threads={cfg.threads_per_block} bx={cfg.block_dim_x} "
            f"padA={int(cfg.pad_a_tile)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid search v5 warp-tiling hyperparameters via ./run.sh <file> ncu."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Kernel source to use as the template. Defaults to v5_3d_tiling.cu.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated variants, logs, and result tables.",
    )
    parser.add_argument(
        "--thread-tile-m",
        type=parse_int_list,
        default=parse_int_list("2,4,8"),
        help="Comma-separated TM candidates. TN stays fixed at 4 in this kernel.",
    )
    parser.add_argument(
        "--warp-tile-m",
        type=parse_int_list,
        default=parse_int_list("8,16,32,64"),
        help="Comma-separated WARP_TILE_M candidates. WARP_TILE_N is derived automatically.",
    )
    parser.add_argument(
        "--block-dim-x",
        type=parse_int_list,
        default=parse_int_list("8,16,32"),
        help="Comma-separated BLOCK_DIM_X candidates. BLOCK_DIM_Y is derived automatically.",
    )
    parser.add_argument(
        "--pad-a-tile",
        type=parse_bool_list,
        default=parse_bool_list("0,1"),
        help="Comma-separated options for A-tile padding. Use 0,1 or false,true.",
    )
    parser.add_argument(
        "--max-warp-aspect",
        type=float,
        default=2.0,
        help=(
            "Filter out very rectangular warp tiles. "
            "Default 2.0 keeps the search focused on square-matrix-friendly shapes."
        ),
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
        tm_values=args.thread_tile_m,
        warp_m_values=args.warp_tile_m,
        block_x_values=args.block_dim_x,
        pad_values=args.pad_a_tile,
        max_warp_aspect=args.max_warp_aspect,
    )
    configs.sort(
        key=lambda cfg: (
            cfg.thread_tile_m,
            cfg.warp_tile_m,
            cfg.warp_tile_n,
            cfg.block_dim_x,
            int(cfg.pad_a_tile),
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
            f"  {cfg.slug}: threads={cfg.threads_per_block}, "
            f"block=({cfg.block_dim_x},{cfg.block_dim_y}), "
            f"warp_aspect={cfg.warp_aspect_ratio:.2f}, "
            f"loads/thread={cfg.loads_per_thread}"
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
