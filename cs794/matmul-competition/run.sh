#!/bin/sh

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_file.cu> [ncu]"
    echo "  ncu  - profile kernel with NVIDIA Nsight Compute"
    exit 1
fi

source_file="$1"
bin_name="${1%.cu}"
REMOTE="sutra"
REMOTE_DIR="code/cuda-kernels"

scp "$source_file" "$REMOTE:~/$REMOTE_DIR/$source_file"
ssh "$REMOTE" "cd $REMOTE_DIR && /usr/local/cuda/bin/nvcc -O3 -lineinfo -o $bin_name $source_file -lcublas"

if [ "$2" = "ncu" ]; then
    echo "=== Correctness Tests (unprofiled) ==="
    ssh "$REMOTE" "~/$REMOTE_DIR/$bin_name"

    echo ""
    echo "========================================="
    echo "  NCU Profiling: 1 warmup + 3 measured"
    echo "========================================="
    # "benchmark" mode launches only: 1 warmup kernel + 3 measured kernels
    # --launch-skip 1  → skip the warmup kernel
    # --launch-count 3 → profile the 3 measured kernels
    ssh "$REMOTE" "cd $REMOTE_DIR && sudo /usr/local/cuda/bin/ncu --launch-skip 1 --launch-count 3 \
        --metrics gpu__time_duration.sum \
        ./$bin_name benchmark" 2>&1 | tee /tmp/ncu_output.txt

    echo ""
    echo "=== Timing Summary ==="
    awk '
    { gsub(/\r/, "") }
    /gpu__time_duration\.sum/ {
        val = $NF
        gsub(",", "", val)
        if (val ~ /^[0-9]/) {
            n++
            sum += val
            unit = $(NF-1)
            printf "  Kernel %d: %s %s\n", n, val, unit
        }
    }
    END {
        if (n > 0) {
            avg = sum / n
            if (unit == "ns" || unit == "nsecond")      avg_ms = avg / 1e6
            else if (unit == "us" || unit == "usecond") avg_ms = avg / 1e3
            else if (unit == "ms" || unit == "msecond") avg_ms = avg
            else if (unit == "s" || unit == "second")   avg_ms = avg * 1000
            else                                        avg_ms = avg
            gflops = (2.0 * 4096 * 4096 * 4096) / (avg_ms * 1e6)
            printf "  ─────────────────────\n"
            printf "  Avg kernel time: %.3f ms\n", avg_ms
            printf "  GFLOPS:          %.2f\n", gflops
        } else {
            print "  Could not parse ncu output — check raw output above."
        }
    }' /tmp/ncu_output.txt
else
    ssh "$REMOTE" "~/$REMOTE_DIR/$bin_name"
fi
