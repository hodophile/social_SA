#!/usr/bin/env bash
# Run all four modality examples (mock PLM, no GPU needed).
set -euo pipefail
cd "$(dirname "$0")"

for t in video audio image text; do
    echo "================ $t ================"
    python3 "test_$t.py"
done

echo
echo "All modality examples PASSED"
