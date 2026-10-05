# Sourced by the bench scripts (§14.3/§14.4). Results go to one markdown table per run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
E=${BENI_ENGINES:-/ssd/beni/engines}
M=${BENI_MODELS:-/ssd/beni/models}
OUT=${BENCH_OUT:-/ssd/beni/logs/bench-$(date +%Y%m%d).md}
PYHOST=/usr/bin/python3                                   # py3.6: tegrastats summaries
PYVENV=${BENI_VENV:-$ROOT/.venv38}/bin/python             # py3.8 (05_py38_venv.sh): memory retrieval
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || printf '| Test | Metric | Result | Target |\n|---|---|---|---|\n' > "$OUT"
row() { printf '| %s | %s | %s | %s |\n' "$1" "$2" "$3" "${4:-}" | tee -a "$OUT"; }
