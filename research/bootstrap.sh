#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
command -v uv >/dev/null
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.11.16 --seed
fi
# Separate environment only. No sudo, global pip, existing-environment upgrades,
# robot SDK installation, or automatic license acceptance.
uv pip install --python .venv/bin/python --no-deps \
  -r research/env/desktop-reference.txt \
  --build-constraint research/env/build-constraints.txt \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-deps \
  'setuptools==80.10.2' 'wheel==0.46.3' 'usd-core==26.3'
if [[ ! -d research/deps/IsaacLab/.git ]]; then
  git clone --depth 1 --branch v2.3.2 \
    https://github.com/isaac-sim/IsaacLab.git research/deps/IsaacLab
fi
[[ "$(git -C research/deps/IsaacLab rev-parse HEAD)" == 37ddf626871758333d6ed89cf64ad702aef127d0 ]]
uv pip install --python .venv/bin/python --no-deps \
  -e research/deps/IsaacLab/source/isaaclab \
  -e research/deps/IsaacLab/source/isaaclab_assets \
  -e research/deps/IsaacLab/source/isaaclab_rl \
  -e research/deps/IsaacLab/source/isaaclab_tasks \
  -e imports/SONIC/gear_sonic
echo 'Research environment installed; simulation and training have NOT started.'
