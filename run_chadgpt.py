"""Entry point RLBot runs for the Python ChadGPT bot (see bot/ChadGPT.bot.toml).

    python run_chadgpt.py --version-dir bot [--device cuda] [--threads auto|N] [--precision fp32|bf16]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch  # noqa: F401
except ImportError:  # the RLBot botpack's build leaves torch out: use the pack's shared torch-archive (CPU PyTorch)
    for d in [p for base in (Path.cwd(), Path(sys.executable).parent) for p in (base, *base.parents)[:5]]:
        if (d / 'torch-archive' / 'torch').is_dir():
            sys.path.insert(0, str(d / 'torch-archive'))
            break

from chadgpt.agent import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
