"""Entry point RLBot runs for the Python ChadGPT bot (see bot/ChadGPT.bot.toml).

    python run_chadgpt.py --version-dir bot [--device cuda] [--threads auto|N] [--precision fp32|bf16]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chadgpt.agent import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
