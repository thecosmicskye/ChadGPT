"""Sets up the Python environment on first use and starts the bot (what RLBot runs, through bot/run_chadgpt.cmd).

    python bootstrap.py --setup-only            create .venv and install everything, then exit
    python bootstrap.py [run_chadgpt.py options] set up if needed, then run the bot

The environment lives in .venv next to this file. PyTorch comes from the PyTorch package index: a CUDA build when the
NVIDIA GPU and driver can run one (so the network runs on the GPU in bfloat16, as the original bot did), otherwise the
CPU build. CHADGPT_TORCH_INDEX=cpu|cu126|cu130|... forces a build. The requirements are reinstalled when
requirements.txt or the chosen build changes. Two team processes starting together share one install (a lock).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
LOCK = ROOT / '.venv.lock'
MARKER = VENV / 'chadgpt-installed.txt'
TORCH_INDEX = 'https://download.pytorch.org/whl/'
# PyTorch is pinned here rather than in requirements.txt, so the RLBot botpack's build (bob) leaves it out: the pack's
# builds use the shared CPU torch-archive (run_chadgpt.py).
TORCH = 'torch==2.14.0'
LOCK_STALE_SECONDS = 3600


def log(msg):
    print(f'[ChadGPT setup] {msg}', flush=True)


def venv_python():
    return VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def torch_variant():
    forced = os.environ.get('CHADGPT_TORCH_INDEX', '').strip()
    if forced:
        return forced
    if sys.platform == 'darwin':
        return 'pypi'
    smi = shutil.which('nvidia-smi')
    if not smi:
        return 'cpu'
    try:
        out = subprocess.run([smi, '--query-gpu=driver_version,compute_cap', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        driver, cap = (f.strip() for f in out[0].split(','))
        major, cap = int(driver.split('.')[0]), float(cap)
    except Exception:  # noqa: BLE001 - no usable driver: CPU build
        return 'cpu'
    # The pinned PyTorch comes as CUDA 12.6 builds (GPUs up to compute capability 9.0, driver 525+) and CUDA 13.0
    # builds (compute capability 7.5 and newer, including the RTX 50-series, driver 580+).
    if major >= 580 and cap >= 7.5:
        return 'cu130'
    if major >= 525 and cap < 10:
        return 'cu126'
    log(f'no CUDA build of the pinned PyTorch runs on this GPU (compute capability {cap}) with driver {driver}; '
        'using the CPU build. With NVIDIA driver 580 or newer, setup switches to the GPU build on the next start.')
    return 'cpu'


def wanted_marker(variant):
    req = (ROOT / 'requirements.txt').read_bytes()
    return f'{hashlib.sha256(req).hexdigest()} {TORCH} {variant}\n'


def installed():
    variant = torch_variant()
    return venv_python().is_file() and MARKER.is_file() and MARKER.read_text() == wanted_marker(variant)


def pip(*args):
    cmd = [str(venv_python()), '-m', 'pip', '--disable-pip-version-check', 'install', '--progress-bar', 'off', *args]
    log('pip install ' + ' '.join(args))
    subprocess.run(cmd, check=True)


def install():
    variant = torch_variant()
    if not venv_python().is_file():
        log(f'creating {VENV} with Python {sys.version.split()[0]}')
        subprocess.run([sys.executable, '-m', 'venv', str(VENV)], check=True)
    log(f'installing PyTorch ({variant} build; the CUDA build is a large download, please wait)')
    pip(TORCH, *(['--index-url', TORCH_INDEX + variant] if variant != 'pypi' else []))
    log('installing the other requirements')
    pip('-r', str(ROOT / 'requirements.txt'))
    MARKER.write_text(wanted_marker(variant))
    log('done')


def ensure_installed():
    if installed():
        return
    start = time.time()
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - LOCK.stat().st_mtime > LOCK_STALE_SECONDS:
                    LOCK.rmdir()
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start < 1.0:
                log('another ChadGPT process is installing; waiting for it')
            time.sleep(2.0)
            if installed():
                return
    try:
        if not installed():
            install()
    finally:
        LOCK.rmdir()


def main():
    if sys.version_info[:2] < (3, 11):
        sys.exit(f'ChadGPT needs Python 3.11 or newer (3.12 recommended); this is {sys.version.split()[0]}')
    args = sys.argv[1:]
    setup_only = '--setup-only' in args
    args = [a for a in args if a != '--setup-only']
    ensure_installed()
    if setup_only:
        return 0
    return subprocess.call([str(venv_python()), str(ROOT / 'run_chadgpt.py'), *args])


if __name__ == '__main__':
    sys.exit(main())
