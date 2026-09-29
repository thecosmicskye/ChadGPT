# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build for the RLBot botpack (bob.toml): one folder, ChadGPT/ChadGPT(.exe) + _internal/ with the weights.
# PyTorch is left out: the pack's builds load its shared CPU torch-archive (run_chadgpt.py), so the standard-library
# modules torch imports are listed here.
a = Analysis(
    ['run_chadgpt.py'],
    pathex=['.'],
    binaries=[],
    datas=[('bot/checkpoint', 'checkpoint'), ('bot/SHA256SUMS', '.'), ('bot/memory/seed.npy', 'memory'),
           ('LICENSE', '.')],
    hiddenimports=['bdb', 'cmath', 'cmd', 'code', 'codeop', 'concurrent.futures', 'ctypes.wintypes', 'dataclasses',
                   'difflib', 'mmap', 'multiprocessing.reduction', 'multiprocessing.resource_sharer', 'pdb',
                   'pickletools', 'pprint', 'timeit', 'unittest.mock', 'uuid'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ChadGPT', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='ChadGPT')
