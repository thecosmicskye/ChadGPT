@echo off
rem ChadGPT for RLBot v5 (what ChadGPT.bot.toml runs).
rem The first start creates ..\.venv and installs the requirements (bootstrap.py); later starts go straight to the bot.
rem Optional: CHADGPT_PYTHON = python.exe to use, CHADGPT_DEVICE = auto (default) / cuda / cpu,
rem CHADGPT_PRECISION = bf16 (default) / fp32.
setlocal
set "ROOT=%~dp0.."
set "PYEXE="
set "PYARG="
if defined CHADGPT_PYTHON set "PYEXE=%CHADGPT_PYTHON%"
if not defined PYEXE if exist "%ROOT%\.venv\Scripts\python.exe" set "PYEXE=%ROOT%\.venv\Scripts\python.exe"
if not defined PYEXE (
  for %%V in (3.12 3.13 3.11) do (
    if not defined PYEXE (
      py -%%V -c "import sys" >nul 2>&1 && (set "PYEXE=py" & set "PYARG=-%%V")
    )
  )
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo ChadGPT needs Python 3.12 ^(https://www.python.org/downloads/^), installed with the "py" launcher. 1>&2
  exit /b 1
)
"%PYEXE%" %PYARG% "%ROOT%\bootstrap.py" --version-dir "%~dp0." %*
exit /b %ERRORLEVEL%
