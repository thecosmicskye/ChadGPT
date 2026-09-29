@echo off
rem One-time setup for ChadGPT: creates .venv here and installs PyTorch (CUDA build when an NVIDIA GPU is present),
rem and RLBot's Python interface. Optional: the bot also does this on its first start.
call "%~dp0bot\run_chadgpt.cmd" --setup-only
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (echo ChadGPT is ready.) else (echo Setup failed with exit code %RC%.)
pause
exit /b %RC%
