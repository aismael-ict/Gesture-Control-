@echo off
title Hand Control Next-Gen 2.0
echo Starting Hand Control Next-Gen...
python hand_control_nextgen.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hand Control exited with error code %ERRORLEVEL%.
    pause
)

