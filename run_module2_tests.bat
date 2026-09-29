@echo off
setlocal
chcp 65001 >nul
set "TEST_PYTHON=D:\Anaconda\envs\AI_common\python.exe"
set "TEST_EXIT_CODE=2"

if not exist "%TEST_PYTHON%" (
    if exist "%LOCALAPPDATA%\Temp\stqa-m2-venv\Scripts\python.exe" (
        set "TEST_PYTHON=%LOCALAPPDATA%\Temp\stqa-m2-venv\Scripts\python.exe"
        echo AI_common was not found. Using the local review interpreter:
        echo %LOCALAPPDATA%\Temp\stqa-m2-venv\Scripts\python.exe
    )
)

if not exist "%TEST_PYTHON%" (
    echo ERROR: Python interpreter not found.
    echo Expected D:\Anaconda\envs\AI_common\python.exe
    goto :finish
)

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the repository directory.
    goto :finish
)

echo Running module 2 CPU tests ^(currently 51 cases^)...
"%TEST_PYTHON%" -m pytest -c pytest.ini tests/module2/
set "TEST_EXIT_CODE=%ERRORLEVEL%"
popd

:finish
echo.
echo Test exit code: %TEST_EXIT_CODE%
echo 0 = all passed; 1 = test failures; 2 or higher = execution/collection issue.
if /I not "%~1"=="--no-pause" pause
exit /b %TEST_EXIT_CODE%
