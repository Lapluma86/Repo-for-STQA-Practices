@echo off
setlocal
set "TEST_PYTHON=D:\Anaconda\envs\AI_common\python.exe"
set "TEST_EXIT_CODE=2"

if not exist "%TEST_PYTHON%" (
    echo ERROR: Python interpreter not found: %TEST_PYTHON%
    echo Please check the AI_common Conda environment.
    goto :finish
)

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the repository directory.
    goto :finish
)

echo Running module 1 CPU tests ^(currently 48 cases^)...
"%TEST_PYTHON%" -m pytest -c pytest.ini tests/module1/
set "TEST_EXIT_CODE=%ERRORLEVEL%"
popd

:finish
echo.
echo Test exit code: %TEST_EXIT_CODE%
echo 0 = all passed; 1 = test failures; 2 or higher = execution/collection issue.
if /I not "%~1"=="--no-pause" pause
exit /b %TEST_EXIT_CODE%
