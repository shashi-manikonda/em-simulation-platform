:: 1. Configure Visual Studio Location for Intel oneAPI
:: Intel oneAPI needs to know where VS is to integrate compiler settings
set "VS_BUILD_TOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if not exist "%VS_BUILD_TOOLS%" goto SkipVSConfig
echo [INFO] Found Visual Studio Build Tools.
set "VS2022INSTALLDIR=%VS_BUILD_TOOLS%"
:SkipVSConfig

:: 2. Initialize Intel oneAPI Environment
if not exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" goto SkipOneAPI
echo [INFO] Initializing Intel oneAPI Environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
goto InitVS

:SkipOneAPI
echo [WARNING] Intel oneAPI 'setvars.bat' not found.

:InitVS
:: 3. Initialize Visual Studio Environment (Fallback/Verification)
:: If setvars didn't set up cl.exe, we do it manually.
where cl.exe >nul 2>nul
if %errorlevel% neq 0 goto ManualVS
echo [INFO] Visual Studio Compiler cl.exe is already available.
goto CheckVenv

:ManualVS

if not exist "%VS_BUILD_TOOLS%\Common7\Tools\VsDevCmd.bat" goto SkipVS
echo [INFO] Initializing Visual Studio Build Tools (Manual)...
call "%VS_BUILD_TOOLS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
goto CheckVenv

:CheckVenv
:: 3. Setup Virtual Environment
if exist ".venv" goto UseVenv
echo [INFO] Creating new virtual environment (.venv)...
python -m venv .venv
    
echo [INFO] Activating .venv...
call .venv\Scripts\activate.bat
    
echo [INFO] Installing 'uv' package manager...
python -m pip install uv
goto InstallSandalwood

:UseVenv
echo [INFO] Using existing .venv...
call .venv\Scripts\activate.bat

:InstallSandalwood
:: 4. Install Sandalwood (Rebuilds COSY Backend)
echo.
echo ==========================================
echo [STEP 1/3] Installing Sandalwood (COSY Backend)
echo ==========================================
cd sandalwood
uv pip install -e .[dev]
if %errorlevel% neq 0 (
    echo [ERROR] Sandalwood install failed!
    exit /b %errorlevel%
)
cd ..

:: 5. Reinstall mpi4py (Link against Intel MPI)
echo.
echo ==========================================
echo [STEP 2/3] Reinstalling mpi4py (Linking to Intel MPI)
echo ==========================================
:: Check for Intel MPI Headers (SDK installed?)
if not exist "C:\Program Files (x86)\Intel\oneAPI\mpi\latest\include\mpi.h" goto MissingMpiSdk

:: Force source build to link against environment's MPI
uv pip install --force-reinstall --no-binary=mpi4py mpi4py
if %errorlevel% neq 0 (
    echo [WARNING] mpi4py build failed. MPI functionality may be broken.
    echo Ensure Visual Studio C++ tools are installed.
)
goto InstallEmApp

:MissingMpiSdk
echo [WARNING] Intel MPI SDK headers (mpi.h) NOT FOUND!
echo You likely installed the Intel MPI *Runtime* but not the *Developer Kit*.
echo 'mpi4py' cannot be built without headers. Skipping source build.
echo.
echo To fix: Open Intel oneAPI Installer and add "Intel MPI Library" (Dev).

:InstallEmApp

:: 6. Install EM-Simulation-Platform
echo.
echo ==========================================
echo [STEP 3/3] Installing EM-Simulation-Platform
echo ==========================================
cd em-simulation-platform
uv pip install -e .[dev,benchmark]
cd ..

echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
echo Environment: .venv
echo To start developing, run: call .venv\Scripts\activate
echo.
pause
