@echo OFF

IF NOT "%*"=="" (
    echo Usage error: metavision_studio should be launched without command line option. Help, file selections and other options are available through the Graphical User Interface
    exit /B
)

setlocal ENABLEEXTENSIONS
set KEY_NAME="HKEY_LOCAL_MACHINE\Software\Prophesee"
set VALUE_NAME=INSTALL_PATH
FOR /F "usebackq skip=2 tokens=2,*" %%A IN (`REG QUERY %KEY_NAME% /v %VALUE_NAME% 2^>nul`) DO (
    set MV_INSTALL_PATH=%%B
)
set MV_STUDIO_CLIENT_PATH=%MV_INSTALL_PATH%\share\metavision\apps\metavision_studio\internal\client\Metavision Studio
set MV_STUDIO_SERVER_PATH=%MV_INSTALL_PATH%\share\metavision\apps\metavision_studio\internal\server\metavision_studio_server
start /b "" "%MV_STUDIO_CLIENT_PATH%" --main-args --server-path "%MV_STUDIO_SERVER_PATH%" 
