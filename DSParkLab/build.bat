@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d "%~dp0"

echo Building DSParkLab...

cl /std:c++20 /O2 /W4 /WX- /EHsc /DUNICODE /DNOMINMAX ^
    /I.. /Ivendor /Ivendor/imgui ^
    main.cpp ^
    vendor/imgui/imgui.cpp ^
    vendor/imgui/imgui_draw.cpp ^
    vendor/imgui/imgui_tables.cpp ^
    vendor/imgui/imgui_widgets.cpp ^
    vendor/imgui/imgui_impl_win32.cpp ^
    vendor/imgui/imgui_impl_dx11.cpp ^
    /Fe:DSParkLab.exe /nologo ^
    /link d3d11.lib d3dcompiler.lib user32.lib gdi32.lib ole32.lib comdlg32.lib

if %ERRORLEVEL% == 0 (
    echo Build OK: DSParkLab.exe
    del *.obj >nul 2>&1
) else (
    echo BUILD FAILED
)
