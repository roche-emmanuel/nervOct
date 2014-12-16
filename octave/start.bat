
set X64_MODE=1

set PATH=W:\Cloud\Dev\Cygwin\bin;%PATH%

if %X64_MODE%==1 goto mode_x64

set PATH=W:\Cloud\Projects\nervtech\bin\x86;%PATH%
start X:\Softwares\octave-3.8.2-x86-mingw\bin\octave-gui.exe
exit

:mode_x64
set PATH=W:\Cloud\Projects\nervtech\bin\x64;%PATH%
start X:\Softwares\octave-3.8.2-x64-mingw\bin\octave-gui.exe

