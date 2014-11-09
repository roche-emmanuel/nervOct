#!/bin/sh

. ~/scripts/libs/base_utils.sh

datapath="$root_path_nervtech"

cd "$datapath/bin/x86"

root_path=/cygdrive/w/Cloud/LiberKey/MyApps/octave-3.8.2-x86-mingw
OCT_HOME=`cygpath -w "$root_path"`
PATH=$root_path/bin:$PATH
target="test_load"
file=`cygpath -w "$datapath/sources/oct_$target/$target.cpp"`

g++ -c -I`cygpath -w "$root_path/include/octave-3.8.2"` -I`cygpath -w "$root_path/include/octave-3.8.2/octave"` -I`cygpath -w "$root_path/include"` -mieee-fp -g -O2 -pthread $file -o $target.o
g++ -shared -Wl,--export-all-symbols -Wl,--enable-auto-import -Wl,--enable-runtime-pseudo-reloc -o $target.oct $target.o -L`cygpath -w "$root_path/lib/octave/3.8.2"` -L`cygpath -w "$root_path/lib"` -loctinterp -loctave -Wl,--export-all-symbols

rm -Rf *.o
