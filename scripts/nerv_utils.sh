#!/bin/sh

. ~/scripts/libs/base_utils.sh

nervpath="$root_path_nervtech"
datapath="$root_path_nervtech/data"

format_date_cb()
{
	local file="$1"
	if [ "$file" == "" ]; then
		file="../data/EURUSDM1_lite.csv"
	fi

	logInfo "Formatting dataset $file..."
	local basename=`getFileName "$file"`
	# local basename=`echo "$1" | sed -e 's/\\.[^.]*$//g'`

	# -i.bak 
	sed -r "s/^([0-9]+).([0-9]+)\.([0-9]+)\s([0-9]+):([0-9]+)/\1,\2,\3,\4,\5/g" "$file" > "$basename.txt"
}

checkInputPattern()
{
	local filename="$1"
	logInfo "Testing file: $1 with pattern $2"
	# logInfo "Testing file $filename with pattern $2"
	local res=`echo -n $filename | grep "$2"`
	if [[ "$res" == "$filename" ]]; then
		logInfo "Found valid file: '$filename'"
		format_date_cb "$filename"
	fi 
}

format_dataset_date()
{
	cd "$datapath"

	foreachFile '\.csv$' checkInputPattern	"$1" # last argument is the pattern.
}

build_octave_plugin()
{
	# Build a plugin for octave using the plugin name and the desired architecture:
	plugname="$1"
	arch="$2"

	logInfo "Compiling octave plugin $plugname for architecture $arch..."

	cd "$nervpath/bin/$arch"

	octave_path="/cygdrive/x/Softwares/octave-3.8.2-$arch-mingw"
	OCT_HOME=`cygpath -w "$octave_path"`
	OLD_PATH="$PATH"
	PATH="$octave_path/bin:$PATH"
	compiler="g++"
	file=`cygpath -w "$nervpath/sources/oct_$plugname/$plugname.cpp"`

	$compiler -c -I`cygpath -w "$octave_path/include/octave-3.8.2"` -I`cygpath -w "$octave_path/include/octave-3.8.2/octave"` -I`cygpath -w "$octave_path/include"` -mieee-fp -g -O2 -pthread $file -o $plugname.o
	$compiler -shared -Wl,--export-all-symbols -Wl,--enable-auto-import -Wl,--enable-runtime-pseudo-reloc -o $plugname.oct $plugname.o -L`cygpath -w "$octave_path/lib/octave/3.8.2"` -L`cygpath -w "$octave_path/lib"` -loctinterp -loctave -Wl,--export-all-symbols

	rm -Rf *.o
	PATH="$OLD_PATH"

	logInfo "Compilation done."
}
