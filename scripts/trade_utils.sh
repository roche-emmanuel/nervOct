#!/bin/sh

. ~/scripts/libs/base_utils.sh

datapath="$root_path_nervtrade/data"

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
		echo "Found valid file: '$filename'"
		format_date_cb "$filename"
	fi 
}

format_dataset_date()
{
	cd "$datapath"

	foreachFile '\.csv$' checkInputPattern	"$1" # last argument is the pattern.
}
