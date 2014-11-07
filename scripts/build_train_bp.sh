#!/bin/sh

. ~/scripts/libs/base_utils.sh

datapath="$root_path_nervtech"

cd "$datapath/bin/x86"
mkoctfile "$datapath/sources/oct_train_bp/train_bp.cpp"
rm -Rf *.o

