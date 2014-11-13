#!/bin/sh

. ~/scripts/config/setup.sh

. ~/scripts/libs/shflags

# this is the main script used to execute multiple dev commands.

# Command 1: 
# devenv.sh commit projectname "message" -n

# DEFINE_boolean 'nolog' false 'Do not log the commit message' 'n'
# DEFINE_boolean 'rebase' true 'Rebase the repository from SVN before commit' 'b'
# DEFINE_boolean 'addfiles' true 'Add the untracked files before commit' 'a'
# DEFINE_boolean 'noreflection' false 'Prevent generation of reflection with sgt' 'r'
# DEFINE_boolean 'forceconfig' false 'Force cmake configuration' 'f'
# DEFINE_boolean 'gc' false 'Collect GIT garbages'

# DEFINE_string 'branch' '' 'branch where to push' 'b'

# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

cmd=$1

case $cmd in
cmdlist | help)
	# output the list of commands available and exit:
	echo "List of available commands:"
	echo "  -> format_date pattern"
	echo "  -> build_octave_plug [test_load | train_bp]"
	echo "  -> build_all"  # build the x86 and x64 versions
	;;

format_date)
	. "$root_path_nervtech/scripts/nerv_utils.sh"

	format_dataset_date "$2"
  ;;

build_octave_plug)
	. "$root_path_nervtech/scripts/nerv_utils.sh"

	build_octave_plugin "$2" "x86"
	build_octave_plugin "$2" "x64"
	;;

build_all)
	. "$root_path_nervtech/scripts/nerv_utils.sh"

	dev.sh build nervtech win32_vs12
	dev.sh build nervtech win64_vs12
	build_octave_plugin "test_load" "x86"
	build_octave_plugin "test_load" "x64"
	build_octave_plugin "train_bp" "x86"
	build_octave_plugin "train_bp" "x64"
	build_octave_plugin "show_cuda_info" "x86"
	build_octave_plugin "show_cuda_info" "x64"
	build_octave_plugin "nn_cost_function" "x86"
	build_octave_plugin "nn_cost_function" "x64"

	;;


esac
