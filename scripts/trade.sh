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
cmdlist)
	# output the list of commands available and exit:
	echo "List of available commands:"
	echo "  -> format_date pattern"
	;;

format_date)
	. ~/cloud/Projects/nervtrade/scripts/trade_utils.sh

	format_dataset_date "$2"
  ;;

esac
