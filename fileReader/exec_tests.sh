#!/bin/bash

function usage {
    echo "Usage: $0 -f|--files <FILES> -l|--logfile <LOGFILE>"
    echo "<FILES>: path to the folder containing input files in e.g. .lp or .mps format"
    echo "<LOGFILE>L path to the log file to save output"
    exit 1
}

# exactly two params required (both with mandatory argumetns)
if [ "$#" -ne 4 ]; then usage ; fi

# read the options
OPTS=$(getopt --options "f:,l:" --long "files:,logfile:" --name "$0" -- "$@")
if [ $? != 0 ] ; then usage ; fi
eval set -- "$OPTS"

# extract options and their arguments into variables.
while true ; do
  case "$1" in
    -f | --files )
      FILES="$2"
      shift 2
      ;;
    -l | --logfile )
      LOGFILE="$2"
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    *)
      echo "Internal error!"
      exit 1
      ;;
  esac
done

for filename in $FILES*; do
    printf "\n\n === $filename ===\n"
    python3 -u run_propagation.py -f $filename 2>&1 | tee -a $LOGFILE
done