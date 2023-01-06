#!/bin/bash

while getopts 'v' OPTION; do
  case "$OPTION" in
    v)
      export VERBOSE="TRUE"
      ;;
  esac
done
shift "$(($OPTIND -1))"
source ./env/bashrc.summit
make main
