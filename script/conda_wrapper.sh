#!/usr/bin/bash

# This script is not supposed to be run directly. Rather, symbolic links to this script are created and called. The
# name of the symbolic link is used to determine the command to run.
source "$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)/run_in_conda.sh" $(basename "$BASH_SOURCE") "$@"
