#!/usr/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wkspc_dir="$(cd "$script_dir/.." && pwd)"
source "$wkspc_dir/conda.env"
set -x
$CONDA_EXE run --prefix "$CONDA_PREFIX" "$@"
