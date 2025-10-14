#!/usr/bin/bash

# This script is used to set up the development and build environment for the project.

set -eu -o pipefail

! getopt --test
[ ${PIPESTATUS[0]} -ne 4 ] && echo "enhanced getopt not found; cannot continue" && exit 1

parsed_args="$(getopt -o p:e:h:u --long conda-env-path:,conda-home:,help,no-update -- "$@")"
[[ $? -ne 0 ]] && echo "Failed to parse options" && exit 1

eval set -- "$parsed_args"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wkspc_dir="$(cd "$script_dir/.." && pwd)"

conda_env_path="$wkspc_dir/.conda_env"
conda_home=~/miniconda3
update_conda=true

function usage
{
    cat <<EOF
Usage: $0 [OPTIONS] -- [CONDA_OPTIONS]
Set up the development environment for the project.
Usage:
  $0 [OPTIONS]
Options:
  -p, --conda-env-path <path> Path to the conda environment; default: $conda_env_path
  -e, --conda-home <path>     Path to the base conda installation directory; default: $conda_home
  -h, --help                  Show this help message
  -u, --no-update             Do not update the conda environment; default: update is performed
EOF
}

while [[ $1 != -- ]]
do
    case "$1" in
        -p|--conda-env-path) conda_env="$2"; shift ;;
        -e|--conda-home) conda_home="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        -u|--no-update) update_conda=false ;;
    esac
    shift
done
shift # Remove the '--' argument

[[ ! -x "$conda_home/bin/conda" ]] && echo "Conda not found at $conda_home" && exit 3
conda_bin="$conda_home/bin/conda"

conda_args=(--yes --channel conda-forge --prefix $conda_env_path --file $wkspc_dir/dev-conda-requirements.txt)
if [[ ! -d "$conda_env_path" ]]
then
    echo "Creating conda environment at $conda_env_path"
    $conda_bin create "${conda_args[@]}"
else
    if $update_conda
    then
        echo "Updating conda environment at $conda_env_path"
        $conda_bin install --all "${conda_args[@]}"
    else
        echo "Skipping conda environment update at $conda_env_path"
    fi
fi

echo "Creating bash environment file at $wkspc_dir/conda.env"
cat <<EOF > "$wkspc_dir/conda.env"
CONDA_PREFIX=$conda_env_path
CONDA_EXE=$conda_bin
EOF

echo "Creating conda wrapper scripts in $wkspc_dir/script"
mkdir -p "$wkspc_dir/script/conda_wrappers"
for cmd in cmake ctest
do
    ln -srf "$script_dir/conda_wrapper.sh" "$wkspc_dir/script/conda_wrappers/$cmd"
    echo "Created wrapper for $cmd at $wkspc_dir/script/conda_wrappers/$cmd"
done
