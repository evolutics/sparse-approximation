#!/bin/bash

set -o errexit -o nounset -o pipefail

main() {
  local -r script_folder="$(dirname "$(readlink --canonicalize "$0")")"
  cd "$(dirname "${script_folder}")"

  for jupytext_path in "$@"; do
    jupytext --execute --from py:nomarker --set-kernel - --to notebook \
      "${jupytext_path}"

    jupyter nbconvert --no-input --to html "${jupytext_path%.*}.ipynb"

    xdg-open "${jupytext_path%.*}.html"
  done
}

main "$@"
