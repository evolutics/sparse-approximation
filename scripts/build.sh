#!/bin/bash

set -o errexit -o nounset -o pipefail

cd -- "$(dirname -- "$0")/.."

for jupytext_path in "$@"; do
  jupytext --execute --from py:nomarker --set-kernel - --to notebook \
    "${jupytext_path}"

  jupyter nbconvert --no-input --to html "${jupytext_path%.*}.ipynb"

  xdg-open "${jupytext_path%.*}.html"
done
