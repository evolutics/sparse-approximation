#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

main() {
  local -r script_folder="$(dirname "$(readlink --canonicalize "$0")")"
  cd "$(dirname "${script_folder}")"

  docker run --entrypoint sh --rm --volume "${PWD}":/workdir \
    evolutics/travel-kit:0.8.0 -c \
    'git ls-files -z | xargs -0 travel-kit check --skip Pylint --'

  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --requirement requirements.txt

  journal test
}

main "$@"
