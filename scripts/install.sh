#!/bin/sh
# From the source checkout: sh scripts/install.sh [--game hack|eob3|skip] [--source PATH]
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$here/install.py" "$@"
