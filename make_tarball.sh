#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-${SCRIPT_DIR}/../h100_profile_portable_20260618.tar.gz}"

tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' -czf "${OUT}" -C "$(dirname "${SCRIPT_DIR}")" "$(basename "${SCRIPT_DIR}")"
echo "${OUT}"
