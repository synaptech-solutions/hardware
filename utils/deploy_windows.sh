#!/usr/bin/env bash
# Deploy the Windows-side helper scripts (*.ps1) from this repo's utils/ to the
# Windows filesystem, where they actually run.
#
# These scripts manage Windows networking (static IP, firewall, UDP forwarding)
# so they must run in Windows PowerShell, not WSL. The repo copy under utils/ is
# the source of truth — edit here, commit, then run this to push to Windows.
#
# Usage:  ./utils/deploy_windows.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_DEST="/mnt/c/Users/james/Documents/CODE/hardware-windows"

mkdir -p "$WIN_DEST"
cp -v "$HERE"/*.ps1 "$WIN_DEST"/
echo ""
echo "Deployed to: C:\\Users\\james\\Documents\\CODE\\hardware-windows"
echo "Run (Administrator PowerShell, first time):"
echo "  powershell -ExecutionPolicy Bypass -File C:\\Users\\james\\Documents\\CODE\\hardware-windows\\vicon_bridge.ps1"
