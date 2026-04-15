#!/usr/bin/env bash
set -euo pipefail

cd /home/xprof/Desktop/githubclone/Meteoric/meteoric

exec nix-shell -p bun nodejs_22 rustup pkg-config gtk3 webkitgtk_4_1 libsoup_3 --run "rustup default 1.88.0 && bun run dev"
