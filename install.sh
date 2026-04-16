#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this install.sh is intended for Ubuntu/Linux only" >&2
  exit 1
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *"debian"* ]]; then
    echo "warning: this script is optimized for Ubuntu/Debian. Continuing anyway..."
  fi
fi

echo "==> Installing system dependencies"
sudo apt update
sudo apt install -y \
  curl build-essential pkg-config libssl-dev \
  libgtk-3-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev zlib1g-dev \
  libx11-dev libxi-dev libxtst-dev

if ! command -v bun >/dev/null 2>&1; then
  echo "==> Installing Bun"
  curl -fsSL https://bun.sh/install | bash
fi
export PATH="$HOME/.bun/bin:$PATH"

if ! command -v rustup >/dev/null 2>&1; then
  echo "==> Installing Rust"
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup default stable >/dev/null

echo "==> Installing JS dependencies"
bun install

echo "==> Building Linux package"
bun run build:backend:linux

echo "==> Installing .deb"
sudo dpkg -i src-tauri/target/release/bundle/deb/Meteoric_0.1.0_amd64.deb || true
sudo apt-get install -f -y

echo ""
echo "✅ Meteoric installed successfully"
echo "Run: meteoric"
echo "Tip: keep Meteoric running in background; double Shift toggles popup."
