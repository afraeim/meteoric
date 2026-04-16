#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/afraeim/meteoric.git"
WORKDIR="${HOME}/.cache/meteoric-install"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_linux() {
  if [ "$(uname -s)" != "Linux" ]; then
    echo "error: this installer currently supports Linux only" >&2
    exit 1
  fi
}

detect_pkg_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    echo "apt"
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    echo "dnf"
    return
  fi
  if command -v yum >/dev/null 2>&1; then
    echo "yum"
    return
  fi
  echo ""
}

install_system_deps() {
  local pm="$1"
  case "$pm" in
    apt)
      sudo apt-get update
      sudo apt-get install -y \
        ca-certificates curl git build-essential pkg-config libssl-dev \
        libgtk-3-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev zlib1g-dev \
        libx11-dev libxi-dev libxtst-dev
      ;;
    dnf)
      sudo dnf install -y \
        ca-certificates curl git gcc gcc-c++ make pkgconf-pkg-config openssl-devel \
        gtk3-devel webkit2gtk4.1-devel libsoup3-devel zlib-devel \
        libX11-devel libXi-devel libXtst-devel
      ;;
    yum)
      sudo yum install -y \
        ca-certificates curl git gcc gcc-c++ make pkgconfig openssl-devel \
        gtk3-devel webkit2gtk4.1-devel libsoup3-devel zlib-devel \
        libX11-devel libXi-devel libXtst-devel
      ;;
    *)
      echo "error: unsupported package manager. Please install build deps manually." >&2
      exit 1
      ;;
  esac
}

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    return
  fi
  echo "==> Installing Bun..."
  curl -fsSL https://bun.sh/install | bash
  export PATH="${HOME}/.bun/bin:${PATH}"
}

ensure_rust() {
  if command -v rustup >/dev/null 2>&1; then
    rustup default stable >/dev/null
    return
  fi
  echo "==> Installing Rust toolchain..."
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  export PATH="${HOME}/.cargo/bin:${PATH}"
  rustup default stable >/dev/null
}

build_and_install() {
  local pm="$1"

  rm -rf "$WORKDIR"
  mkdir -p "$WORKDIR"

  echo "==> Cloning Meteoric..."
  git clone --depth 1 "$REPO_URL" "$WORKDIR/meteoric"
  cd "$WORKDIR/meteoric"

  echo "==> Installing JS dependencies..."
  bun install

  echo "==> Building Linux installer packages..."
  bun run build:backend:linux

  case "$pm" in
    apt)
      echo "==> Installing .deb package..."
      sudo dpkg -i src-tauri/target/release/bundle/deb/Meteoric_0.1.0_amd64.deb || true
      sudo apt-get install -f -y
      ;;
    dnf|yum)
      echo "==> Installing .rpm package..."
      local rpm_pkg
      rpm_pkg="$(ls -1 src-tauri/target/release/bundle/rpm/*.rpm | head -n1)"
      if [ -z "$rpm_pkg" ]; then
        echo "error: rpm artifact not found" >&2
        exit 1
      fi
      sudo rpm -Uvh --replacepkgs "$rpm_pkg"
      ;;
  esac
}

main() {
  require_linux
  need_cmd curl
  need_cmd git
  need_cmd sudo

  local pm
  pm="$(detect_pkg_manager)"
  if [ -z "$pm" ]; then
    echo "error: could not detect apt/dnf/yum; install dependencies manually and build from source" >&2
    exit 1
  fi

  echo "==> Installing system dependencies (${pm})..."
  install_system_deps "$pm"

  ensure_bun
  ensure_rust
  build_and_install "$pm"

  echo
  echo "✅ Meteoric installed successfully"
  echo "Run: meteoric"
  echo "Tip: keep Meteoric running in background; double Shift toggles popup."
}

main "$@"
