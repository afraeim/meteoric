#!/usr/bin/env bash
set -euo pipefail

REPO="quiet-node/meteoric"
BIN_NAME="meteoric"
INSTALL_DIR="${HOME}/.local/bin"
API_BASE="https://api.github.com/repos/${REPO}/releases/latest"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

detect_os() {
  case "$(uname -s)" in
    Linux*) echo "linux" ;;
    Darwin*) echo "macos" ;;
    *)
      echo "error: unsupported OS: $(uname -s)" >&2
      exit 1
      ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x64" ;;
    arm64|aarch64) echo "arm64" ;;
    *)
      echo "error: unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
}

resolve_artifact_pattern() {
  local os="$1"
  local arch="$2"
  if [ "$os" = "linux" ]; then
    if [ "$arch" = "x64" ]; then
      echo "linux.*x86_64.*\.AppImage$"
    else
      echo "linux.*aarch64.*\.AppImage$"
    fi
    return
  fi

  if [ "$arch" = "x64" ]; then
    echo "darwin.*x86_64.*\.app\.tar\.gz$"
  else
    echo "darwin.*aarch64.*\.app\.tar\.gz$"
  fi
}

sha256_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    echo "shasum -a 256"
  else
    echo ""
  fi
}

verify_checksum() {
  local checksums_file="$1"
  local artifact_name="$2"
  local artifact_path="$3"
  local hash_cmd expected actual

  hash_cmd="$(sha256_cmd)"
  if [ -z "$hash_cmd" ]; then
    echo "warning: sha256 checksum tool not found; skipping checksum verification" >&2
    return 0
  fi

  expected="$(grep -E "  ${artifact_name}$" "$checksums_file" | head -n1 | awk '{print $1}' || true)"
  if [ -z "$expected" ]; then
    echo "error: checksum entry for ${artifact_name} not found in checksums file" >&2
    exit 1
  fi

  actual="$(eval "$hash_cmd \"$artifact_path\"" | awk '{print $1}')"
  if [ "$expected" != "$actual" ]; then
    echo "error: checksum verification failed for ${artifact_name}" >&2
    exit 1
  fi
}

main() {
  need_cmd curl
  need_cmd grep
  need_cmd sed
  need_cmd tar
  need_cmd mktemp

  local os arch pattern release_json artifact_url artifact_name tmp_dir checksums_url
  os="$(detect_os)"
  arch="$(detect_arch)"
  pattern="$(resolve_artifact_pattern "$os" "$arch")"

  echo "==> Detecting latest Meteoric release for ${os}/${arch}..."
  release_json="$(curl -fsSL "$API_BASE")"

  artifact_url="$(printf '%s' "$release_json" | grep -Eo 'https://[^"]+' | grep -E "$pattern" | head -n1 || true)"
  if [ -z "$artifact_url" ]; then
    echo "error: no matching release artifact found for ${os}/${arch}" >&2
    echo "hint: ensure release artifacts are published in ${REPO} releases" >&2
    exit 1
  fi

  artifact_name="$(basename "$artifact_url")"
  checksums_url="$(printf '%s' "$release_json" | grep -Eo 'https://[^"]+' | grep -E 'checksums\.txt$' | head -n1 || true)"

  if [ -z "$checksums_url" ]; then
    echo "error: checksums.txt asset not found in latest release" >&2
    echo "hint: publish checksums.txt with release artifacts" >&2
    exit 1
  fi

  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  echo "==> Downloading ${artifact_name}..."
  curl -fsSL "$artifact_url" -o "${tmp_dir}/${artifact_name}"
  echo "==> Downloading checksums.txt..."
  curl -fsSL "$checksums_url" -o "${tmp_dir}/checksums.txt"
  echo "==> Verifying checksum..."
  verify_checksum "${tmp_dir}/checksums.txt" "$artifact_name" "${tmp_dir}/${artifact_name}"

  mkdir -p "$INSTALL_DIR"

  if [ "$os" = "linux" ]; then
    local target="${INSTALL_DIR}/${BIN_NAME}.AppImage"
    cp "${tmp_dir}/${artifact_name}" "$target"
    chmod +x "$target"
    ln -sf "$target" "${INSTALL_DIR}/${BIN_NAME}"
  else
    tar -xzf "${tmp_dir}/${artifact_name}" -C "$tmp_dir"
    local app_path
    app_path="$(find "$tmp_dir" -maxdepth 2 -name '*.app' | head -n1 || true)"
    if [ -z "$app_path" ]; then
      echo "error: downloaded archive did not contain a .app bundle" >&2
      exit 1
    fi
    mkdir -p "${HOME}/Applications"
    cp -R "$app_path" "${HOME}/Applications/"
  fi

  echo
  echo "✅ Meteoric installed successfully"
  if [ "$os" = "linux" ]; then
    echo "Run: ${BIN_NAME}"
  else
    echo "Run from: ~/Applications/Meteoric.app"
  fi
}

main "$@"
