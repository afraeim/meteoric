# Meteoric

Meteoric is a cross-platform floating AI desktop assistant built with Tauri + React + Rust.

It starts from the Meteroic architecture and is now adapted for **OpenAI**, **Anthropic**, and **Ollama**.

## One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/afraeim/meteoric/main/scripts/install.sh | bash
```

The installer:

- supports Linux (`apt`, `dnf`, `yum`),
- installs build dependencies + Bun + Rust (if missing),
- clones and builds Meteoric from source,
- installs the generated `.deb`/`.rpm` as a regular system app.

## Local development

```bash
git clone https://github.com/quiet-node/meteoric.git
cd meteoric
bun install
cp .env.example .env
bun run dev
```

## Provider configuration

Configure providers in `.env`:

- `METEORIC_AI_PROVIDER=openai|anthropic|ollama`
- `OPENAI_API_KEY=...` (required for OpenAI)
- `ANTHROPIC_API_KEY=...` (required for Anthropic)
- `METEORIC_OPENAI_MODEL=gpt-4.1-mini`
- `METEORIC_ANTHROPIC_MODEL=claude-3-7-sonnet-latest`
- `METEORIC_SUPPORTED_AI_MODELS=gemma4:e2b,gemma4:e4b` (used for Ollama)

## Build and test

```bash
bun run typecheck
bun run test
bun run build:frontend
bun run build:backend
```

## Linux notes

- Linux packaging is first-class in release flow.
- Installer target is user-local (`~/.local/bin`).
- Runtime app data is stored in platform-resolved app data directories by Tauri.

## Windows install

Windows installers are published in each GitHub Release:

- `*.msi` (recommended)
- `*.exe` (NSIS)

Download from the repository Releases page and run the installer normally.

Need an installer before the next release? Run the GitHub Actions workflow
`Build Windows Installer` and download the artifact named
`meteoric-windows-exe-installer`.

## Repository status

This repository is an actively iterated migration from Meteroic into a dedicated `meteoric` project identity with provider abstraction and release hardening.
# meteoric
