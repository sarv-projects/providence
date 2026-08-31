# Screenshots

## Available assets

| File | Description |
|------|-------------|
| [architecture.svg](architecture.svg) | High-level system topology diagram |

## Capturing live UI screenshots

With the dev stack running (`bash scripts/start-dev.sh`):

1. Open http://localhost:3000 — capture **chat** and **research** modes
2. Settings → http://localhost:3000/settings
3. Vault → http://localhost:3000/vault
4. History → http://localhost:3000/history
5. API docs → http://localhost:8001/docs

Suggested filenames for README:

- `hero.png` — main chat UI
- `research-progress.png` — research with progress banner
- `settings.png` — provider catalog
- `terminal.png` — `main.py research` CLI output

Use PNG, ≤500KB, dark mode preferred.
