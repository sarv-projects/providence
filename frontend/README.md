# Frontend — Providence

Next.js 14 UI for chat + multi-agent Deep Research (A4 backend).

## Features

- **Chat** — multi-turn streaming; optional escalate to research  
- **Research** — modes, autonomy L1–L3, plan editor (L2 / plan-first)  
- **Thinking panel** — learned / gaps / next action from progress SSE  
- **History / Vault / Settings** — past runs, vault search, provider prefs  
- **Markdown + KaTeX** — report rendering  

## Stack

Next.js 14 · React 18 · TypeScript · Tailwind · Lucide · react-markdown  

## Setup

Backend API on **:8001** by default (see root README).

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Or from repo root:

```bash
bash scripts/start-dev.sh   # API + UI together
```

`next.config.js` rewrites `/api/*` to the FastAPI backend.
