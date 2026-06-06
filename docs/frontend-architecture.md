# Frontend Architecture

React 19 + TypeScript + Vite single-page app in `frontend/`. Deployed on Vercel. It is a
thin client over the backend's `POST /recommend`.

## Stack & scripts
- React `^19.2`, TypeScript `~5.9`, Vite `^7`, ESLint `^9`. No router, no state library,
  no CSS framework.
- `npm run dev` (Vite dev server), `npm run build` (`tsc -b && vite build`),
  `npm run lint`, `npm run preview`.

## Structure

```
frontend/src/
├── App.tsx               # all state lives here (useState); renders the views
├── main.tsx              # React root (StrictMode)
├── api/recommend.ts      # fetchRecommend(), warmUp()
├── types/recommend.ts    # RecommendRequest / ItemScore / RecommendResponse
├── hooks/useLoadingMessage.ts
└── components/           # Header, Hero, ProductCard, SearchFooter,
                          # WarmupBanner, SkeletonCard (each with its own .css)
```

## State model (SPA, no router)

All state is `useState` in `App.tsx`; views are switched by conditionals, not routes:

| State | Meaning |
|---|---|
| `query` / `submittedQuery` | current input / last submitted text |
| `result` | `RecommendResponse \| null` |
| `isLoading` | request in flight |
| `error` | error message string |
| `warmupStatus` | `"warming" \| "warm" \| "failed"` |

View selection: Hero (idle) → 3 `SkeletonCard`s while `isLoading` → error state with retry
→ `ProductCard` list when `result` is set. `handleSubmit` ignores blank input and
re-entrancy while loading.

## API client (`api/recommend.ts`)
- `API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"`.
- `fetchRecommend(payload)` → `POST {API_BASE}/recommend`. The payload includes
  `top_k: 5`, but **the backend ignores `top_k`** (see [api-spec.md](api-spec.md)); the
  TS `RecommendResponse` also omits the backend's `latency` field — harmless, just unused.
- `warmUp()` → `GET {API_BASE}/health`, called once on mount to wake an idle/cold backend
  instance while the user reads the hero, hiding cold-start latency. (Free hosts like
  Render / Fly.io / HF Spaces idle-spin-down, so this matters again; the old Cloud Run
  `min-instances=1` config was retired 2026-06-02.)

## UX helpers
- `useLoadingMessage(isLoading)` rotates Korean status messages by elapsed seconds
  (<5s, <15s, <30s "waking the server", <50s, else) to make cold starts feel intentional.
- `WarmupBanner` surfaces `warmupStatus`.

## Styling
- Plain per-component CSS files (e.g. `ProductCard.tsx` + `ProductCard.css`); no CSS
  modules / Tailwind / CSS-in-JS. Global styles in `index.css`.
- Design tokens: pink gradient `#d94080 → #e8588c`; neutrals `#1d1d1f`/`#86868b`;
  background `#f5f5f7`; system font stack + `Playfair Display` for the logo; mobile
  breakpoint at `640px`.

## Build / deploy
- Env: `VITE_API_URL` (set in `frontend/.env` and in Vercel project settings).
- No `vercel.json` — Vercel uses framework defaults (Vite). `index.html` is `lang="ko"`
  with SEO/OG tags.
