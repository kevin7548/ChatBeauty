# frontend/

React 19 + TypeScript + Vite SPA. Deployed on Vercel. Thin client over `POST /recommend`.

## Key facts
- **No router, no state library, no CSS framework.** All state is `useState` in `App.tsx`;
  views switch by conditionals (Hero → SkeletonCards → error → ProductCard list).
- Per-component CSS files (e.g. `ProductCard.tsx` + `ProductCard.css`); no CSS modules/Tailwind.

## Layout
```
src/
├── App.tsx               # all state + view switching
├── api/recommend.ts      # fetchRecommend(), warmUp()
├── types/recommend.ts    # RecommendRequest / ItemScore / RecommendResponse
├── hooks/useLoadingMessage.ts
└── components/           # Header, Hero, ProductCard, SearchFooter, WarmupBanner, SkeletonCard
```

## API client
- `API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"`.
- `warmUp()` GETs `/health` on mount to hide free-host (HF Space) cold-start latency.
- The payload sends `top_k: 5` but the **backend ignores `top_k`** (always Top-5).

## Run
```bash
cd frontend && cp .env.example .env   # VITE_API_URL
npm install && npm run dev            # :5173
```

## Docs
State model, components, design tokens, build/env → `docs/frontend-architecture.md`
