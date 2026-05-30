---
name: frontend
description: Frontend tasks — React UI, components, routing, state management, styling. Use for changes scoped entirely to the frontend layer.
---

You work only in the `frontend/` directory. Never modify `backend/`, `ml/`, `db/`, or `deploy/` files.

## Stack
- React 18, TypeScript, Vite
- Deployed on Vercel (static hosting + CDN)

## Structure
```
frontend/src/
├── components/       # Header, Hero, ProductCard, SkeletonCard, WarmupBanner, SearchFooter
├── hooks/            # useLoadingMessage
├── api/              # backend API calls
├── types/            # TypeScript types
└── App.tsx           # state management and component composition only
```

## Key behaviors
- App.tsx is orchestration only — state + component composition, no inline markup
- Each component owns its own .tsx and .css file
- API base URL comes from environment variables, not hardcoded
- The backend returns Top-5 product recommendations with LLM-generated explanations

## Constraints
- Do not hardcode backend URLs — use env vars
- Keep App.tsx as orchestration only; add new UI to components/
