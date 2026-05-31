# Frontend

ChatBeauty의 React 19 + TypeScript + Vite 단일 페이지 앱입니다. Vercel에 배포되며
백엔드의 `POST /recommend`를 호출하는 얇은 클라이언트입니다.

## Quick Start

```bash
cp .env.example .env   # VITE_API_URL 설정 (예: http://localhost:8080)
npm install
npm run dev            # http://localhost:5173
```

스크립트: `npm run dev` · `npm run build` (`tsc -b && vite build`) · `npm run lint` ·
`npm run preview`.

## Docs

- 작업 컨텍스트: [`CLAUDE.md`](./CLAUDE.md)
- 아키텍처(상태 모델, 컴포넌트, 디자인 토큰, 빌드/환경): [docs/frontend-architecture.md](../docs/frontend-architecture.md)
- API 계약: [docs/api-spec.md](../docs/api-spec.md)
