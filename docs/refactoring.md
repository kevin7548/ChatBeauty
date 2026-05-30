# Project Refactoring Summary

## 1. ML Pipeline Separation

### Before

`ml/` was nested inside `backend/`, mixing training/experimentation code with the API server:

```
backend/
├── app/                  # FastAPI server (serving)
├── ml/                   # ML pipeline (training & experimentation)
│   ├── item_ranker/
│   ├── pipeline/
│   ├── scripts/
│   ├── model/
│   └── data/
├── notebooks/            # Colab notebooks
├── Dockerfile
└── docker-compose.yml
```

### Why It Matters

- **Different lifecycles**: The API server and ML pipeline are developed, tested, and deployed independently. Training runs happen offline; the server runs in production. Nesting one inside the other conflates these concerns.
- **Docker build context**: The Dockerfile only needs `app/` and `item_ranker/` at runtime, but the build context included all raw data, processed data, and training scripts — unnecessary bloat.
- **Navigation**: Contributors looking for ML code had to know it was inside `backend/`, which is counterintuitive for a root-level concern.

### After

`ml/` is a top-level directory alongside `backend/` and `frontend/`:

```
.
├── backend/              # API server only
│   ├── app/
│   ├── sql/
│   ├── Dockerfile
│   └── docker-compose.yml
├── ml/                   # ML pipeline, training, models
│   ├── item_ranker/
│   ├── pipeline/
│   ├── scripts/
│   ├── notebooks/        # (moved from backend/notebooks/)
│   ├── model/
│   └── data/
└── frontend/
```

**Files updated:**
- `backend/Dockerfile` — COPY paths adjusted for root-level build context (`backend/pyproject.toml`, `backend/app/`, `ml/setup.py`, `ml/item_ranker/`)
- `backend/docker-compose.yml` — build context changed to `..` (project root), model volume mounts updated to `../ml/model/`
- `ml/README.md` — path instructions updated
- `README.md`, `README_EN.md` — repository structure and quick start commands updated

---

## 2. Frontend Component Split

### Before

The entire UI lived in two files:

```
src/
├── App.tsx    (338 lines)  — hook, utility, component, constants, all UI states
├── App.css    (514 lines)  — every style in one flat file
├── api/
├── types/
└── main.tsx
```

`App.tsx` contained:
- `useLoadingMessage` custom hook (loading message based on elapsed time)
- `renderStars` utility function
- `SkeletonCard` component (loading placeholder)
- `EXAMPLE_QUERIES` constant
- All UI states (hero, loading, error, results) in a single `App` component

### Why It Matters

- **Findability**: Locating how a product card renders requires scrolling through 338 lines of unrelated code. With separate files, you open `ProductCard.tsx` directly.
- **Isolation**: Changing one component's markup or styles risks breaking another when they share the same file. Separate files keep changes scoped.
- **Reusability**: Components like `SkeletonCard` or `Header` can't be imported elsewhere when they're embedded inside `App.tsx`.
- **Code review**: A change to the search footer produces a focused diff in `SearchFooter.tsx`, not a hard-to-read diff buried in a 338-line file.

### After

```
src/
├── components/
│   ├── Header.tsx          + Header.css
│   ├── WarmupBanner.tsx    + WarmupBanner.css
│   ├── Hero.tsx            + Hero.css
│   ├── ProductCard.tsx     + ProductCard.css
│   ├── SkeletonCard.tsx    + SkeletonCard.css
│   └── SearchFooter.tsx    + SearchFooter.css
├── hooks/
│   └── useLoadingMessage.ts
├── api/
├── types/
├── App.tsx    (125 lines)  — state management and component composition only
├── App.css     (77 lines)  — layout, results area, error state only
└── main.tsx
```

Each component owns its markup and styles. `App.tsx` is now purely orchestration — managing state and composing the pieces together.
