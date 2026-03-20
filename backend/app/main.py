import os
from fastapi import FastAPI
from app.api.routes import recommend
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.latency import LatencyMiddleware

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173",
).split(",")

app = FastAPI(debug=DEBUG)

app.add_middleware(LatencyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(recommend.router)


@app.get("/")
def root():
    return {"message": "hi"}


@app.get("/health")
def health():
    return {"status": "ok"}
