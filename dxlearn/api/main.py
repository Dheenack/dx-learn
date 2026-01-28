from fastapi import FastAPI
from dxlearn.api.routes import router

app = FastAPI(
    title="DX-learn API",
    version="0.1.0"
)

app.include_router(router)
