"""
main.py
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Logistics SQL Agent API",
    description="AI-powered logistics analytics using LangChain + Gemini + PostgreSQL",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


@app.get("/")
async def root():
    return {
        "message": "🚚 Logistics SQL Agent API is running!",
        "docs": "/docs",
        "health": "/health",
    }
