from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Production-grade Retrieval-Augmented "
        "Generation system with hybrid retrieval, "
        "citation verification, and confidence scoring."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",          # Vite dev server
        "https://northbridge-ai-liart.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """
    Health check endpoint.
    """

    return {
        "message": "Enterprise RAG API is running."
    }