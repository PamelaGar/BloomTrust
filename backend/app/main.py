"""Punto de entrada de la API BloomTrust (FastAPI + Uvicorn)."""

import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="BloomTrust",
    description=(
        "Plataforma de optimización de presupuestos y análisis de "
        "abastecimiento floral para el mercado B2B de eventos en New York City."
    ),
    version="0.1.0",
)

# API Routers — app/api/v1/
# Gestión de endpoints prevista (se montarán más adelante):
#   auth       — autenticación de floristas y proveedores
#   providers  — catálogos de distribuidores de NYC
#   sourcing   — cotización y abastecimiento
#   chat       — asistente conversacional (Planner, Critic, Narrator)
#   admin      — administración
#
# from app.api.v1.auth import router as auth_router
# from app.api.v1.providers import router as providers_router
# from app.api.v1.sourcing import router as sourcing_router
# from app.api.v1.chat import router as chat_router
# from app.api.v1.admin import router as admin_router
#
# app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
# app.include_router(providers_router, prefix="/api/v1/providers", tags=["providers"])
# app.include_router(sourcing_router, prefix="/api/v1/sourcing", tags=["sourcing"])
# app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
# app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/")
async def welcome() -> dict:
    """Endpoint de prueba: bienvenida corporativa a BloomTrust."""
    return {
        "service": "BloomTrust",
        "message": "Bienvenido a BloomTrust.",
        "description": (
            "Plataforma de optimización de presupuestos y análisis de "
            "abastecimiento floral para el mercado B2B de eventos en New York City."
        ),
        "market": "NYC",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
