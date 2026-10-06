"""Punto de entrada de la API BloomTrust (FastAPI + Uvicorn)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_BACKEND = Path(__file__).resolve().parents[1]
_RAIZ = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_COLUMNAS_CATALOGO = {
    "especie",
    "proveedor",
    "banda",
    "precio_tallo",
    "stock",
    "merma",
    "saldo_excedente",
    "meses",
}
_logger = logging.getLogger("bloomtrust")

app = FastAPI(
    title="BloomTrust",
    description=(
        "Plataforma de optimización de presupuestos y análisis de "
        "abastecimiento floral para el mercado B2B de eventos en New York City."
    ),
    version="0.1.0",
)

# El index.html del escritorio se abre como archivo local (origen "null")
# o desde un servidor en localhost. Sin este middleware el navegador bloquea
# el POST hacia el puerto 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cargar_env() -> bool:
    """Lee OPENAI_API_KEY desde .env sin pisar el entorno ni registrar el secreto."""
    vistos: set[Path] = set()
    for directorio in (Path.cwd(), _RAIZ, _BACKEND):
        ruta = (directorio / ".env").resolve()
        if ruta in vistos or not ruta.is_file():
            vistos.add(ruta)
            continue
        vistos.add(ruta)
        try:
            lineas = ruta.read_text(encoding="utf-8").splitlines()
        except OSError:
            _logger.warning("No se pudo leer %s", ruta)
            continue
        for linea in lineas:
            limpia = linea.strip()
            if not limpia or limpia.startswith("#") or "=" not in limpia:
                continue
            clave, valor = limpia.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))
        break
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _cargar_catalogo():
    """Incorpora df_inventario y descarta el marco si no trae el esquema esperado."""
    try:
        from app.services.market.inventory import df_inventario
    except Exception:
        _logger.warning("No se pudo importar df_inventario", exc_info=True)
        return None
    columnas = set(getattr(df_inventario, "columns", []))
    if df_inventario is None or not _COLUMNAS_CATALOGO.issubset(columnas):
        _logger.warning("df_inventario no tiene el esquema del catálogo floral")
        return None
    if df_inventario.empty:
        _logger.warning("df_inventario está vacío")
        return None
    return df_inventario.copy()


app.state.openai_configurada = _cargar_env()
app.state.df_inventario = _cargar_catalogo()

from app.api.v1.chat import router as chat_router
from app.api.v1.sourcing import router as sourcing_router

# API Routers — app/api/v1/
# Gestión de endpoints prevista (se montarán más adelante):
#   auth       — autenticación de floristas y proveedores
#   providers  — catálogos de distribuidores de NYC
#   chat       — asistente conversacional (Planner, Critic, Narrator)
#   admin      — administración
#
# from app.api.v1.auth import router as auth_router
# from app.api.v1.providers import router as providers_router
# from app.api.v1.admin import router as admin_router
#
# app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
# app.include_router(providers_router, prefix="/api/v1/providers", tags=["providers"])
# app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])

app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(sourcing_router, prefix="/api/v1/sourcing", tags=["sourcing"])


@app.get("/")
async def welcome() -> dict:
    """Endpoint de prueba: bienvenida corporativa a BloomTrust."""
    catalogo = app.state.df_inventario
    especies = sorted(catalogo["especie"].unique()) if catalogo is not None else []
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
        "catalogo_cargado": catalogo is not None,
        "especies": especies,
        "openai_configurada": bool(app.state.openai_configurada),
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
