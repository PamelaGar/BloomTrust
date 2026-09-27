"""Cotización de abastecimiento: imagen, stock estacional y tres bandas."""

from __future__ import annotations

import asyncio
import base64
import unicodedata

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.agents.multimodal_agent import AnalisisReferencia, analizar_imagen
from app.services.market.inventory import df_inventario, filtrar_inventario
from app.valuation.engine import BANDAS, BloomTrustCostEngine

router = APIRouter()

_LIMITE_IMAGEN = 8 * 1024 * 1024
_DECLARACION = (
    "BloomTrust es una herramienta de optimización logística y comercial de abastecimiento. "
    "No asume el transporte ni el almacenamiento físico de las flores. "
    "La cotización solo incluye stock en temporada para el mes del evento."
)
_MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_NOMBRE_MES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def _plano(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def resolver_mes(mes_evento: str) -> tuple[int, str]:
    """Acepta el nombre del mes, un número 1-12 o una fecha AAAA-MM."""
    token = _plano(mes_evento)
    if not token:
        raise HTTPException(status_code=422, detail="mes_evento es obligatorio.")
    if token in _MESES:
        numero = _MESES[token]
        return numero, _NOMBRE_MES[numero]
    pieza = token.replace("/", "-").split("-")
    candidato = pieza[1] if len(pieza) >= 2 and len(pieza[0]) == 4 else pieza[0]
    if candidato.isdigit():
        numero = int(candidato)
        if 1 <= numero <= 12:
            return numero, _NOMBRE_MES[numero]
    raise HTTPException(
        status_code=422,
        detail="mes_evento debe ser un mes válido, por ejemplo 'mayo' o '5'.",
    )


def componer_respuesta(
    analisis: AnalisisReferencia,
    presupuesto_max: float,
    mes_evento: str,
) -> dict:
    """Filtra el inventario en temporada y, si hay stock, calcula las tres bandas."""
    mes, nombre_mes = resolver_mes(mes_evento)
    detectadas = list(analisis.flores_detectadas)
    filtrado = filtrar_inventario(detectadas, mes, df_inventario)
    en_temporada = sorted(set(filtrado["especie"])) if not filtrado.empty else []
    fuera = [flor for flor in detectadas if flor not in set(en_temporada)]
    cumplimiento = bool(detectadas) and not fuera

    bandas = None
    if not filtrado.empty:
        bandas = BloomTrustCostEngine().calcular(filtrado, presupuesto_max)

    return {
        "tipo": "cumplimiento_stock_estacional",
        "mercado": "NYC",
        "mes_evento": nombre_mes,
        "cumplimiento": cumplimiento,
        "flujo_detenido": not cumplimiento,
        "flores_detectadas": detectadas,
        "flores_en_temporada": en_temporada,
        "flores_fuera_de_temporada": fuera,
        "declaracion": _DECLARACION,
        "estilo_estetico": analisis.estilo_estetico,
        "paleta_colores": list(analisis.paleta_colores),
        "presupuesto_max": presupuesto_max,
        "bandas": bandas,
        "bandas_exigidas": list(BANDAS),
    }


@router.post("/analyze-event")
async def analyze_event(
    archivo: UploadFile = File(..., description="Imagen de referencia del event planner."),
    presupuesto_max: float = Form(..., description="Presupuesto máximo del evento, en USD."),
    mes_evento: str = Form(..., description="Mes del evento, por ejemplo 'mayo'."),
) -> dict:
    """Analiza la imagen, valida temporada contra el inventario de NYC y cotiza tres bandas."""
    if presupuesto_max <= 0:
        raise HTTPException(status_code=422, detail="presupuesto_max debe ser mayor que cero.")
    resolver_mes(mes_evento)

    contenido = await archivo.read(_LIMITE_IMAGEN + 1)
    if not contenido:
        raise HTTPException(status_code=422, detail="La imagen de referencia está vacía.")
    if len(contenido) > _LIMITE_IMAGEN:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 8 MB.")

    mime = archivo.content_type or ""
    if mime and not mime.startswith("image/") and mime != "application/octet-stream":
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen.")

    imagen_b64 = base64.b64encode(contenido).decode("ascii")
    try:
        analisis = await asyncio.to_thread(
            analizar_imagen,
            imagen_b64,
            tipo_mime=mime if mime.startswith("image/") else None,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return componer_respuesta(analisis, presupuesto_max, mes_evento)
