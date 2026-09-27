"""Agente Crítico visual de BloomTrust.

Analiza una imagen de referencia del event planner (URL o Base64) con
gpt-4o-mini y devuelve solo especies, colores y estilo que están a la vista.
La salida queda fijada por JSON Schema mediante salidas estructuradas.
"""

from __future__ import annotations

import base64
import binascii
import os
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PROMPT_AGENTE_CRITICO = """Eres el Agente Crítico visual de BloomTrust, plataforma B2B de abastecimiento floral para eventos en New York City.

Tu única tarea es inspeccionar la imagen de referencia que cargó el event planner y extraer lo que se ve. No diseñas, no cotizas y no completas el ramo con especies probables.

Reglas obligatorias:
- Incluye en flores_detectadas solo especies botánicas claramente visibles. Si no puedes distinguir la especie, omítela.
- No alucines flores, capullos ni rellenos a partir del estilo, la ocasión, el fondo o la paleta.
- El follaje, los jarrones, las cintas y la tela no son flores. No los conviertas en especies.
- Una lista vacía es una respuesta válida cuando no hay flores identificables.
- flores_detectadas: nombre común en español, minúsculas y sin acentos (rosa, orquidea, peonia). Si solo estás seguro del nombre en inglés, devuélvelo en minúsculas (peony) y no lo traduzcas.
- paleta_colores: únicamente colores predominantes que aparecen en el diseño, en minúsculas y sin acentos.
- estilo_estetico: una sola etiqueta corta del estilo visual observable (boho, minimalista, clasico, romantico, tropical, corporativo, jardin). No asignes un estilo que la imagen no muestre.
- No calcules tallos, precios ni disponibilidad."""

_ALIAS_FLOR = {
    "rose": "rosa",
    "roses": "rosa",
    "rosas": "rosa",
    "orchid": "orquidea",
    "orchids": "orquidea",
    "orquideas": "orquidea",
    "peony": "peonia",
    "peonies": "peonia",
    "peonias": "peonia",
    "tulip": "tulipan",
    "tulips": "tulipan",
    "tulipanes": "tulipan",
    "ranunculus": "ranunculo",
    "ranunculos": "ranunculo",
    "hydrangea": "hortensia",
    "hortensias": "hortensia",
    "dahlia": "dalia",
    "dahlias": "dalia",
    "dalias": "dalia",
    "sunflower": "girasol",
    "sunflowers": "girasol",
    "girasoles": "girasol",
    "anemone": "anemona",
    "anemones": "anemona",
    "anemonas": "anemona",
}

_ALIAS_ESTILO = {
    "classic": "clasico",
    "classical": "clasico",
    "minimalist": "minimalista",
    "minimal": "minimalista",
    "bohemian": "boho",
    "bohemio": "boho",
    "romantic": "romantico",
    "tropical": "tropical",
    "corporate": "corporativo",
    "garden": "jardin",
}

_ALIAS_COLOR = {
    "white": "blanco",
    "ivory": "marfil",
    "cream": "crema",
    "green": "verde",
    "pink": "rosa",
    "red": "rojo",
    "burgundy": "burdeos",
    "blush": "blush",
    "yellow": "amarillo",
    "orange": "naranja",
    "purple": "morado",
    "lilac": "lila",
    "blue": "azul",
    "black": "negro",
    "gold": "dorado",
}


class AnalisisReferencia(BaseModel):
    """Esquema estricto que OpenAI debe devolver para la imagen."""

    flores_detectadas: list[str] = Field(
        description="Especies botánicas visibles. Minúsculas, sin acentos. Vacío si no hay certeza."
    )
    paleta_colores: list[str] = Field(
        description="Colores predominantes del diseño, en minúsculas y sin acentos."
    )
    estilo_estetico: str = Field(
        description="Estilo visual observable: boho, minimalista, clasico u otra etiqueta corta."
    )

    @field_validator("flores_detectadas")
    @classmethod
    def _normalizar_flores(cls, valores: list[str]) -> list[str]:
        return _unicos(_canon_flor(item) for item in valores)

    @field_validator("paleta_colores")
    @classmethod
    def _normalizar_colores(cls, valores: list[str]) -> list[str]:
        return _unicos(_canon_color(item) for item in valores)

    @field_validator("estilo_estetico")
    @classmethod
    def _normalizar_estilo(cls, valor: str) -> str:
        token = _plano(valor)
        return _ALIAS_ESTILO.get(token, token)


def _plano(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto.strip().lower())
    sin_acentos = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return " ".join(sin_acentos.split())


def _unicos(tokens) -> list[str]:
    vistos: set[str] = set()
    salida: list[str] = []
    for token in tokens:
        if not token or token in vistos:
            continue
        vistos.add(token)
        salida.append(token)
    return salida


def _canon_flor(nombre: str) -> str:
    token = _plano(nombre)
    return _ALIAS_FLOR.get(token, token)


def _canon_color(nombre: str) -> str:
    token = _plano(nombre)
    return _ALIAS_COLOR.get(token, token)


def _cargar_env() -> None:
    vistos: set[Path] = set()
    origen = Path(__file__).resolve().parent
    for directorio in (Path.cwd(), *origen.parents):
        ruta = (directorio / ".env").resolve()
        if ruta in vistos:
            continue
        vistos.add(ruta)
        if not ruta.is_file():
            continue
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            limpia = linea.strip()
            if not limpia or limpia.startswith("#") or "=" not in limpia:
                continue
            clave, valor = limpia.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))
        return


def _configuracion() -> tuple[str, str, float]:
    _cargar_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está definida. Añádela al archivo .env de BloomTrust."
        )
    modelo = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    try:
        temperatura = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
    except ValueError:
        temperatura = 0.5
    return api_key, modelo, temperatura


def _mime_desde_base64(payload: str) -> str:
    if payload.startswith("/9j/"):
        return "image/jpeg"
    if payload.startswith("iVBOR"):
        return "image/png"
    if payload.startswith("R0lGOD"):
        return "image/gif"
    if payload.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _es_base64(payload: str) -> bool:
    if len(payload) < 16:
        return False
    relleno = "=" * (-len(payload) % 4)
    try:
        base64.b64decode(payload + relleno, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def preparar_imagen(imagen: str, tipo_mime: str | None = None) -> str:
    """Acepta una URL http(s), un data URL o Base64 crudo y devuelve el valor de image_url."""
    valor = imagen.strip()
    if not valor:
        raise ValueError("La imagen de referencia está vacía.")
    if valor.startswith(("http://", "https://")):
        return valor
    if valor.startswith("data:image/"):
        return "".join(valor.split())
    payload = "".join(valor.split())
    if not _es_base64(payload):
        raise ValueError("La imagen debe ser una URL http(s) o un Base64 válido.")
    mime = tipo_mime if tipo_mime and tipo_mime.startswith("image/") else _mime_desde_base64(payload)
    return f"data:{mime};base64,{payload}"


def analizar_imagen(
    imagen: str,
    *,
    tipo_mime: str | None = None,
    detalle: Literal["low", "high", "auto"] = "auto",
) -> AnalisisReferencia:
    """Identifica flores, paleta y estilo visibles mediante salida estructurada."""
    image_url = preparar_imagen(imagen, tipo_mime)
    api_key, modelo, temperatura = _configuracion()
    from openai import OpenAI

    cliente = OpenAI(api_key=api_key, timeout=60.0)
    completado = cliente.responses.parse(
        model=modelo,
        temperature=temperatura,
        input=[
            {"role": "system", "content": PROMPT_AGENTE_CRITICO},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analiza esta imagen de referencia del event planner. "
                            "Extrae solo las especies botánicas, los colores y el estilo que se ven."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detalle,
                    },
                ],
            },
        ],
        text_format=AnalisisReferencia,
    )
    analisis = completado.output_parsed
    if analisis is None:
        raise RuntimeError("OpenAI no devolvió la salida estructurada del análisis visual.")
    return analisis
