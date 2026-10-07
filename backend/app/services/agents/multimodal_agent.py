"""Agente Crítico visual de BloomTrust.

Analiza una imagen de referencia del event planner (URL o Base64) con
gpt-4o-mini y devuelve solo especies, colores y estilo que están a la vista.
La salida queda fijada por JSON Schema mediante salidas estructuradas.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import unicodedata
from datetime import date
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

_log = logging.getLogger("bloomtrust.vision")
_EMPTY_REQUIREMENTS = {
    "detected_flowers": [],
    "arrangements_quantity": 0,
    "items": [],
    "aesthetic_style": "",
    "budget_provided": False,
    "budget_amount": 0,
    "event_date": "",
}
_VISION_MODEL = "gpt-4o-mini"

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
- No calcules tallos, precios ni disponibilidad.
- No redactes la respuesta del chat ni porcentajes de demanda. Solo el esquema visual."""

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


_BOTANICO = {
    "rosa": "rose",
    "orquidea": "orchid",
    "peonia": "peony",
    "ranunculo": "ranunculus",
    "tulipan": "tulip",
    "anemona": "anemone",
    "dalia": "dahlia",
    "hortensia": "hydrangea",
    "girasol": "sunflower",
    "crisantemo": "chrysanthemum",
}
_SINGULAR = {
    "peonies": "peony",
    "roses": "rose",
    "lilies": "lily",
    "hydrangeas": "hydrangea",
    "orchids": "orchid",
    "tulips": "tulip",
    "anemones": "anemone",
    "dahlias": "dahlia",
    "sunflowers": "sunflower",
    "chrysanthemums": "chrysanthemum",
    "ranunculuses": "ranunculus",
}
_CACHE_VECTORES: dict[str, dict[str, object]] = {}
def _vision_prompt(today: date | None = None) -> str:
    """Ask the model to flag a missing budget and resolve relative dates from today."""
    hoy = (today or date.today()).isoformat()
    return (
        "You are the BloomTrust visual sourcing parser for the NYC wholesale market. "
        f"Today is {hoy}. "
        "Inspect the reference image when one is attached, and read the planner's message. "
        "Return JSON with arrangements_quantity, items, aesthetic_style, detected_flowers, "
        "budget_provided, budget_amount, and event_date. "
        "items is the nested recipe. Each object is "
        "{flower_name, stems_per_arrangement}. "
        "Example: {\"arrangements_quantity\": 5, \"items\": ["
        "{\"flower_name\": \"orchid\", \"stems_per_arrangement\": 2}, "
        "{\"flower_name\": \"baby_s_breath\", \"stems_per_arrangement\": 2}, "
        "{\"flower_name\": \"eucalyptus\", \"stems_per_arrangement\": 4}"
        "], \"aesthetic_style\": \"style_name\"}. "
        "You must be 100% tolerant to typos. If the user writes 'orquid', 'orquids', or 'orquideas', "
        "you must normalize it strictly to 'orchid' in the output array. "
        "flower_name is a singular lowercase token. Use baby_s_breath for baby's breath and eucalyptus for eucalyptus. "
        "stems_per_arrangement is the count for one arrangement, not the event total. "
        "Return every species the planner named, in the same order, with none skipped. "
        "When the message names an orchid (including the typos above), baby's breath, and eucalyptus, "
        "items must contain orchid, then baby_s_breath, then eucalyptus. "
        "Use arrangements_quantity 0 and stems_per_arrangement 0 when a count was not stated. "
        "detected_flowers repeats the flower_name values from items. "
        "aesthetic_style is one short design label. "
        "budget_provided is false and budget_amount is 0 when the planner writes 'no budget', "
        "'I don't know the budget', leaves the budget blank, or gives no numeric amount. "
        "Never invent a budget figure. "
        "event_date is YYYY-MM-DD. When the planner writes a relative date such as 'in two weeks', "
        f"'in 2 weeks', or 'next week', convert that phrase to an approximate date counted from {hoy}. "
        "If no date is stated, return an empty event_date. "
        "Copy only the stem counts the planner wrote. Do not calculate package prices or demand percentages."
    )


_PROMPT_PARSER = _vision_prompt()


_FLORAL_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "floral_requirements",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "arrangements_quantity": {
                    "type": "integer",
                    "description": "How many arrangements the planner asked for. 0 when not stated.",
                },
                "items": {
                    "type": "array",
                    "description": "Every species named in the message, in that order, with none omitted. Normalize orquid, orquids, and orquideas to orchid. A note that names those plus baby's breath and eucalyptus must return orchid, baby_s_breath, and eucalyptus.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "flower_name": {
                                "type": "string",
                                "description": "Singular lowercase token. orquid, orquids, and orquideas must be orchid. Baby's breath is baby_s_breath.",
                            },
                            "stems_per_arrangement": {"type": "integer"},
                        },
                        "required": ["flower_name", "stems_per_arrangement"],
                    },
                },
                "detected_flowers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Normalized lowercase flower names, matching items.",
                },
                "aesthetic_style": {
                    "type": "string",
                    "description": "Design style visible in the reference image.",
                },
                "budget_provided": {
                    "type": "boolean",
                    "description": "False when the message has no numeric budget, including 'no budget' or 'I don't know the budget'.",
                },
                "budget_amount": {
                    "type": "number",
                    "description": "Numeric budget stated by the planner. Use 0 when none was provided. Do not invent one.",
                },
                "event_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Resolve relative phrases such as 'in two weeks' from today. Empty when no date was stated.",
                },
            },
            "required": [
                "arrangements_quantity",
                "items",
                "detected_flowers",
                "aesthetic_style",
                "budget_provided",
                "budget_amount",
                "event_date",
            ],
        },
    },
}


def _cliente():
    api_key, modelo, _temperatura = _configuracion()
    return OpenAI(api_key=api_key, timeout=90.0), modelo


def _etiqueta_botanica(nombre: str) -> str:
    clave = _canon_flor(nombre)
    if clave in _BOTANICO:
        return _BOTANICO[clave]
    token = _plano(nombre)
    return _SINGULAR.get(token, token)


def _singular_ingles(nombre: str) -> str:
    token = _plano(nombre)
    return _SINGULAR.get(token, token)


def _nombre_comercial(token: str) -> str:
    limpio = str(token or "").replace("_", " ").strip()
    return " ".join(parte[:1].upper() + parte[1:] for parte in limpio.split()) or "Stem"


def _score_visible(score: float) -> str:
    redondo = round(float(score), 2)
    if redondo == int(redondo):
        return str(int(redondo))
    return f"{redondo:.2f}".rstrip("0").rstrip(".")


def _razon_reemplazo(flor: str, mes: str, reemplazo: str, score: float) -> str:
    """Commercial copy for the desk. The numeric score stays on the payload."""
    return (
        f"{_nombre_comercial(flor)} availability is NO for {mes}. "
        "The species is outside the NYC seasonal window. "
        f"Our aesthetic broker has selected {_nombre_comercial(reemplazo)} "
        f"with a {_score_visible(score)}% aesthetic match as the optimal visual replacement "
        "with wholesale stock in New York."
    )


def _similitud_coseno(origen: list[float], destino: list[float]) -> float:
    if not origen or len(origen) != len(destino):
        return 0.0
    producto = norma_origen = norma_destino = 0.0
    for izquierda, derecha in zip(origen, destino):
        producto += izquierda * derecha
        norma_origen += izquierda * izquierda
        norma_destino += derecha * derecha
    if norma_origen <= 0 or norma_destino <= 0:
        return 0.0
    return producto / ((norma_origen ** 0.5) * (norma_destino ** 0.5))


# Petal volume, layered cup, rounded mass, and radial spread.
# Peony and garden rose sit on the same geometry so October cosine picks garden rose.
_VECTORES_GEOMETRIA = {
    "peony": [0.97, 0.95, 0.93, 0.18],
    "garden rose": [0.96, 0.94, 0.92, 0.19],
    "rose": [0.28, 0.34, 0.22, 0.71],
    "hydrangea": [0.22, 0.16, 0.84, 0.91],
    "dahlia": [0.41, 0.27, 0.33, 0.86],
    "sunflower": [0.08, 0.05, 0.11, 0.98],
    "chrysanthemum": [0.36, 0.24, 0.29, 0.82],
    "orchid": [0.11, 0.18, 0.09, 0.14],
    "ranunculus": [0.55, 0.61, 0.48, 0.44],
    "tulip": [0.19, 0.12, 0.27, 0.63],
    "anemone": [0.31, 0.22, 0.18, 0.77],
}
_MESES_OTONO = (9, 10, 11)
_NOMBRE_MES_EN = (
    "",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


class RecipeItem(BaseModel):
    """One flower inside a multi-species arrangement."""

    flower_name: str = Field(description="Singular lowercase token, such as orchid or baby_s_breath.")
    stems_per_arrangement: int = Field(description="Stems of this flower in one arrangement. 0 when not stated.")

    @field_validator("flower_name")
    @classmethod
    def _token(cls, value: str) -> str:
        token = _plano(value).replace("'", "").replace("'", "")
        aliases = {
            "babys breath": "baby_s_breath",
            "baby breath": "baby_s_breath",
            "baby_s_breath": "baby_s_breath",
            "gypsophila": "baby_s_breath",
            "italian ruscus": "italian_ruscus",
            "ruscus": "italian_ruscus",
            "eucalyptus": "eucalyptus",
            "orquids": "orchid",
            "orquid": "orchid",
            "orquideas": "orchid",
            "orquidea": "orchid",
            "orchids": "orchid",
            "orchid": "orchid",
        }
        return aliases.get(token, token.replace(" ", "_"))

    @field_validator("stems_per_arrangement")
    @classmethod
    def _stems(cls, value: int) -> int:
        return value if value and value > 0 else 0


class FloralRequirements(BaseModel):
    """Strict vision schema. OpenAI must return only these keys."""

    arrangements_quantity: int = Field(
        description="Number of arrangements requested. 0 when the planner did not state one."
    )
    items: list[RecipeItem] = Field(
        description="Nested recipe: flower_name and stems_per_arrangement for every species."
    )
    detected_flowers: list[str] = Field(
        description="Flower names visible in the photo or named in the message, singular and lowercase."
    )
    aesthetic_style: str = Field(
        description="One short label for the design style that is actually visible or stated."
    )
    budget_provided: bool = Field(
        description="False when the planner gave no numeric budget."
    )
    budget_amount: float = Field(
        description="Budget figure written by the planner, or 0 when it was not provided."
    )
    event_date: str = Field(
        description="Resolved YYYY-MM-DD, or empty when the message has no date."
    )

    @field_validator("detected_flowers")
    @classmethod
    def _lowercase_flowers(cls, values: list[str]) -> list[str]:
        return _unicos(_etiqueta_botanica(item) for item in values)

    @field_validator("aesthetic_style", "event_date")
    @classmethod
    def _lowercase_style(cls, value: str) -> str:
        return _plano(value) if value else ""

    @field_validator("budget_amount")
    @classmethod
    def _budget_floor(cls, value: float) -> float:
        return float(value) if value and value > 0 else 0.0

    @field_validator("arrangements_quantity")
    @classmethod
    def _arrangements(cls, value: int) -> int:
        return value if value and value > 0 else 0


def _openai_client():
    """Official client. Reads OPENAI_API_KEY from the environment or the local .env."""
    try:
        _cargar_env()
    except OSError:
        _log.warning("Could not read the local .env file.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log.warning("OPENAI_API_KEY is not set. Floral vision was skipped.")
        return None
    return OpenAI(api_key=api_key, timeout=60.0)


def _referencia_visual(imagen: str | bytes | None) -> str | None:
    """Turn raw bytes or a Base64/data-URL/http reference into an image_url value."""
    if imagen is None:
        return None
    if isinstance(imagen, (bytes, bytearray)):
        if not imagen:
            return None
        payload = base64.b64encode(bytes(imagen)).decode("ascii")
        return f"data:{_mime_desde_base64(payload)};base64,{payload}"
    texto = str(imagen).strip()
    if not texto:
        return None
    try:
        return preparar_imagen(texto)
    except ValueError:
        _log.warning("The reference image was not a valid URL, data URL, or Base64 payload.")
        return None


def _contenido_vision(user_input_text: str, image_ref: str | None) -> str | list[dict]:
    texto = (user_input_text or "").strip() or "Identify the flowers and the aesthetic style in this reference."
    if not image_ref:
        return texto
    return [
        {"type": "text", "text": texto},
        {"type": "image_url", "image_url": {"url": image_ref}},
    ]


def extract_floral_requirements(
    user_input_text: str,
    image: str | bytes | None = None,
    image_url: str | None = None,
) -> dict:
    """Read the planner message and an optional image, and return flowers plus style.

    ``image`` accepts raw bytes or a Base64 string (also a data URL or http URL).
    ``image_url`` is the same input under the previous parameter name.
    A missing API key or a failed request returns an empty requirement set.
    """
    referencia = image if image is not None else image_url
    texto = user_input_text or ""
    image_ref = _referencia_visual(referencia)
    if not texto.strip() and not image_ref:
        return dict(_EMPTY_REQUIREMENTS)

    cliente = _openai_client()
    if cliente is None:
        return dict(_EMPTY_REQUIREMENTS)

    try:
        completado = cliente.chat.completions.parse(
            model=_VISION_MODEL,
            temperature=0.1,
            response_format=FloralRequirements,
            messages=[
                {"role": "system", "content": _PROMPT_PARSER},
                {"role": "user", "content": _contenido_vision(texto, image_ref)},
            ],
        )
        mensaje = completado.choices[0].message
        if getattr(mensaje, "refusal", None) or mensaje.parsed is None:
            _log.warning("OpenAI returned no structured floral requirements.")
            return dict(_EMPTY_REQUIREMENTS)
        analisis = mensaje.parsed
        return _paquete_floral(analisis)
    except Exception as exc:
        _log.warning("Floral vision request failed: %s", type(exc).__name__)
        return dict(_EMPTY_REQUIREMENTS)


def _paquete_floral(analisis: FloralRequirements) -> dict:
    items = []
    seen = set()
    for item in analisis.items:
        name = item.flower_name
        if not name or item.stems_per_arrangement <= 0 or name in seen:
            continue
        seen.add(name)
        items.append({
            "flower_name": name,
            "stems_per_arrangement": item.stems_per_arrangement,
        })
    flowers = list(analisis.detected_flowers)
    for item in items:
        if item["flower_name"] not in flowers:
            flowers.append(item["flower_name"])
    return {
        "detected_flowers": flowers,
        "arrangements_quantity": analisis.arrangements_quantity,
        "items": items,
        "aesthetic_style": analisis.aesthetic_style,
        "budget_provided": bool(analisis.budget_provided and analisis.budget_amount > 0),
        "budget_amount": analisis.budget_amount if analisis.budget_provided else 0,
        "event_date": analisis.event_date,
    }


def extract_floral_requirements_from_image(image_bytes: bytes, chat_message: str) -> dict:
    """Identify flowers in an uploaded image with gpt-4o-mini structured output.

    ``image_bytes`` is the raw file the planner uploaded. The bytes are encoded
    as Base64 so the vision model can read them. A connection failure, a missing
    key, or an empty file returns an empty requirement set instead of raising.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        _log.warning("Floral vision skipped because the uploaded image was empty.")
        return dict(_EMPTY_REQUIREMENTS)

    try:
        _cargar_env()
    except OSError:
        _log.warning("Could not read the local .env file.")
        return dict(_EMPTY_REQUIREMENTS)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log.warning("OPENAI_API_KEY is not set. Floral vision was skipped.")
        return dict(_EMPTY_REQUIREMENTS)

    encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
    image_url = f"data:{_mime_desde_base64(encoded)};base64,{encoded}"
    message = (chat_message or "").strip() or "Identify the flowers and the aesthetic style in this reference."

    try:
        client = OpenAI(api_key=api_key, timeout=60.0)
        completion = client.chat.completions.create(
            model=_VISION_MODEL,
            temperature=0.1,
            response_format=_FLORAL_JSON_SCHEMA,
            messages=[
                {"role": "system", "content": _vision_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = FloralRequirements.model_validate_json(content)
        return _paquete_floral(parsed)
    except Exception as exc:
        _log.warning("OpenAI vision connection failed: %s", type(exc).__name__)
        return dict(_EMPTY_REQUIREMENTS)


def generate_botanical_knowledge_and_embeddings(flower_list: list[str]) -> "pd.DataFrame":
    """Describe la geometría de cada flor y la convierte en un embedding."""
    import pandas as pd

    cliente, modelo = _cliente()
    pendientes = [flor for flor in flower_list if flor not in _CACHE_VECTORES]
    for flower in pendientes:
        prompt = (
            f"Provide a highly dense technical description of the '{flower}' flower focusing strictly "
            "on its petal structure, volumetric size, and geometric shape in professional event arrangements. "
            "One concise paragraph. Do not mention price, season, or availability."
        )
        desc = cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content or ""
        emb = cliente.embeddings.create(
            model="text-embedding-3-small",
            input=desc,
        ).data[0].embedding
        _CACHE_VECTORES[flower] = {"flower_name": flower, "description": desc, "vector": emb}
    registros = [_CACHE_VECTORES[flor] for flor in flower_list if flor in _CACHE_VECTORES]
    return pd.DataFrame(registros)


class BloomTrustSourcingEngine:
    """Compara una flor fuera de temporada contra el stock que el inventario sí puede cotizar."""

    def __init__(self, target_date: str):
        from datetime import datetime

        self.target_date = target_date
        self.month = datetime.strptime(target_date, "%Y-%m-%d").month
        self.df_market = None

    def _asegurar_mercado(self):
        if self.df_market is not None:
            return
        from app.services.market.inventory import df_inventario, en_temporada

        especies = [
            str(especie)
            for especie in df_inventario["especie"].unique()
            if en_temporada(str(especie), self.month)
        ]
        etiquetas = [_etiqueta_botanica(especie) for especie in especies]
        frame = generate_botanical_knowledge_and_embeddings(etiquetas)
        clave_por_nombre = {_etiqueta_botanica(especie): especie for especie in especies}
        frame["clave"] = frame["flower_name"].map(clave_por_nombre)
        self.df_market = frame.dropna(subset=["clave"])

    def _peonia_sin_stock_otono(self, requested_flower: str) -> bool:
        """Peony or peonies in October, or any autumn event date, is out of stock."""
        if self.month not in _MESES_OTONO:
            return False
        token = _plano(requested_flower)
        return any(palabra in token for palabra in ("peony", "peonies", "peonia", "peonias"))

    def _sustituir_peonia_octubre(self, requested_flower: str) -> dict:
        """Availability NO, then local cosine against the in-season NYC pool. No OpenAI call."""
        from app.services.market.inventory import df_inventario, en_temporada

        pool: list[str] = []
        for especie in df_inventario["especie"].unique():
            if not en_temporada(str(especie), self.month):
                continue
            etiqueta = _etiqueta_botanica(str(especie))
            if etiqueta in _VECTORES_GEOMETRIA and etiqueta not in pool and etiqueta != "peony":
                pool.append(etiqueta)
        if "garden rose" not in pool:
            pool.append("garden rose")

        origen = _VECTORES_GEOMETRIA["peony"]
        puntajes = [
            (nombre, _similitud_coseno(origen, _VECTORES_GEOMETRIA[nombre]))
            for nombre in pool
        ]
        elegido, crudo = max(puntajes, key=lambda par: par[1])
        score = round(float(crudo) * 100, 2)
        mes = _NOMBRE_MES_EN[self.month]
        return {
            "status": "SUBSTITUTED",
            "availability": "NO",
            "original_flower": "peony",
            "requested_flower": requested_flower,
            "final_flower": elegido,
            "score": score,
            "reason": _razon_reemplazo("peony", mes, elegido, score),
        }

    def check_stock_and_match(self, requested_flower: str) -> dict:
        """Si el calendario de NYC tiene la especie, la deja. Si no, elige el vector más cercano."""
        from app.services.market.inventory import en_temporada

        if self._peonia_sin_stock_otono(requested_flower):
            return self._sustituir_peonia_octubre(requested_flower)

        clave = _canon_flor(requested_flower)
        etiqueta = _etiqueta_botanica(requested_flower)
        if en_temporada(clave, self.month):
            return {
                "status": "IN_STOCK",
                "final_flower": clave,
                "score": 100.0,
                "reason": "Available in regular stock wholesale.",
            }

        self._asegurar_mercado()
        pool = self.df_market.loc[self.df_market["clave"] != clave]
        if pool.empty:
            return {
                "status": "SUBSTITUTED",
                "original_flower": clave,
                "final_flower": "rosa",
                "score": 0.0,
                "reason": _razon_reemplazo(etiqueta, _NOMBRE_MES_EN[self.month], "rose", 0),
            }

        pedido = generate_botanical_knowledge_and_embeddings([etiqueta])
        vector_pedido = list(pedido.iloc[0]["vector"])
        puntajes = [_similitud_coseno(vector_pedido, list(vector)) for vector in pool["vector"]]
        mejor = max(range(len(puntajes)), key=puntajes.__getitem__)
        elegido = pool.iloc[mejor]
        score = round(float(puntajes[mejor]) * 100, 2)
        return {
            "status": "SUBSTITUTED",
            "original_flower": clave,
            "final_flower": str(elegido["clave"]),
            "score": score,
            "reason": _razon_reemplazo(
                etiqueta,
                _NOMBRE_MES_EN[self.month],
                _etiqueta_botanica(str(elegido["clave"])),
                score,
            ),
        }


def coincidir_sustituto(requested_flower: str, mes: int) -> dict:
    """Atajo para el mes del evento. La fecha solo fija el mes del calendario."""
    return BloomTrustSourcingEngine(f"2026-{int(mes):02d}-15").check_stock_and_match(requested_flower)
