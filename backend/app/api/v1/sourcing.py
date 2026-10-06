"""Real sourcing endpoint: vision, stem count, and three wholesale packages."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from datetime import date, timedelta

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from openai import OpenAI

from app.services.agents.multimodal_agent import _VISION_MODEL, _cargar_env, extract_floral_requirements_from_image
from app.services.market.inventory import df_inventario
from app.valuation.engine import (
    BloomTrustCostEngine,
    BloomTrustPredictiveForecaster,
    BloomTrustVolumeEstimator,
)

router = APIRouter()
_log = logging.getLogger("bloomtrust.sourcing")

_IMAGE_LIMIT = 8 * 1024 * 1024
_FALLBACK_WHOLESALE = 3.50
_SUPPLIER = {
    "Premium": "Chelsea Cut Flowers",
    "Standard": "Midtown Stem Supply",
    "Opportunity": "Hunts Point Surplus",
}
_INVENTORY_KEY = {
    "rose": "rosa",
    "rosa": "rosa",
    "orchid": "orquidea",
    "orquidea": "orquidea",
    "peony": "peonia",
    "peonia": "peonia",
    "hydrangea": "hortensia",
    "hortensia": "hortensia",
    "tulip": "tulipan",
    "tulipan": "tulipan",
    "ranunculus": "ranunculo",
    "ranunculo": "ranunculo",
    "anemone": "anemona",
    "anemona": "anemona",
    "dahlia": "dalia",
    "dalia": "dalia",
    "sunflower": "girasol",
    "girasol": "girasol",
    "chrysanthemum": "crisantemo",
    "crisantemo": "crisantemo",
    "carnation": "clavel",
    "clavel": "clavel",
}
_DISPLAY = {
    "rose": "Rose",
    "orchid": "Orchid",
    "peony": "Peony",
    "hydrangea": "Hydrangea",
    "tulip": "Tulip",
    "ranunculus": "Ranunculus",
    "anemone": "Anemone",
    "dahlia": "Dahlia",
    "sunflower": "Sunflower",
    "chrysanthemum": "Chrysanthemum",
    "carnation": "Carnation",
}
_MONTHS = {
    "january": 1, "enero": 1,
    "february": 2, "febrero": 2,
    "march": 3, "marzo": 3,
    "april": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "june": 6, "junio": 6,
    "july": 7, "julio": 7,
    "august": 8, "agosto": 8,
    "september": 9, "septiembre": 9, "setiembre": 9,
    "october": 10, "octubre": 10,
    "november": 11, "noviembre": 11,
    "december": 12, "diciembre": 12,
}
_MONTH_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
_DESIGN_HINTS = (
    ("floral arch", "floral_arch"),
    ("arco floral", "floral_arch"),
    ("bridal bouquet", "bridal_bouquet"),
    ("bouquet de novia", "bridal_bouquet"),
    ("high centerpiece", "high_centerpiece"),
    ("centro de mesa alto", "high_centerpiece"),
    ("low centerpiece", "low_centerpiece"),
    ("centro de mesa bajo", "low_centerpiece"),
    ("high density", "high_centerpiece"),
    ("arch", "floral_arch"),
    ("arco", "floral_arch"),
    ("bouquet", "bridal_bouquet"),
    ("runner", "high_centerpiece"),
)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _catalog_key(species: str) -> str:
    token = _fold(species)
    return _INVENTORY_KEY.get(token, token)


def _display_name(species: str) -> str:
    token = _fold(species)
    if token in _DISPLAY:
        return _DISPLAY[token]
    text = species.strip()
    return text[:1].upper() + text[1:] if text else "Stems"


def _standard_wholesale(species: str) -> float:
    """Standard-band stem price. Unknown species fall back to 3.50."""
    key = _catalog_key(species)
    rows = df_inventario.loc[df_inventario["especie"] == key]
    standard = rows.loc[rows["banda"] == "standard"] if not rows.empty else rows
    if not standard.empty:
        return float(standard.iloc[0]["precio_tallo"])
    if not rows.empty:
        return float(rows.iloc[0]["precio_tallo"])
    return _FALLBACK_WHOLESALE


def _design_type(message: str) -> str:
    folded = _fold(message)
    for hint, design in _DESIGN_HINTS:
        if hint in folded:
            return design
    return "low_centerpiece"


def _quantity(message: str) -> int:
    match = re.search(
        r"\b(\d+)\s*(?:tables|table|mesas|mesa|arrangements|arrangement|centerpieces|centerpiece)\b",
        _fold(message),
    )
    if not match:
        return 1
    count = int(match.group(1))
    return count if 1 <= count <= 200 else 1


def _stated_stems(raw: str, message: str) -> int | None:
    """Exact stem count from the form field, or from a phrase such as '150 stems'."""
    cleaned = (raw or "").strip().replace(",", "")
    if re.fullmatch(r"\d+", cleaned):
        count = int(cleaned)
        return count if count > 0 else None
    match = re.search(r"\b(\d+)\s*(?:stems?|tallos?)\b", _fold(message))
    if not match:
        return None
    count = int(match.group(1))
    return count if count > 0 else None


_BUDGET_REFUSAL = (
    "no budget",
    "i don't know the budget",
    "i dont know the budget",
    "i do not know the budget",
    "don't know the budget",
    "dont know the budget",
    "do not know the budget",
    "unknown budget",
    "without a budget",
)
_REQUEST_LABEL = {
    "rose": "Roses",
    "orchid": "Orchids",
    "peony": "Peonies",
    "hydrangea": "Hydrangeas",
    "tulip": "Tulips",
    "ranunculus": "Ranunculus",
    "anemone": "Anemones",
    "dahlia": "Dahlias",
    "sunflower": "Sunflowers",
    "chrysanthemum": "Chrysanthemums",
    "carnation": "Carnations",
    "rosa": "Roses",
    "orquidea": "Orchids",
    "peonia": "Peonies",
    "hortensia": "Hydrangeas",
}


def _iso_date(value: str) -> str | None:
    token = (value or "").strip()
    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", token):
        return None
    try:
        date.fromisoformat(token)
    except ValueError:
        return None
    return token


def _relative_date(message: str, today: date | None = None) -> str | None:
    """Turn phrases such as 'in two weeks' into a calendar date from today."""
    hoy = today or date.today()
    folded = _fold(message or "")
    if not folded:
        return None
    if "in two weeks" in folded or "in 2 weeks" in folded:
        return (hoy + timedelta(days=14)).isoformat()
    weeks = re.search(r"\bin\s+(\d+)\s+weeks?\b", folded)
    if weeks:
        return (hoy + timedelta(days=7 * int(weeks.group(1)))).isoformat()
    days = re.search(r"\bin\s+(\d+)\s+days?\b", folded)
    if days:
        return (hoy + timedelta(days=int(days.group(1)))).isoformat()
    if "next week" in folded:
        return (hoy + timedelta(days=7)).isoformat()
    if "tomorrow" in folded:
        return (hoy + timedelta(days=1)).isoformat()
    return None


def _event_date(message: str, extracted: str = "", today: date | None = None) -> str:
    explicit = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message or "")
    if explicit and _iso_date(explicit.group(1)):
        return explicit.group(1)
    relative = _relative_date(message, today)
    if relative:
        return relative
    resolved = _iso_date(extracted)
    if resolved:
        return resolved
    tokens = re.split(r"[^a-z]+", _fold(message or ""))
    year = (today or date.today()).year
    for token in tokens:
        month = _MONTHS.get(token)
        if month:
            return f"{year}-{month:02d}-15"
    return (today or date.today()).isoformat()


def _form_budget(raw: str) -> float | None:
    token = (raw or "").strip().replace(",", "").replace("$", "")
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if value > 0 else None


def _message_budget(message: str) -> float | None:
    source = (message or "").replace(",", "")
    labeled = re.search(r"(?:budget|presupuesto|usd|\$)\s*[:.]?\s*(\d+(?:\.\d+)?)", source, re.I)
    if not labeled:
        labeled = re.search(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars)", source, re.I)
    if labeled:
        value = float(labeled.group(1))
        return value if value > 0 else None
    bare = [
        float(item)
        for item in re.findall(r"(?:^|[^\d])(\d{3,7}(?:\.\d+)?)(?!\d)", source)
        if not 1900 <= float(item) <= 2100
    ]
    return bare[0] if bare else None


def _budget_is_refused(message: str) -> bool:
    folded = _fold(message or "")
    return any(phrase in folded for phrase in _BUDGET_REFUSAL)


def _quoted_budget(message: str, form_raw: str, requirements: dict) -> float | None:
    """A stated number wins. 'no budget' and an empty field do not reach the cost engine."""
    if _budget_is_refused(message):
        return None
    stated = _form_budget(form_raw) or _message_budget(message)
    if stated:
        return stated
    if not requirements.get("budget_provided"):
        return None
    amount = requirements.get("budget_amount")
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    if value <= 0 or not _message_budget(message):
        return None
    return value


def _request_label(detected: list[str]) -> str:
    labels = []
    for item in detected:
        token = _fold(item)
        label = _REQUEST_LABEL.get(token) or _REQUEST_LABEL.get(_catalog_key(token))
        if not label:
            visible = _display_name(item)
            label = visible if visible.endswith("s") else f"{visible}s"
        if label not in labels:
            labels.append(label)
    if not labels:
        return "Hydrangeas"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def _respuesta_sin_presupuesto(detected: list[str]) -> dict:
    text = (
        f"🌸 I have successfully identified your request for {_request_label(detected)}! "
        "However, to generate your trilateral packages (Premium, Standard, Opportunity), "
        "I need you to provide an estimated budget figure and a specific date. "
        "Please let me know your target investment so I can balance your stem counts."
    )
    return {"type": "text", "text": text, "detected_flowers": detected}


_INTENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "planner_intent",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["GREETING", "CREATIVE_ADVICE", "SOURCING"],
                    "description": "GREETING has priority over every other label.",
                },
                "detected_flowers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Singular lowercase flower names named in the message. Empty for a greeting.",
                },
                "aesthetic_style": {"type": "string"},
            },
            "required": ["intent", "detected_flowers", "aesthetic_style"],
        },
    },
}
_FLOWER_TOKENS = (
    ("hydrangea", "hydrangea"), ("hydrangeas", "hydrangea"), ("hortensia", "hydrangea"),
    ("orchid", "orchid"), ("orchids", "orchid"), ("orquidea", "orchid"),
    ("peony", "peony"), ("peonies", "peony"), ("peonia", "peony"),
    ("rose", "rose"), ("roses", "rose"), ("rosa", "rose"),
    ("tulip", "tulip"), ("tulips", "tulip"), ("tulipan", "tulip"),
    ("ranunculus", "ranunculus"), ("ranunculo", "ranunculus"),
    ("carnation", "carnation"), ("carnations", "carnation"), ("clavel", "carnation"),
    ("anemone", "anemone"), ("anemones", "anemone"),
    ("dahlia", "dahlia"), ("dahlias", "dahlia"),
    ("sunflower", "sunflower"), ("sunflowers", "sunflower"),
    ("chrysanthemum", "chrysanthemum"), ("chrysanthemums", "chrysanthemum"),
)


def _es_saludo(message: str) -> bool:
    """A pure hello, including 'Hi, I am Pam', outranks sourcing and advice."""
    folded = re.sub(r"[^a-z\s]", " ", _fold(message or ""))
    folded = " ".join(folded.split())
    if not folded:
        return False
    return re.fullmatch(
        r"(?:hi|hello|hey|good morning|good afternoon|good evening)(?: there)?"
        r"(?: i am| im| my name is)?(?: [a-z]+){0,2}",
        folded,
    ) is not None


def _nombre_saludo(message: str) -> str:
    match = re.search(r"\b(?:i am|i'm|im|my name is)\s+([A-Za-z]+)", message or "", re.I)
    if not match:
        return ""
    nombre = match.group(1)
    return nombre[:1].upper() + nombre[1:]


def _es_consejo(message: str) -> bool:
    folded = _fold(message or "")
    return any(phrase in folded for phrase in (
        "i don't know what to do",
        "i dont know what to do",
        "do not know what to do",
        "need inspiration",
        "low budget",
        "creative advice",
    ))


def _flores_en_texto(message: str) -> list[str]:
    folded = _fold(message or "")
    halladas = []
    for token, canon in _FLOWER_TOKENS:
        if re.search(rf"\b{token}\b", folded) and canon not in halladas:
            halladas.append(canon)
    return halladas


def _clasificar_local(message: str) -> dict:
    if _es_saludo(message):
        return {"intent": "GREETING", "detected_flowers": [], "aesthetic_style": ""}
    flores = _flores_en_texto(message)
    if _es_consejo(message) and not (_message_budget(message) and flores):
        return {"intent": "CREATIVE_ADVICE", "detected_flowers": flores, "aesthetic_style": ""}
    return {"intent": "SOURCING", "detected_flowers": flores, "aesthetic_style": ""}


def _prompt_intencion() -> str:
    return (
        "You classify a BloomTrust planner message for the New York wholesale flower desk. "
        "Return JSON with intent, detected_flowers, and aesthetic_style. "
        "GREETING has maximum priority. A hello that only introduces the speaker, "
        "such as 'Hi, I am Pam', is GREETING even though it contains a name. "
        "Do not treat that as a sourcing request and do not ask for a photo. "
        "CREATIVE_ADVICE is for inspiration, 'I don't know what to do', or a low-budget design question "
        "that does not yet name a numeric budget. "
        "SOURCING is a request to price stems, dates, or named flowers. "
        "detected_flowers uses singular lowercase English names stated in the text. "
        "Do not calculate prices or stem counts."
    )


def classify_intent(message: str) -> dict:
    """Text-only gpt-4o-mini intent call. Greetings never depend on the image pipeline."""
    local = _clasificar_local(message)
    if local["intent"] == "GREETING":
        return local
    try:
        _cargar_env()
    except OSError:
        _log.warning("Could not read the local .env file.")
        return local
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log.warning("OPENAI_API_KEY is not set. Intent fell back to the local reading.")
        return local
    try:
        client = OpenAI(api_key=api_key, timeout=30.0)
        completion = client.chat.completions.create(
            model=_VISION_MODEL,
            temperature=0,
            response_format=_INTENT_SCHEMA,
            messages=[
                {"role": "system", "content": _prompt_intencion()},
                {"role": "user", "content": message},
            ],
        )
        crudo = json.loads(completion.choices[0].message.content or "{}")
        intent = str(crudo.get("intent") or "")
        if intent not in {"GREETING", "CREATIVE_ADVICE", "SOURCING"}:
            return local
        flores = [
            _fold(str(item))
            for item in crudo.get("detected_flowers") or []
            if str(item).strip()
        ]
        return {
            "intent": intent,
            "detected_flowers": flores or local["detected_flowers"],
            "aesthetic_style": _fold(str(crudo.get("aesthetic_style") or "")),
        }
    except Exception as exc:
        _log.warning("Text intent request failed: %s", type(exc).__name__)
        return local


def _respuesta_saludo(message: str) -> dict:
    nombre = _nombre_saludo(message)
    saludo = f"Hi {nombre}." if nombre else "Hi."
    return {
        "type": "text",
        "intent": "GREETING",
        "text": (
            f"{saludo} I am your BloomTrust Sourcing Broker. "
            "Whenever you are ready, share the flowers, a date, and a budget, "
            "and I will compare Premium, Standard, and Opportunity."
        ),
    }


def _respuesta_creativa(detected: list[str]) -> dict:
    return {
        "type": "text",
        "intent": "CREATIVE_ADVICE",
        "detected_flowers": detected,
        "text": (
            "If the recipe is still open, build a low packed runner: one hero bloom carries the color, "
            "and a second layer fills every gap so the table reads as a single cloud. "
            "Add the liquidation flowers from the Opportunity tier so the budget stretches without losing density."
        ),
    }


async def _leer_imagen(file: UploadFile | None) -> bytes | None:
    if file is None or not str(file.filename or "").strip():
        return None
    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
        raise HTTPException(status_code=415, detail="The upload must be an image.")
    data = await file.read(_IMAGE_LIMIT + 1)
    if len(data) > _IMAGE_LIMIT:
        raise HTTPException(status_code=413, detail="The image exceeds the 8 MB limit.")
    return data or None


def _financial_rows(variety: str, packages: dict) -> list[dict]:
    rows = []
    for tier, package in packages.items():
        rows.append(
            {
                "tier": tier,
                "variety": variety,
                "supplier": _SUPPLIER[tier],
                "unit_price_usd": package["unit_price_USD"],
                "stems": package["allocated_stems"],
                "subtotal_usd": package["package_total_USD"],
                "remaining_budget_usd": package["remaining_budget_USD"],
                "tier_concept": package["tier_concept"],
            }
        )
    return rows


@router.post("/analyze-event", redirect_slashes=False))
async def analyze_event(
    file: UploadFile | None = File(None),
    max_budget: str = Form("", description="Maximum event budget in USD. Empty when the planner has no figure."),
    message: str = Form("", description="Planner message from the chat."),
    required_stems: str = Form("", description="Exact stem count when the planner wrote one, such as 150 stems."),
) -> dict:
    """Price a text note, a reference image, or both. A greeting never requires a photo."""
    texto = (message or "").strip()
    image_bytes = await _leer_imagen(file)
    if image_bytes is None and _es_saludo(texto):
        return _respuesta_saludo(texto)
    if image_bytes is None and not texto:
        return _respuesta_saludo("")

    if image_bytes is None:
        classified = await asyncio.to_thread(classify_intent, texto)
        intent = classified.get("intent")
        detected = [str(item) for item in classified.get("detected_flowers") or [] if str(item).strip()]
        if intent == "GREETING":
            return _respuesta_saludo(texto)
        if intent == "CREATIVE_ADVICE":
            return _respuesta_creativa(detected)
        requirements = {
            "detected_flowers": detected,
            "aesthetic_style": classified.get("aesthetic_style") or "",
            "budget_provided": False,
            "event_date": "",
        }
    else:
        requirements = await asyncio.to_thread(
            extract_floral_requirements_from_image,
            image_bytes,
            texto,
        )
        detected = [str(item).strip().lower() for item in requirements.get("detected_flowers") or [] if str(item).strip()]

    budget = _quoted_budget(texto, max_budget, requirements)
    if budget is None:
        return _respuesta_sin_presupuesto(detected)

    aesthetic_style = str(requirements.get("aesthetic_style") or "")
    primary = detected[0] if detected else "rose"
    event_date = _event_date(texto, str(requirements.get("event_date") or ""))
    design = _design_type(texto)
    quantity = _quantity(texto)
    stated = _stated_stems(required_stems, texto)
    try:
        sizing = BloomTrustVolumeEstimator().estimate_required_stems(design, quantity)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if stated is not None:
        sizing["total_stems"] = stated
        sizing["required_stems"] = stated

    wholesale = _standard_wholesale(primary)
    try:
        packages = BloomTrustCostEngine(
            target_flower=primary,
            base_wholesale_price=wholesale,
        ).generate_packages(budget, int(sizing["total_stems"]))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    demand = BloomTrustPredictiveForecaster(event_date).calculate_market_volatility()
    month_number = int(event_date[5:7])
    month_name = _MONTH_NAME[month_number]
    surge = int(demand["demand_surge_percentage"])
    variety = _display_name(primary)
    alert = (
        f"Reserve stems this week. Predictive demand for {month_name} "
        f"is {surge}% on the fixed New York event calendar."
    )
    return {
        "detected_flowers": detected,
        "aesthetic_style": aesthetic_style,
        "intent": "SOURCING",
        "botanical": {
            "species": [_display_name(item) for item in detected],
            "primary_species": variety,
            "catalog_key": _catalog_key(primary),
            "aesthetic_style": aesthetic_style,
            "species_confirmed": bool(detected),
        },
        "sizing": sizing,
        "demand_alert": {
            "event_date": event_date,
            "month": month_name,
            "demand_surge_percentage": surge,
            "market_justification": demand["market_justification"],
            "alert": alert,
        },
        "type": "quote",
        "base_wholesale_price_usd": wholesale,
        "max_budget": budget,
        "packages": packages,
        "financials": _financial_rows(variety, packages),
    }
