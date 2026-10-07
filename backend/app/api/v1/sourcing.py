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

from app.services.agents.multimodal_agent import (
    _VISION_MODEL,
    _cargar_env,
    coincidir_sustituto,
    extract_floral_requirements,
    extract_floral_requirements_from_image,
)
from app.services.market.inventory import df_inventario
from app.valuation.engine import (
    BloomTrustCareEngine,
    BloomTrustCostEngine,
    BloomTrustDesignConsultant,
    BloomTrustPredictiveForecaster,
    BloomTrustTrendsEngine,
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
    "garden rose": "Garden Rose",
    "baby_s_breath": "Baby's Breath",
    "eucalyptus": "Eucalyptus",
    "italian_ruscus": "Italian Ruscus",
    "lisianthus": "Lisianthus",
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


def _canon_especie(name: str) -> str:
    """One key per species. Orchid, orchids and orquidea all stay orchid."""
    token = _fold(name or "").replace("'", "").replace("'", "").replace("_", " ")
    token = " ".join(token.split())
    if not token:
        return ""
    aliases = (
        ("phalaenopsis", "orchid"),
        ("orquids", "orchid"),
        ("orquid", "orchid"),
        ("orquideas", "orchid"),
        ("orquidea", "orchid"),
        ("eucalipto", "eucalyptus"),
        ("eucalyptus", "eucalyptus"),
        ("baby s breath", "baby_s_breath"),
        ("babys breath", "baby_s_breath"),
        ("baby breath", "baby_s_breath"),
        ("gypsophila", "baby_s_breath"),
    )
    for phrase, canon in (*aliases, *_FLOWER_TOKENS):
        if token == phrase or token == canon.replace("_", " "):
            return canon
    return token.replace(" ", "_")


def _catalog_key(species: str) -> str:
    token = _canon_especie(species) or _fold(species)
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


def _integer_quantity(message: str) -> int:
    """First stem count in the note. Money figures and years are not stem counts."""
    text = (message or "").replace(",", "")
    explicit = re.search(r"\b(\d+)\s*(?:stems?|tallos?)\b", _fold(text))
    if explicit:
        return int(explicit.group(1))
    stripped = re.sub(
        r"(?:presupuesto|budget|usd|\$)\s*[:.]?\s*\d+(?:\.\d+)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:usd|dollars|dolares|dólares)\b",
        " ",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\b(?:19|20)\d{2}\b", " ", stripped)
    stripped = re.sub(r"\d+\.\d+", " ", stripped)
    found = re.search(r"\b(\d+)\b", stripped)
    return int(found.group(1)) if found else 0


def _stated_stems(raw: str, message: str) -> int:
    """Form field wins, including 0 when the planner named no quantity. Never invent a recipe count."""
    cleaned = (raw or "").strip().replace(",", "")
    if re.fullmatch(r"\d+", cleaned):
        return int(cleaned)
    return _integer_quantity(message)


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
    without_volume = re.sub(r"\b\d+(?:\.\d+)?\s*(?:stems?|tallos?)\b", " ", source, flags=re.I)
    bare = [
        float(item)
        for item in re.findall(r"(?:^|[^\d])(\d{3,7}(?:\.\d+)?)(?!\d)", without_volume)
        if not 1900 <= float(item) <= 2100
    ]
    return bare[0] if bare else None


def _explicit_money(message: str) -> float | None:
    """A dollar cap stated as budget, USD, or $. A stem count is not money."""
    source = (message or "").replace(",", "")
    labeled = re.search(r"(?:budget|presupuesto|usd|\$)\s*[:.]?\s*(\d+(?:\.\d+)?)", source, re.I)
    if not labeled:
        labeled = re.search(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars)\b", source, re.I)
    if not labeled:
        return None
    value = float(labeled.group(1))
    return value if value > 0 else None


def _presupuesto_motor(message: str, extracted: float | None = None) -> float:
    """Value passed to BloomTrustCostEngine. 0.0 opens the fixed-volume quote."""
    explicit = _explicit_money(message)
    if explicit is None:
        return 0.0
    try:
        suggested = float(extracted) if extracted is not None else 0.0
    except (TypeError, ValueError):
        suggested = 0.0
    if suggested > 0 and abs(suggested - explicit) < 0.01:
        return suggested
    return explicit


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


def _care_data(flowers: list[str]) -> list[dict]:
    """One conditioning card per detected species, using the Colab care dictionary."""
    engine = BloomTrustCareEngine()
    cards = []
    seen = set()
    for raw in flowers or []:
        token = _fold(str(raw))
        if not token or token in seen:
            continue
        seen.add(token)
        guide = engine.query_botanical_guide(token)
        cards.append({
            "flower": _display_name(token),
            "scientific_name": guide.get("scientific_name") or "",
            "hydration": guide.get("hydration") or "",
            "temperature": guide.get("temperature") or "",
            "alert": guide.get("alert") or "",
        })
    return cards


def _respuesta_sin_presupuesto(detected: list[str]) -> dict:
    text = (
        f"🌸 I have successfully identified your request for {_request_label(detected)}! "
        "However, to generate your trilateral packages (Premium, Standard, Opportunity), "
        "I need you to provide an estimated budget figure and a specific date. "
        "Please let me know your target investment so I can balance your stem counts."
    )
    return {"type": "text", "text": text, "detected_flowers": detected, "care_data": _care_data(detected)}


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
                    "enum": ["GREETING", "CREATIVE_ADVICE", "SOURCING", "OUT_OF_DOMAIN"],
                    "description": "GREETING has priority over every other label. OUT_OF_DOMAIN rejects anything outside flowers, events, and NYC wholesale.",
                },
                "detected_flowers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Singular lowercase flower names named in the message. Empty for a greeting.",
                },
                "aesthetic_style": {"type": "string"},
                "max_budget": {
                    "type": "number",
                    "description": "Explicit monetary cap in USD. Use 0.0 when the planner did not state a dollar budget. A stem count such as 100 stems is not a budget.",
                },
            },
            "required": ["intent", "detected_flowers", "aesthetic_style", "max_budget"],
        },
    },
}
_FLOWER_TOKENS = (
    ("baby's breath", "baby_s_breath"),
    ("babys breath", "baby_s_breath"),
    ("baby breath", "baby_s_breath"),
    ("gypsophila", "baby_s_breath"),
    ("italian ruscus", "italian_ruscus"),
    ("eucalyptus", "eucalyptus"),
    ("garden roses", "garden rose"),
    ("garden rose", "garden rose"),
    ("hydrangea", "hydrangea"), ("hydrangeas", "hydrangea"), ("hortensia", "hydrangea"),
    ("orquids", "orchid"), ("orquid", "orchid"),
    ("orchid", "orchid"), ("orchids", "orchid"), ("orquideas", "orchid"), ("orquidea", "orchid"),
    ("peony", "peony"), ("peonies", "peony"), ("peonias", "peony"), ("peonia", "peony"),
    ("carnation", "carnation"), ("carnations", "carnation"), ("claveles", "carnation"), ("clavel", "carnation"),
    ("lisianthus", "lisianthus"),
    ("rose", "rose"), ("roses", "rose"), ("rosas", "rose"), ("rosa", "rose"),
    ("tulip", "tulip"), ("tulips", "tulip"), ("tulipan", "tulip"),
    ("ranunculus", "ranunculus"), ("ranunculo", "ranunculus"),
    ("anemone", "anemone"), ("anemones", "anemone"),
    ("dahlia", "dahlia"), ("dahlias", "dahlia"),
    ("sunflower", "sunflower"), ("sunflowers", "sunflower"),
    ("chrysanthemum", "chrysanthemum"), ("chrysanthemums", "chrysanthemum"),
)


def _receta_local(message: str) -> dict | None:
    """Read '5 arrangements with 2 orchids, 2 baby's breath and 4 eucalyptus' without the model."""
    folded = _fold(message or "")
    counted = re.search(
        r"\b(\d+)\s*(?:arrangements?|arreglos?|centerpieces?|bouquets?|ramos?)\b",
        folded,
    )
    if not counted:
        return None
    arrangements = int(counted.group(1))
    if arrangements <= 0:
        return None
    found_items = []
    seen = set()
    for token, canon in _FLOWER_TOKENS:
        if canon in seen:
            continue
        before = re.search(rf"\b(\d+)\s+(?:stems?\s+)?(?:of\s+|de\s+)?{re.escape(token)}\b", folded)
        after = re.search(rf"\b{re.escape(token)}\b\s*(?:\(|:|x|×)\s*(\d+)", folded)
        found = before or after
        if not found:
            continue
        per = int(found.group(1))
        if per <= 0:
            continue
        seen.add(canon)
        found_items.append((found.start(), {"flower_name": canon, "stems_per_arrangement": per}))
    items = [item for _, item in sorted(found_items, key=lambda pair: pair[0])]
    if not items:
        return None
    return {"arrangements_quantity": arrangements, "items": items, "aesthetic_style": ""}


def _receta_desde_requisitos(requirements: dict | None) -> dict | None:
    if not requirements:
        return None
    items = []
    seen = set()
    for raw in requirements.get("items") or []:
        if not isinstance(raw, dict):
            continue
        name = _canon_especie(str(raw.get("flower_name") or raw.get("flower") or raw.get("name") or ""))
        per = int(raw.get("stems_per_arrangement") or raw.get("stems") or raw.get("quantity") or 0)
        if not name or per <= 0 or name in seen:
            continue
        seen.add(name)
        items.append({"flower_name": name, "stems_per_arrangement": per})
    arrangements = int(requirements.get("arrangements_quantity") or 0)
    if arrangements <= 0 or not items:
        return None
    return {
        "arrangements_quantity": arrangements,
        "items": items,
        "aesthetic_style": str(requirements.get("aesthetic_style") or ""),
    }


def _acumular_costo_receta(arrangements_quantity: int, items: list[dict]) -> dict:
    """Add every species. Orchid is not optional when it is in the list."""
    arrangements = int(arrangements_quantity)
    total_base_price = 0.0
    total_stems = 0
    rows = []
    for item in list(items):
        name = _canon_especie(str(item.get("flower_name") or ""))
        per = int(item.get("stems_per_arrangement") or 0)
        if not name or per <= 0 or arrangements <= 0:
            continue
        item_total_stems = per * arrangements
        item_base_price = _standard_wholesale(name)
        subtotal = item_base_price * item_total_stems
        total_base_price += subtotal
        total_stems += item_total_stems
        rows.append({
            "flower_name": name,
            "display_name": _display_name(name),
            "stems_per_arrangement": per,
            "total_stems": item_total_stems,
            "unit_price_usd": round(item_base_price, 2),
            "premium_subtotal_usd": round(subtotal * 1.5, 2),
            "standard_subtotal_usd": round(subtotal, 2),
            "opportunity_subtotal_usd": round(subtotal * 0.7, 2),
        })
    return {
        "total_base_price": round(total_base_price, 2),
        "total_stems": total_stems,
        "items": rows,
    }


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


def _es_consulta_peonia(nombre: str) -> bool:
    token = _fold(nombre or "")
    return any(palabra in token for palabra in ("peony", "peonies", "peonia", "peonias"))


def _match_peonia_otono(nombres: list[str], mes: int) -> dict | None:
    """Run the October peony override once. Other species keep their own stock path."""
    if not any(_es_consulta_peonia(nombre) for nombre in nombres):
        return None
    match = coincidir_sustituto("peony", mes)
    if match.get("availability") != "NO":
        return None
    return match


def _flores_en_texto(message: str) -> list[str]:
    folded = _fold(message or "")
    ocupado = [False] * len(folded)
    halladas = []
    for token, canon in _FLOWER_TOKENS:
        for match in re.finditer(rf"\b{re.escape(token)}\b", folded):
            if any(ocupado[match.start():match.end()]):
                continue
            for index in range(match.start(), match.end()):
                ocupado[index] = True
            if canon not in halladas:
                halladas.append(canon)
    return halladas


_DOMAIN_WORDS = (
    "flower", "flowers", "floral", "bloom", "blooms", "bouquet", "stem", "stems", "tallo", "tallos",
    "wedding", "gala", "event", "events", "centerpiece", "centrepiece", "wholesale", "chelsea",
    "greenery", "filler", "fillers", "arrangement", "bride", "ceremony", "greenhouse", "sourcing",
    "tablescape", "vase", "garland", "boutonniere", "corsage", "reception", "planner", "market",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)
_RECHAZO_DOMINIO = "I am sorry, I can only assist you with flower inquiries and event sourcing analytics."


def _en_dominio_floral(message: str) -> bool:
    """Flowers, event quoting, and the NYC wholesale desk stay in domain."""
    if _flores_en_texto(message) or _explicit_money(message):
        return True
    folded = _fold(message or "")
    return any(re.search(rf"\b{re.escape(word)}\b", folded) for word in _DOMAIN_WORDS)


def _clasificar_local(message: str) -> dict:
    presupuesto = _presupuesto_motor(message)
    if _es_saludo(message):
        return {"intent": "GREETING", "detected_flowers": [], "aesthetic_style": "", "max_budget": 0.0}
    flores = _flores_en_texto(message)
    if _es_consejo(message) and not (_explicit_money(message) and flores):
        return {"intent": "CREATIVE_ADVICE", "detected_flowers": flores, "aesthetic_style": "", "max_budget": presupuesto}
    if not flores and not _en_dominio_floral(message):
        return {"intent": "OUT_OF_DOMAIN", "detected_flowers": [], "aesthetic_style": "", "max_budget": 0.0}
    return {"intent": "SOURCING", "detected_flowers": flores, "aesthetic_style": "", "max_budget": presupuesto}


def _prompt_intencion() -> str:
    return (
        "You classify a BloomTrust planner message for the New York wholesale flower desk. "
        "Return JSON with intent, detected_flowers, and aesthetic_style. "
        "GREETING has maximum priority. A hello that only introduces the speaker, "
        "such as 'Hi, I am Pam', is GREETING even though it contains a name. "
        "Do not treat that as a sourcing request and do not ask for a photo. "
        "CREATIVE_ADVICE is for inspiration, 'I don't know what to do', or a low-budget design question "
        "that does not yet name a numeric budget. "
        "SOURCING is a request to price stems, dates, or named flowers for an event or the NYC wholesale market. "
        "OUT_OF_DOMAIN is mandatory when the text is not about the floral industry, event quoting, "
        "or the New York wholesale flower market. Weather, sports, software, recipes, jokes, and general chat "
        "are OUT_OF_DOMAIN. A flower name, a stem count, an event date, greenery, or a wholesale budget "
        "is never OUT_OF_DOMAIN. "
        "detected_flowers uses singular lowercase English names stated in the text. "
        "'garden rose' is the species garden rose, not rose. "
        "max_budget is an explicit dollar cap. Set max_budget to 0.0 when the planner "
        "does not state a monetary budget. 'what about 100 stems of peonies?' has no budget, so max_budget is 0.0. "
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
        if intent not in {"GREETING", "CREATIVE_ADVICE", "SOURCING", "OUT_OF_DOMAIN"}:
            return local
        if intent == "OUT_OF_DOMAIN" and (local["detected_flowers"] or _en_dominio_floral(message)):
            intent = local["intent"] if local["intent"] != "OUT_OF_DOMAIN" else "SOURCING"
        flores_modelo = [
            _fold(str(item))
            for item in crudo.get("detected_flowers") or []
            if str(item).strip()
        ]
        flores = list(local["detected_flowers"])
        for item in flores_modelo:
            if item == "rose" and "garden rose" in flores:
                continue
            if item not in flores:
                flores.append(item)
        return {
            "intent": intent,
            "detected_flowers": flores,
            "aesthetic_style": _fold(str(crudo.get("aesthetic_style") or "")),
            "max_budget": _presupuesto_motor(message, crudo.get("max_budget")),
        }
    except Exception as exc:
        _log.warning("Text intent request failed: %s", type(exc).__name__)
        return local


def _respuesta_saludo(message: str) -> dict:
    nombre = _nombre_saludo(message)
    saludo = f"Hi {nombre}." if nombre else "Hi."
    texto = (
        f"{saludo} I am your BloomTrust Sourcing Broker. "
        "Whenever you are ready, share the flowers, a date, and a budget, "
        "and I will compare Premium, Standard, and Opportunity."
    )
    return {"type": "text", "intent": "GREETING", "text": texto, "bot_response": texto}


def _tip_por_especie(species: str) -> str:
    nombre = (species or "this species").strip() or "this species"
    return (
        f"✨ Proactive Design Tip: Based on recent NYC luxury trends, {nombre} is frequently designed "
        "alongside premium greenery or white fillers. Would you like to check current market "
        "availability for those complementary items?"
    )


def _inspiracion_respaldo() -> str:
    return (
        "Tell me the mood of the event and I will pair the hero bloom with premium greenery "
        "and white fillers from the Chelsea Flower Market."
    )


def _consultar_diseno(message: str, species: str, modo: str) -> str:
    """Live design note from gpt-4o-mini. A local sentence covers a missing key or a failed call."""
    respaldo = _tip_por_especie(species) if modo == "quote" else _inspiracion_respaldo()
    try:
        _cargar_env()
    except OSError:
        return respaldo
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return respaldo
    if modo == "quote":
        system = (
            "You are the BloomTrust design consultant for the Chelsea Flower Market in New York. "
            "Write one proactive English tip for the named species. Mention premium greenery or white fillers "
            "and ask whether the planner wants current wholesale availability for those complements. "
            "Two sentences at most. Do not calculate prices or stem counts."
        )
        user = f"Species: {species or 'this species'}\nPlanner note:\n{message}"
    else:
        system = (
            "You are the BloomTrust design consultant for the Chelsea Flower Market in New York. "
            "The planner wants floral inspiration. Reply in English in two or three sentences "
            "about greenery, white fillers, and event design. Do not calculate prices or stem counts."
        )
        user = message or "The planner wants floral inspiration."
    try:
        client = OpenAI(api_key=api_key, timeout=30.0)
        completion = client.chat.completions.create(
            model=_VISION_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        texto = (completion.choices[0].message.content or "").strip()
        if not texto or "low packed runner" in texto.lower():
            return respaldo
        return texto
    except Exception as exc:
        _log.warning("Design consultant request failed: %s", type(exc).__name__)
        return respaldo


_TALLOS_COMPLEMENTO = 20


def _lote_complementario(consejo: dict) -> dict:
    """Price the named complement with the cost engine. The model never sets the dollars."""
    especie = str(consejo.get("species") or "Eucalyptus")
    wholesale = _standard_wholesale(especie)
    standard = BloomTrustCostEngine(especie, wholesale).generate_packages(0.0, _TALLOS_COMPLEMENTO)["Standard"]
    return {
        "species": especie,
        "image_url": str(consejo.get("image_url") or ""),
        "unsplash_page": str(consejo.get("unsplash_page") or "https://unsplash.com"),
        "supplier": "Midtown Stem Supply",
        "tier": "Standard",
        "stems": standard["allocated_stems"],
        "unit_price_usd": standard["unit_price_USD"],
        "subtotal_usd": standard["package_total_USD"],
    }


def _respuesta_creativa(message: str, detected: list[str]) -> dict:
    especie = _display_name(detected[0]) if detected else ""
    consejo = BloomTrustDesignConsultant().generate_styling_advice(message, especie)
    texto = str(consejo.get("advice") or "").strip()
    return {
        "type": "text",
        "intent": "CREATIVE_ADVICE",
        "detected_flowers": detected,
        "care_data": _care_data(detected),
        "text": texto,
        "bot_response": texto,
        "complement": _lote_complementario(consejo),
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


@router.post("/analyze-event")
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
        if intent == "OUT_OF_DOMAIN":
            return {
                "type": "text",
                "intent": "OUT_OF_DOMAIN",
                "text": _RECHAZO_DOMINIO,
                "bot_response": _RECHAZO_DOMINIO,
            }
        if intent == "CREATIVE_ADVICE":
            return await asyncio.to_thread(_respuesta_creativa, texto, detected)
        requirements = {
            "detected_flowers": detected,
            "aesthetic_style": classified.get("aesthetic_style") or "",
            "budget_provided": float(classified.get("max_budget") or 0) > 0,
            "budget_amount": float(classified.get("max_budget") or 0),
            "event_date": "",
        }
        motor_budget = _presupuesto_motor(texto, classified.get("max_budget"))
    else:
        requirements = await asyncio.to_thread(
            extract_floral_requirements_from_image,
            image_bytes,
            texto,
        )
        detected = [str(item).strip().lower() for item in requirements.get("detected_flowers") or [] if str(item).strip()]
        motor_budget = _presupuesto_motor(texto)

    aesthetic_style = str(requirements.get("aesthetic_style") or "")
    recipe = _receta_local(texto)
    if recipe is None and image_bytes is not None:
        recipe = _receta_desde_requisitos(requirements)
    elif recipe is None:
        parsed = await asyncio.to_thread(extract_floral_requirements, texto)
        recipe = _receta_desde_requisitos(parsed)
    if recipe:
        detected = [item["flower_name"] for item in recipe["items"]]
        if recipe.get("aesthetic_style"):
            aesthetic_style = recipe["aesthetic_style"]
    primary = detected[0] if detected else "rose"
    event_date = _event_date(texto, str(requirements.get("event_date") or ""))
    design = _design_type(texto)
    quantity = _quantity(texto)
    month_number = int(event_date[5:7])
    mencionadas = _flores_en_texto(texto)
    if mencionadas and all(_es_consulta_peonia(item) for item in mencionadas):
        detected = ["peony"]
        primary = "peony"
    elif any(_es_consulta_peonia(item) for item in mencionadas) and not any(_es_consulta_peonia(item) for item in detected):
        detected.append("peony")
    consulta = [*detected, *mencionadas]
    if recipe:
        consulta.extend(str(item.get("flower_name") or "") for item in recipe["items"])
    sustitucion = _match_peonia_otono(consulta, month_number)
    if sustitucion:
        reemplazo = str(sustitucion["final_flower"])
        detected = [reemplazo if _es_consulta_peonia(item) else item for item in detected]
        if not detected or _es_consulta_peonia(primary):
            primary = reemplazo
            if reemplazo not in detected:
                detected = [reemplazo, *detected]
        if recipe:
            recipe = {
                **recipe,
                "items": [
                    {**item, "flower_name": reemplazo}
                    if _es_consulta_peonia(str(item.get("flower_name") or ""))
                    else item
                    for item in recipe["items"]
                ],
            }
    recipe_payload = None
    if recipe:
        acumulado = _acumular_costo_receta(recipe["arrangements_quantity"], recipe["items"])
        try:
            packages = BloomTrustCostEngine.price_from_accumulated_total(
                motor_budget,
                acumulado["total_base_price"],
                acumulado["total_stems"],
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        rows = acumulado["items"]
        total_stems = acumulado["total_stems"]
        sizing = {
            "design_type": design,
            "quantity": recipe["arrangements_quantity"],
            "total_stems": total_stems,
            "required_stems": total_stems,
        }
        variety = " + ".join(row["display_name"] for row in rows)
        wholesale = packages["Standard"]["unit_price_USD"]
        recipe_payload = {
            "arrangements_quantity": recipe["arrangements_quantity"],
            "arrangements_quoted": recipe["arrangements_quantity"],
            "items": rows,
            "aesthetic_style": aesthetic_style,
            "total_stems": total_stems,
            "premium_total_usd": packages["Premium"]["package_total_USD"],
            "standard_total_usd": packages["Standard"]["package_total_USD"],
            "opportunity_total_usd": packages["Opportunity"]["package_total_USD"],
        }
    else:
        stated = _stated_stems(required_stems, texto)
        try:
            sizing = BloomTrustVolumeEstimator().estimate_required_stems(design, quantity)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        sizing["total_stems"] = stated
        sizing["required_stems"] = stated
        wholesale = _standard_wholesale(primary)
        try:
            packages = BloomTrustCostEngine(
                target_flower=primary,
                base_wholesale_price=wholesale,
            ).generate_packages(motor_budget, int(sizing["total_stems"]))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        variety = _display_name(primary)
    demand = BloomTrustPredictiveForecaster(event_date).calculate_market_volatility()
    month_name = _MONTH_NAME[month_number]
    surge = int(demand["demand_surge_percentage"])
    design_tip = await asyncio.to_thread(_consultar_diseno, texto, variety, "quote")
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
        "recipe": recipe_payload,
        "bot_response": design_tip,
        "design_tip": design_tip,
        "base_wholesale_price_usd": wholesale,
        "max_budget": motor_budget,
        "packages": packages,
        "financials": _financial_rows(variety, packages),
        "botanical_guide": BloomTrustCareEngine().query_botanical_guide(primary),
        "care_data": _care_data(detected or [primary]),
        "market_feed": BloomTrustTrendsEngine().get_live_market_feed(),
        **(
            {
                "stock_status": sustitucion["reason"],
                "substitution": {
                    "availability": sustitucion["availability"],
                    "original_flower": sustitucion["original_flower"],
                    "final_flower": sustitucion["final_flower"],
                    "score": sustitucion["score"],
                },
            }
            if sustitucion
            else {}
        ),
    }
