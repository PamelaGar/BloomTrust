"""Agente floral de BloomTrust: extracción estructurada y Agente Crítico de temporada.

El modelo (gpt-4o-mini) solo identifica flores, estilo y temporada.
La decisión de cumplimiento estacional es determinista y usa el calendario
botánico local: si una especie cae fuera de la temporada detectada, el flujo
se detiene y la respuesta incluye sustitutos estéticos.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

TEMPORADAS = ("primavera", "verano", "otoño", "invierno")
_TEMPORADA_ALIAS = {
    "otono": "otoño",
    "fall": "otoño",
    "autumn": "otoño",
    "spring": "primavera",
    "summer": "verano",
    "winter": "invierno",
}


@dataclass(frozen=True)
class _Especie:
    nombre: str
    ventanas: frozenset[str]
    sustitutos: tuple[str, ...]
    motivo: str


def _todas() -> frozenset[str]:
    return frozenset(TEMPORADAS)


# Calendario del mercado mayorista de NYC. Las rosas y orquídeas de importación
# están todo el año; cultivos de ventana corta (peonía, lila, dalia) no.
_CATALOGO: dict[str, _Especie] = {
    "peonia": _Especie(
        "peonía",
        frozenset({"primavera"}),
        ("ranunculo", "anemona", "tulipan", "rosa"),
        "Pétalos densos y silueta romántica, habituales en el mercado de NYC.",
    ),
    "rosa": _Especie("rosa", _todas(), (), ""),
    "orquidea": _Especie("orquídea", _todas(), (), ""),
    "ranunculo": _Especie(
        "ranúnculo",
        frozenset({"invierno", "primavera"}),
        ("anemona", "tulipan", "rosa", "peonia"),
        "Capas de pétalos suaves, cercanas al volumen de una peonía.",
    ),
    "anemona": _Especie(
        "anémona",
        frozenset({"invierno", "primavera"}),
        ("ranunculo", "tulipan", "fresia"),
        "Centro contrastado y porte ligero de temporada fría.",
    ),
    "tulipan": _Especie(
        "tulipán",
        frozenset({"invierno", "primavera"}),
        ("ranunculo", "anemona", "narciso"),
        "Línea limpia de primavera, disponible desde el invierno en NYC.",
    ),
    "narciso": _Especie(
        "narciso",
        frozenset({"invierno", "primavera"}),
        ("tulipan", "jacinto", "ranunculo"),
        "Lectura fresca de finales de invierno.",
    ),
    "jacinto": _Especie(
        "jacinto",
        frozenset({"invierno", "primavera"}),
        ("narciso", "tulipan", "fresia"),
        "Espiga perfumada de temporada fría.",
    ),
    "fresia": _Especie(
        "fresia",
        frozenset({"invierno", "primavera"}),
        ("ranunculo", "anemona", "rosa"),
        "Perfume y curvas suaves en la ventana fría.",
    ),
    "lila": _Especie(
        "lila",
        frozenset({"primavera"}),
        ("guisante", "rosa", "ranunculo"),
        "Racimos aireados de primavera.",
    ),
    "guisante": _Especie(
        "guisante de olor",
        frozenset({"primavera"}),
        ("rosa", "ranunculo", "fresia"),
        "Textura delicada y paleta pastel de primavera.",
    ),
    "hortensia": _Especie(
        "hortensia",
        frozenset({"verano", "otoño"}),
        ("dalia", "rosa", "crisantemo"),
        "Masa de pétalos para arreglos amplios de verano y otoño.",
    ),
    "dalia": _Especie(
        "dalia",
        frozenset({"verano", "otoño"}),
        ("crisantemo", "rosa", "zinnia"),
        "Geometría ornamental de verano y otoño.",
    ),
    "zinnia": _Especie(
        "zinnia",
        frozenset({"verano", "otoño"}),
        ("dalia", "girasol", "crisantemo"),
        "Color plano y alegre de cultivo de calor.",
    ),
    "girasol": _Especie(
        "girasol",
        frozenset({"verano", "otoño"}),
        ("dalia", "crisantemo", "zinnia"),
        "Disco y pétalos radiados de temporada cálida.",
    ),
    "crisantemo": _Especie(
        "crisantemo",
        frozenset({"otoño"}),
        ("dalia", "rosa", "hortensia"),
        "Volumen otoñal de pétalo denso.",
    ),
    "lisianthus": _Especie(
        "lisianthus",
        frozenset({"verano"}),
        ("rosa", "dalia", "ranunculo"),
        "Rosa suave de pétalo fino, propia del verano.",
    ),
    "delfinio": _Especie(
        "delfinio",
        frozenset({"verano"}),
        ("rosa", "lisianthus", "hortensia"),
        "Espiga vertical de verano.",
    ),
    "amarilis": _Especie(
        "amarilis",
        frozenset({"invierno"}),
        ("rosa", "tulipan", "orquidea"),
        "Flor grande de invierno, de presencia formal.",
    ),
}

_ALIAS = {
    "peonies": "peonia",
    "peony": "peonia",
    "peonias": "peonia",
    "roses": "rosa",
    "rose": "rosa",
    "rosas": "rosa",
    "orchids": "orquidea",
    "orchid": "orquidea",
    "orquideas": "orquidea",
    "ranunculus": "ranunculo",
    "ranunculos": "ranunculo",
    "anemone": "anemona",
    "anemones": "anemona",
    "anemonas": "anemona",
    "tulip": "tulipan",
    "tulips": "tulipan",
    "tulipanes": "tulipan",
    "daffodil": "narciso",
    "narcisos": "narciso",
    "hyacinth": "jacinto",
    "jacintos": "jacinto",
    "freesia": "fresia",
    "fresias": "fresia",
    "lilac": "lila",
    "lilas": "lila",
    "sweet pea": "guisante",
    "guisantes": "guisante",
    "hydrangea": "hortensia",
    "hortensias": "hortensia",
    "dahlia": "dalia",
    "dahlias": "dalia",
    "dalias": "dalia",
    "zinnias": "zinnia",
    "sunflower": "girasol",
    "sunflowers": "girasol",
    "girasoles": "girasol",
    "chrysanthemum": "crisantemo",
    "mum": "crisantemo",
    "crisantemos": "crisantemo",
    "delphinium": "delfinio",
    "amaryllis": "amarilis",
}


class FlorSolicitada(BaseModel):
    nombre_flor: str = Field(description="Nombre de la especie, en minúsculas.")
    cantidad: int = Field(ge=0, description="Tallos u objetos pedidos. 0 si no se indicó cantidad.")

    @field_validator("nombre_flor")
    @classmethod
    def _en_minusculas(cls, valor: str) -> str:
        return " ".join(valor.strip().lower().split())


class ExtraccionFloral(BaseModel):
    """Esquema que OpenAI debe cumplir de forma estricta."""

    flores: list[FlorSolicitada]
    estilo_estetico: str = Field(
        description="Estilo breve: minimalista, clásico, boho, corporativo, romántico, tropical u otro."
    )
    temporada_detectada: str = Field(
        description="primavera, verano, otoño, invierno, o 'no especificada' si el texto no da señal."
    )


class SustitutoEstetico(BaseModel):
    flor_solicitada: str
    alternativas: list[str]
    motivo: str


class RespuestaCritico(BaseModel):
    alerta: bool
    mensaje: str
    sustitutos: list[SustitutoEstetico]


class ResultadoFloral(BaseModel):
    flores: list[FlorSolicitada]
    estilo_estetico: str
    temporada_detectada: str
    respuesta: RespuestaCritico


def _sin_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def _cargar_env() -> None:
    """Carga .env sin pisar variables ya definidas en el entorno."""
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


def _clave_especie(nombre_flor: str) -> str | None:
    clave = _sin_acentos(nombre_flor)
    if clave in _ALIAS:
        return _ALIAS[clave]
    if clave in _CATALOGO:
        return clave
    if clave.endswith("s") and clave[:-1] in _CATALOGO:
        return clave[:-1]
    if clave.endswith("es") and clave[:-2] in _CATALOGO:
        return clave[:-2]
    return _ALIAS.get(clave)


def _temporada(valor: str) -> str | None:
    clave = _sin_acentos(valor)
    if clave in _TEMPORADA_ALIAS:
        return _TEMPORADA_ALIAS[clave]
    if clave in {_sin_acentos(item) for item in TEMPORADAS}:
        return {"otono": "otoño"}.get(clave, valor.strip().lower())
    return None


def aplicar_agente_critico(extraccion: ExtraccionFloral) -> ResultadoFloral:
    """Detiene el flujo si alguna flor queda fuera de la temporada detectada."""
    temporada = _temporada(extraccion.temporada_detectada)
    temporada_salida = temporada or extraccion.temporada_detectada.strip().lower()
    sustitutos: list[SustitutoEstetico] = []

    if temporada is None:
        mensaje = (
            "No se infirió una temporada concreta. El Agente Crítico no bloquea "
            "el flujo hasta que el florista indique la época del evento."
        )
        return ResultadoFloral(
            flores=extraccion.flores,
            estilo_estetico=extraccion.estilo_estetico.strip(),
            temporada_detectada=temporada_salida,
            respuesta=RespuestaCritico(alerta=False, mensaje=mensaje, sustitutos=[]),
        )

    for flor in extraccion.flores:
        clave = _clave_especie(flor.nombre_flor)
        ficha = _CATALOGO.get(clave) if clave else None
        if ficha is None or temporada in ficha.ventanas:
            continue
        alternativas: list[str] = []
        for candidato in ficha.sustitutos:
            opcion = _CATALOGO[candidato]
            if temporada in opcion.ventanas and opcion.nombre not in alternativas:
                alternativas.append(opcion.nombre)
        sustitutos.append(
            SustitutoEstetico(
                flor_solicitada=flor.nombre_flor,
                alternativas=alternativas,
                motivo=ficha.motivo,
            )
        )

    if not sustitutos:
        mensaje = (
            f"La solicitud cumple la ventana de {temporada}. "
            "El flujo de abastecimiento puede continuar."
        )
        alerta = False
    else:
        detalle = []
        temporada_visible = temporada[:1].upper() + temporada[1:]
        for item in sustitutos:
            alternativa = item.alternativas[0] if item.alternativas else "rosa"
            flor = item.flor_solicitada[:1].upper() + item.flor_solicitada[1:]
            opcion = alternativa[:1].upper() + alternativa[1:]
            detalle.append(
                f"⚠️ {flor} fuera de temporada para {temporada_visible}. Sustituida por {opcion}."
            )
        mensaje = "\n".join(detalle)
        alerta = True

    return ResultadoFloral(
        flores=extraccion.flores,
        estilo_estetico=extraccion.estilo_estetico.strip(),
        temporada_detectada=temporada,
        respuesta=RespuestaCritico(alerta=alerta, mensaje=mensaje, sustitutos=sustitutos),
    )


def analizar_solicitud(mensaje: str) -> ResultadoFloral:
    """Extrae la intención del florista con salida estructurada y aplica el Crítico."""
    texto = mensaje.strip()
    if not texto:
        raise ValueError("El mensaje del florista está vacío.")

    api_key, modelo, temperatura = _configuracion()
    from openai import OpenAI

    cliente = OpenAI(api_key=api_key, timeout=30.0)
    completado = cliente.responses.parse(
        model=modelo,
        temperature=temperatura,
        input=[
            {
                "role": "system",
                "content": (
                    "Eres el extractor floral de BloomTrust, plataforma B2B de "
                    "abastecimiento para eventos en New York City. Del mensaje del "
                    "florista devuelve únicamente el esquema: flores (nombre_flor "
                    "en español y minúsculas, cantidad entera; 0 si no hay número), "
                    "estilo_estetico breve y temporada_detectada (primavera, verano, "
                    "otoño, invierno, o 'no especificada'). No inventes especies que "
                    "no estén en el texto. No calcules precios, tallos, porcentajes ni "
                    "presupuestos. No redactes la respuesta del chat: si una flor está "
                    "fuera de temporada, la única frase permitida es "
                    "'⚠️ {Flor} fuera de temporada para {Mes}. Sustituida por {Alternativa}.' "
                    "Si todas están en temporada, no escribas esa línea."
                ),
            },
            {"role": "user", "content": texto},
        ],
        text_format=ExtraccionFloral,
    )
    extraido = completado.output_parsed
    if extraido is None:
        raise RuntimeError("OpenAI no devolvió la salida estructurada del esquema floral.")
    return aplicar_agente_critico(extraido)
