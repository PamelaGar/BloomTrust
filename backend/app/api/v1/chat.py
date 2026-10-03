"""Adapta las respuestas del chat al idioma del mensaje, sin calificar calidad."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

_PROMPT = (
    "Adaptas el chat de BloomTrust al idioma del mensaje del usuario, sin alargarlo. "
    "La respuesta del bróker ya viene cerrada: líneas ⚠️ de una sola frase, una línea 📈 de alerta de mercado "
    "y tres tablas Markdown (Premium, Standard, Opportunity) con las columnas "
    "Variedad, Proveedor, Precio Tallo, Tallos y Subtotal. "
    "Conserva ese orden, los emojis, cada número, cada porcentaje, cada importe y los encabezados de tabla. "
    "No agregues introducciones, cierres ni párrafos. No calcules precios ni porcentajes. "
    "Si el mensaje está en español, deja el texto en español. "
    "Si está en otro idioma, traduce solo las palabras y deja intacta la estructura Markdown. "
    "Nunca uses merma, waste, shrinkage, defect, baja calidad, desperdicio ni deterioro."
)


class PedidoAdaptacion(BaseModel):
    texto_usuario: str = Field(min_length=1, max_length=4000)
    parrafos: list[str] = Field(min_length=1, max_length=12)


class TextoAdaptado(BaseModel):
    parrafos: list[str]


def _adaptar(texto_usuario: str, parrafos: list[str]) -> list[str]:
    from openai import OpenAI

    cliente = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0)
    modelo = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    completado = cliente.responses.parse(
        model=modelo,
        temperature=0.2,
        input=[
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": (
                    f"User message:\n{texto_usuario}\n\n"
                    "Paragraphs:\n" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parrafos))
                ),
            },
        ],
        text_format=TextoAdaptado,
    )
    adaptado = completado.output_parsed
    if adaptado is None or len(adaptado.parrafos) != len(parrafos):
        return parrafos
    return adaptado.parrafos


@router.post("/adapt")
async def adapt(pedido: PedidoAdaptacion) -> dict:
    """Devuelve los párrafos en el idioma del usuario. Si no hay clave, conserva el inglés."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"parrafos": pedido.parrafos}
    try:
        parrafos = await asyncio.to_thread(_adaptar, pedido.texto_usuario.strip(), pedido.parrafos)
    except Exception:
        parrafos = pedido.parrafos
    return {"parrafos": parrafos}
