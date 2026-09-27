"""Motor determinista de costos de BloomTrust.

El LLM no interviene. Cada banda compra tallos reales del inventario filtrado,
descuenta la merma y, en Opportunity, consume primero el saldo excedente.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

BANDAS = ("Premium", "Standard", "Opportunity")
_BANDA_INTERNA = {"Premium": "premium", "Standard": "standard", "Opportunity": "opportunity"}


def _decimal(valor: float | int | str | Decimal) -> Decimal:
    return Decimal(str(valor))


def _dinero(valor: Decimal) -> float:
    return float(valor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class BloomTrustCostEngine:
    """Reparte `presupuesto_max` en tres bandas comerciales reproducibles."""

    def calcular(self, inventario: pd.DataFrame, presupuesto_max: float) -> dict[str, dict]:
        presupuesto = _decimal(presupuesto_max).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if presupuesto <= 0:
            raise ValueError("presupuesto_max debe ser mayor que cero.")

        especies = sorted(set(inventario["especie"])) if not inventario.empty else []
        asignaciones = _repartir(presupuesto, len(especies))
        cupo = dict(zip(especies, asignaciones, strict=True))

        return {
            nombre: self._banda(nombre, inventario, cupo, presupuesto)
            for nombre in BANDAS
        }

    def _banda(
        self,
        nombre: str,
        inventario: pd.DataFrame,
        cupo: dict[str, Decimal],
        presupuesto: Decimal,
    ) -> dict:
        clave = _BANDA_INTERNA[nombre]
        lineas: list[dict] = []
        costo_total = Decimal("0")
        tallos_comprados = 0
        tallos_utiles = 0
        merma_tallos = 0
        saldo_usado = 0

        for especie in cupo:
            filas = inventario.loc[inventario["especie"] == especie]
            oferta = _elegir_oferta(filas, clave)
            linea, costo = _cotizar_linea(especie, oferta, cupo[especie], clave)
            lineas.append(linea)
            costo_total += costo
            tallos_comprados += linea["tallos_comprados"]
            tallos_utiles += linea["tallos_utiles"]
            merma_tallos += linea["merma_tallos"]
            saldo_usado += linea["saldo_excedente_usado"]

        restante = presupuesto - costo_total
        return {
            "costo_total": _dinero(costo_total),
            "presupuesto_max": _dinero(presupuesto),
            "presupuesto_restante": _dinero(restante),
            "tallos_comprados": tallos_comprados,
            "tallos_utiles": tallos_utiles,
            "merma_tallos": merma_tallos,
            "saldo_excedente_usado": saldo_usado,
            "lineas": lineas,
        }


def _repartir(presupuesto: Decimal, partes: int) -> list[Decimal]:
    if partes <= 0:
        return []
    centavos = int((presupuesto * 100).to_integral_value(rounding=ROUND_HALF_UP))
    base, resto = divmod(centavos, partes)
    return [Decimal(base + (1 if indice < resto else 0)) / Decimal(100) for indice in range(partes)]


def _elegir_oferta(filas: pd.DataFrame, banda: str) -> pd.Series | None:
    if filas.empty or "banda" not in filas.columns:
        return None
    candidatas = filas.loc[filas["banda"] == banda]
    if candidatas.empty:
        return None
    if banda == "premium":
        return candidatas.sort_values(["precio_tallo", "proveedor"], ascending=[False, True]).iloc[0]
    if banda == "opportunity":
        con_saldo = candidatas.loc[candidatas["saldo_excedente"] > 0]
        pool = con_saldo if not con_saldo.empty else candidatas
        return pool.sort_values(["precio_tallo", "proveedor"], ascending=[True, True]).iloc[0]
    ordenadas = candidatas.sort_values(["precio_tallo", "proveedor"], ascending=[True, True])
    return ordenadas.iloc[(len(ordenadas) - 1) // 2]


def _cotizar_linea(
    especie: str,
    oferta: pd.Series | None,
    asignacion: Decimal,
    banda: str,
) -> tuple[dict, Decimal]:
    if oferta is None:
        return _linea_vacia(especie, asignacion), Decimal("0")

    precio = _decimal(oferta["precio_tallo"])
    merma = _decimal(oferta["merma"])
    stock = int(oferta["stock"])
    saldo = int(oferta["saldo_excedente"])
    if precio <= 0 or stock <= 0:
        return _linea_vacia(especie, asignacion, str(oferta["proveedor"]), precio), Decimal("0")

    comprados = int(asignacion // precio)
    comprados = min(comprados, stock)
    if banda == "opportunity":
        comprados = min(comprados, saldo)

    merma_tallos = int((Decimal(comprados) * merma).to_integral_value(rounding=ROUND_HALF_UP))
    if comprados > 0 and merma < 1 and merma_tallos >= comprados:
        merma_tallos = comprados - 1
    utiles = comprados - merma_tallos
    costo = (Decimal(comprados) * precio).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    linea = {
        "especie": especie,
        "proveedor": str(oferta["proveedor"]),
        "precio_tallo": _dinero(precio),
        "oferta_disponible": True,
        "tallos_comprados": comprados,
        "tallos_utiles": utiles,
        "merma_tallos": merma_tallos,
        "saldo_excedente_usado": comprados if banda == "opportunity" else 0,
        "costo": _dinero(costo),
        "presupuesto_asignado": _dinero(asignacion),
    }
    return linea, costo


def _linea_vacia(
    especie: str,
    asignacion: Decimal,
    proveedor: str = "",
    precio: Decimal = Decimal("0"),
) -> dict:
    return {
        "especie": especie,
        "proveedor": proveedor,
        "precio_tallo": _dinero(precio),
        "oferta_disponible": False,
        "tallos_comprados": 0,
        "tallos_utiles": 0,
        "merma_tallos": 0,
        "saldo_excedente_usado": 0,
        "costo": 0.0,
        "presupuesto_asignado": _dinero(asignacion),
    }
