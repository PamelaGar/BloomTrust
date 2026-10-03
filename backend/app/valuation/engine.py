"""Motor determinista de costos de BloomTrust.

El LLM no interviene. Cada banda compra tallos reales del inventario filtrado,
descuenta la merma y, en Opportunity, consume primero el saldo excedente.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP

import pandas as pd

BANDAS = ("Premium", "Standard", "Opportunity")
_BANDA_INTERNA = {"Premium": "premium", "Standard": "standard", "Opportunity": "opportunity"}
# Índice fijo del calendario social de NYC. No lo estima el modelo.
_PICO_DEMANDA_NYC = {
    1: 12,
    2: 15,
    3: 18,
    4: 26,
    5: 34,
    6: 40,
    7: 16,
    8: 12,
    9: 28,
    10: 33,
    11: 30,
    12: 36,
}


def _decimal(valor: float | int | str | Decimal) -> Decimal:
    return Decimal(str(valor))


def _dinero(valor: Decimal) -> float:
    return float(valor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class BloomTrustCostEngine:
    """Reparte `presupuesto_max` en tres bandas comerciales reproducibles."""

    def __init__(self, target_flower: str | None = None, base_wholesale_price: float | None = None):
        self.target_flower = target_flower or ""
        self.base_price = None if base_wholesale_price is None else _decimal(base_wholesale_price)

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

    def generate_packages(self, max_budget: float, required_stems: int) -> dict:
        """Escala un precio base en Premium, Standard y Opportunity sin intervenir el modelo."""
        if self.base_price is None or self.base_price <= 0:
            raise ValueError("base_wholesale_price debe ser mayor que cero.")
        if required_stems < 0:
            raise ValueError("required_stems no puede ser negativo.")
        presupuesto = _decimal(max_budget).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if presupuesto <= 0:
            raise ValueError("max_budget debe ser mayor que cero.")

        premium_precio = self.base_price * Decimal("1.5")
        standard_precio = self.base_price
        opportunity_precio = self.base_price * Decimal("0.7")
        premium_tallos = _tallos_por_precio(presupuesto, premium_precio)
        standard_tallos = min(_tallos_por_precio(presupuesto, standard_precio), required_stems)
        opportunity_tallos = _tallos_por_precio(presupuesto, opportunity_precio)

        return {
            "Premium": _paquete(
                "Prioritizes maximum aesthetic volume & fresh high-end sourcing grading.",
                premium_precio,
                premium_tallos,
                presupuesto,
            ),
            "Standard": _paquete(
                "Perfect commercial equilibrium between baseline price and design fulfillment.",
                standard_precio,
                standard_tallos,
                presupuesto,
            ),
            "Opportunity": _paquete(
                "Leverages NYC wholesaler surplus & overstock. Maximum savings and high volume.",
                opportunity_precio,
                opportunity_tallos,
                presupuesto,
            ),
        }


def _tallos_por_precio(presupuesto: Decimal, precio: Decimal) -> int:
    if precio <= 0:
        return 0
    return int((presupuesto / precio).to_integral_value(rounding=ROUND_FLOOR))


def _paquete(concepto: str, precio: Decimal, tallos: int, presupuesto: Decimal) -> dict:
    costo = (Decimal(tallos) * precio).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "tier_concept": concepto,
        "unit_price_USD": _dinero(precio),
        "allocated_stems": tallos,
        "package_total_USD": _dinero(costo),
        "remaining_budget_USD": _dinero(presupuesto - costo),
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


def porcentaje_demanda(mes: int) -> int:
    """Porcentaje de alza del mes según el índice fijo de eventos en NYC."""
    return _PICO_DEMANDA_NYC[mes]


def _moneda(valor: float) -> str:
    return f"${valor:,.2f}"


def tabla_markdown(banda: dict, nombres: dict[str, str] | None = None) -> str:
    """Tabla corta de una banda. El subtotal es el costo ya calculado por el motor."""
    nombres = nombres or {}
    filas = [
        "| Variedad | Proveedor | Precio Tallo | Tallos | Subtotal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for linea in banda.get("lineas") or []:
        if not linea.get("tallos_comprados"):
            continue
        especie = str(linea.get("especie") or "")
        variedad = nombres.get(especie, especie[:1].upper() + especie[1:] if especie else "—")
        filas.append(
            "| {variedad} | {proveedor} | {precio} | {tallos} | {subtotal} |".format(
                variedad=variedad,
                proveedor=linea.get("proveedor") or "—",
                precio=_moneda(float(linea.get("precio_tallo") or 0)),
                tallos=int(linea["tallos_comprados"]),
                subtotal=_moneda(float(linea.get("costo") or 0)),
            )
        )
    return "\n".join(filas)


def redactar_markdown(
    bandas: dict | None,
    mes_visible: str,
    porcentaje: int,
    avisos: list[str],
    nombres: dict[str, str] | None = None,
) -> str:
    """Arma la respuesta del bróker: avisos de una línea, alerta de demanda y tres tablas."""
    bloques: list[str] = []
    if avisos:
        bloques.append("\n".join(avisos))
    bloques.append(
        "📈 Alerta de Mercado: Se recomienda reservar tus tallos entre esta semana y la siguiente. "
        f"La demanda aumentará un {porcentaje}% en {mes_visible} debido a picos estacionales de eventos en NYC."
    )
    if bandas:
        for nombre in BANDAS:
            bloques.append(f"**{nombre}**\n\n{tabla_markdown(bandas[nombre], nombres)}")
    return "\n\n".join(bloques)


_JUSTIFICACION_MES = {
    1: "Enero abre el calendario corporativo de Nueva York, con galas de inicio de año por encima de la base.",
    2: "Febrero concentra San Valentín y cenas privadas; la presión mayorista sube sobre la semana del 14.",
    3: "Marzo sostiene galas de primavera y eventos corporativos antes del pico de bodas.",
    4: "Abril entra en la temporada de bodas y fines de semana sociales en la ciudad.",
    5: "Mayo junta fines de semana de boda y temporada alta de hospitalidad.",
    6: "Junio es el pico de bodas del calendario de Nueva York.",
    7: "Julio baja frente a junio y conserva eventos corporativos de verano.",
    8: "Agosto es el mes más quieto del índice, fuera del pico de bodas.",
    9: "Septiembre reactiva galas, moda y cenas corporativas tras el verano.",
    10: "Octubre sostiene la temporada de galas de otoño.",
    11: "Noviembre suma festivos y cenas corporativas de cierre.",
    12: "Diciembre cierra con fiestas y eventos de fin de año.",
}


class BloomTrustPredictiveForecaster:
    """Traduce una fecha al alza publicada del mes. El porcentaje es el índice fijo de NYC."""

    def __init__(self, target_date: str):
        self.target_date = target_date

    def calculate_market_volatility(self) -> dict:
        fecha = datetime.strptime(self.target_date, "%Y-%m-%d")
        porcentaje = porcentaje_demanda(fecha.month)
        return {
            "demand_surge_percentage": porcentaje,
            "market_justification": _JUSTIFICACION_MES[fecha.month],
        }


class BloomTrustVolumeEstimator:
    """Recetas B2B de Nueva York: tallos por arreglo, sin precios."""

    def __init__(self):
        self.design_recipes = {
            "low_centerpiece": {"roses": 15, "hydrangeas": 5, "eucalyptus": 8},
            "high_centerpiece": {"roses": 30, "hydrangeas": 10, "eucalyptus": 15},
            "floral_arch": {"roses": 120, "hydrangeas": 40, "eucalyptus": 60},
            "bridal_bouquet": {"roses": 24, "hydrangeas": 2, "eucalyptus": 5},
        }
        self._alias = {
            "low_centerpiece": "low_centerpiece",
            "centro de mesa bajo": "low_centerpiece",
            "centro-bajo": "low_centerpiece",
            "high_centerpiece": "high_centerpiece",
            "centro de mesa alto": "high_centerpiece",
            "centro-alto": "high_centerpiece",
            "floral_arch": "floral_arch",
            "arco floral": "floral_arch",
            "arco": "floral_arch",
            "bridal_bouquet": "bridal_bouquet",
            "bouquet de novia": "bridal_bouquet",
            "bouquet": "bridal_bouquet",
        }

    def estimate_required_stems(self, design_type: str, arrangement_quantity: int) -> dict:
        clean_type = self._alias.get(design_type.lower().strip())
        if clean_type not in self.design_recipes:
            raise ValueError(f"Design type '{design_type}' not registered in B2B standards.")
        if arrangement_quantity < 0:
            raise ValueError("arrangement_quantity no puede ser negativo.")
        base_recipe = self.design_recipes[clean_type]
        calculated_stems = {flower: count * arrangement_quantity for flower, count in base_recipe.items()}
        return {
            "design_type": clean_type,
            "quantity": arrangement_quantity,
            "stem_breakdown": calculated_stems,
            "total_stems": sum(calculated_stems.values()),
        }
