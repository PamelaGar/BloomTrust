"""Inventario mayorista de NYC usado para validar stock estacional.

Cada fila es una oferta de proveedor. `meses` indica la ventana en la que
esa especie está en temporada; el filtro del endpoint no cotiza fuera de ella.
"""

from __future__ import annotations

import pandas as pd

# especie, meses (1-12), precios premium/standard/opportunity, merma, saldo excedente
_CATALOGO = (
    ("rosa", tuple(range(1, 13)), 4.50, 2.25, 1.10, 0.05, 0.10, 0.15, 180),
    ("orquidea", tuple(range(1, 13)), 12.00, 7.50, 4.00, 0.04, 0.08, 0.12, 60),
    ("peonia", (3, 4, 5), 9.00, 6.00, 3.50, 0.08, 0.12, 0.18, 40),
    ("ranunculo", (12, 1, 2, 3, 4, 5), 3.80, 2.10, 1.05, 0.06, 0.10, 0.16, 90),
    ("tulipan", (12, 1, 2, 3, 4, 5), 2.40, 1.35, 0.70, 0.05, 0.09, 0.14, 120),
    ("anemona", (12, 1, 2, 3, 4, 5), 3.20, 1.90, 0.95, 0.06, 0.11, 0.15, 70),
    ("dalia", (6, 7, 8, 9, 10), 4.10, 2.40, 1.20, 0.07, 0.11, 0.16, 50),
    ("hortensia", (6, 7, 8, 9, 10, 11), 5.50, 3.25, 1.60, 0.08, 0.12, 0.18, 45),
    ("girasol", (6, 7, 8, 9, 10), 2.80, 1.60, 0.80, 0.05, 0.09, 0.13, 80),
    ("crisantemo", (9, 10, 11), 2.20, 1.25, 0.65, 0.06, 0.10, 0.14, 100),
)

_PROVEEDOR = {
    "premium": "Chelsea Cut Flowers",
    "standard": "Midtown Stem Supply",
    "opportunity": "Hunts Point Surplus",
}


def _filas() -> list[dict]:
    filas: list[dict] = []
    for especie, meses, precio_p, precio_s, precio_o, merma_p, merma_s, merma_o, saldo in _CATALOGO:
        ofertas = (
            ("premium", precio_p, merma_p, 80, 0),
            ("standard", precio_s, merma_s, 200, 0),
            ("opportunity", precio_o, merma_o, saldo, saldo),
        )
        for banda, precio, merma, stock, saldo_banda in ofertas:
            filas.append(
                {
                    "especie": especie,
                    "proveedor": _PROVEEDOR[banda],
                    "banda": banda,
                    "precio_tallo": precio,
                    "stock": stock,
                    "merma": merma,
                    "saldo_excedente": saldo_banda,
                    "meses": meses,
                }
            )
    return filas


df_inventario = pd.DataFrame(_filas())


def filtrar_inventario(
    flores: list[str],
    mes: int,
    inventario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deja solo las ofertas cuya especie fue detectada y está en temporada ese mes."""
    origen = df_inventario if inventario is None else inventario
    if origen.empty or not flores:
        return origen.iloc[0:0].copy()
    especies = set(flores)
    mascara = origen["especie"].isin(especies) & origen["meses"].map(lambda meses: mes in meses)
    return origen.loc[mascara].reset_index(drop=True)
