"""Deterministic BloomTrust engines ported from the Colab notebook.

The language model never prices a stem. Care, Manhattan trends, and the
three commercial packages live here so the API can answer from the same
dictionaries that were verified in the notebook.
"""

from __future__ import annotations

import math
from datetime import datetime

# Fixed New York social calendar. The model does not estimate these percents.
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
_CARE_ALIASES = {
    "rose": "rose",
    "roses": "rose",
    "rosa": "rose",
    "garden rose": "garden rose",
    "garden roses": "garden rose",
    "hydrangea": "hydrangea",
    "hydrangeas": "hydrangea",
    "hortensia": "hydrangea",
    "orchid": "orchid",
    "orchids": "orchid",
    "orquidea": "orchid",
    "lisianthus": "lisianthus",
    "carnation": "carnation",
    "carnations": "carnation",
    "clavel": "carnation",
    "baby_s_breath": "baby_s_breath",
    "baby's breath": "baby_s_breath",
    "babys breath": "baby_s_breath",
    "gypsophila": "baby_s_breath",
    "eucalyptus": "eucalyptus",
    "eucalipto": "eucalyptus",
}


def porcentaje_demanda(mes: int) -> int:
    """Published demand lift for a month on the fixed NYC event index."""
    return _PICO_DEMANDA_NYC[mes]


class BloomTrustCareEngine:
    def __init__(self):
        # Global Botanical Knowledge Base for B2B Post-Sourcing Conditioning
        self.care_database = {
            "rose": {
                "scientific_name": "Rosa rubiginosa",
                "hydration": "Clean bucket with high-acidity water and standard floral commercial food nutrients.",
                "temperature": "34°F - 38°F (1°C - 3°C) for stabilization.",
                "alert": "Remove lower foliage entirely before immersion to prevent extreme microbial shrink.",
            },
            "garden rose": {
                "scientific_name": "Rosa centifolia",
                "hydration": "Warm water initial shock, then transfer to cold hydration with high glucose sugars.",
                "temperature": "36°F - 40°F (2°C - 4°C) to slow petal opening.",
                "alert": "Very delicate petals; do not mist directly or physical bruising mermas will trigger.",
            },
            "hydrangea": {
                "scientific_name": "Hydrangea macrophylla",
                "hydration": "Alum powder stem dip treatment. Submerge heavy dense blooms completely for 30 mins if flagging.",
                "temperature": "38°F - 42°F (3°C - 5°C) inside the workshop cold-room.",
                "alert": "High transpiration rate. Stems require deep vertical cross-cuts for massive water intake.",
            },
            "orchid": {
                "scientific_name": "Phalaenopsis amabilis",
                "hydration": "Hydrate in shallow, room-temperature fresh water. Avoid heavy industrial processing solutions.",
                "temperature": "50°F - 55°F (10°C - 13°C). CRITICAL: Do NOT store in standard low-temperature coolers.",
                "alert": "Extreme cold sensitivity will cause translucent petal rot. Keep high ambient humidity.",
            },
            "lisianthus": {
                "scientific_name": "Eustoma russellianum",
                "hydration": "Standard fresh conditioning with antibacterial agents. Requires frequent water rotations.",
                "temperature": "36°F - 38°F (2°C - 3°C) for maximum node stiffness.",
                "alert": "Stems are structurally brittle at junctions. Handle with high geometric care during unpacking.",
            },
            "carnation": {
                "scientific_name": "Dianthus caryophyllus",
                "hydration": "Highly resilient. Standard wholesale water conditioning with minimal nutrient overhead.",
                "temperature": "34°F - 36°F (1°C - 2°C) for prolonged storage windows.",
                "alert": "Keep strictly away from ethylene gas emitters like ripening fruits or logistics exhaust.",
            },
            "baby_s_breath": {
                "scientific_name": "Gypsophila paniculata",
                "hydration": "Recut stems under water and condition in a clean bucket with commercial flower food.",
                "temperature": "34°F - 36°F (1°C - 2°C). Keep the clouds dry; do not mist the florets.",
                "alert": "Ethylene sensitive. Store away from ripening fruit and keep the bunches loose so the stems do not mold.",
            },
            "eucalyptus": {
                "scientific_name": "Eucalyptus cinerea",
                "hydration": "Split woody stem ends and condition in warm water, then move the bucket to the cooler.",
                "temperature": "36°F - 38°F (2°C - 3°C) with the foliage kept out of the water line.",
                "alert": "Leaves left underwater slime the bucket. Strip submerged foliage before the cold room.",
            },
        }

    def query_botanical_guide(self, flower_query: str) -> dict:
        """Acts as the underlying search engine for the Storage UI catalog layer."""
        clean_query = (flower_query or "").lower().strip()
        key = _CARE_ALIASES.get(clean_query, clean_query)
        return self.care_database.get(key, {
            "scientific_name": "Unknown Especies",
            "hydration": "Standard hydration with fresh water treatment.",
            "temperature": "36°F - 40°F baseline range.",
            "alert": "Handle with standard professional florist protocols.",
        })


class BloomTrustTrendsEngine:
    def __init__(self):
        # Live NYC Hospitality & Runway Design Feed Data Structure
        self.current_trends = {
            "headline": "LIVE MANHATTAN TRENDS: MONOCHROMATIC DENSITY & STRUCTURED ANTHURIMS DOMINATING WALL STREET GALAS",
            "featured_items": [
                {
                    "style": "Classic Luxury Runner",
                    "flower": "garden rose",
                    "visual_reference": "https://unsplash.com",
                    "offer_justification": "Chelsea Market Liquidation: 30% OFF overstock due to immediate greenhouse surplus imports.",
                },
                {
                    "style": "Minimalist Exotic Clean",
                    "flower": "orchid",
                    "visual_reference": "https://unsplash.com",
                    "offer_justification": "Midtown Surplus Event: Wholesale overstock lots available for immediate local delivery.",
                },
            ],
        }

    def get_live_market_feed(self) -> dict:
        return self.current_trends


class BloomTrustCostEngine:
    def __init__(self, target_flower: str, base_wholesale_price: float):
        self.target_flower = target_flower
        self.base_price = base_wholesale_price

    def generate_packages(self, max_budget: float | None, required_stems: int) -> dict:
        """Price three tiers. No cap bills the full stem count; a budget caps each band with min()."""
        if self.base_price is None or self.base_price <= 0:
            raise ValueError("base_wholesale_price debe ser mayor que cero.")
        if required_stems < 0:
            raise ValueError("required_stems no puede ser negativo.")

        stems = int(required_stems)
        premium_stem_price = self.base_price * 1.5
        standard_stem_price = self.base_price * 1.0
        opportunity_stem_price = self.base_price * 0.7
        open_quote = max_budget is None or float(max_budget) <= 0

        if open_quote:
            premium_final_stems = stems
            standard_final_stems = stems
            opportunity_final_stems = stems
            budget_for_remainder = 0.0
        else:
            budget_for_remainder = float(max_budget)
            premium_final_stems = min(math.floor(budget_for_remainder / premium_stem_price), stems)
            standard_final_stems = min(math.floor(budget_for_remainder / standard_stem_price), stems)
            opportunity_final_stems = min(math.floor(budget_for_remainder / opportunity_stem_price), stems)

        premium_total_cost = premium_final_stems * premium_stem_price
        standard_total_cost = standard_final_stems * standard_stem_price
        opportunity_total_cost = opportunity_final_stems * opportunity_stem_price

        def remaining(cost: float) -> float:
            if open_quote:
                return 0.0
            return round(budget_for_remainder - cost, 2)

        return {
            "Premium": {
                "tier_concept": "Prioritizes maximum aesthetic volume & superior grading sorting.",
                "unit_price_USD": round(premium_stem_price, 2),
                "allocated_stems": premium_final_stems,
                "package_total_USD": round(premium_total_cost, 2),
                "remaining_budget_USD": remaining(premium_total_cost),
            },
            "Standard": {
                "tier_concept": "Perfect commercial equilibrium between price and design fulfillment.",
                "unit_price_USD": round(standard_stem_price, 2),
                "allocated_stems": standard_final_stems,
                "package_total_USD": round(standard_total_cost, 2),
                "remaining_budget_USD": remaining(standard_total_cost),
            },
            "Opportunity": {
                "tier_concept": "Leverages wholesaler surplus. Maximum savings on immediate overstock liquidation.",
                "unit_price_USD": round(opportunity_stem_price, 2),
                "allocated_stems": opportunity_final_stems,
                "package_total_USD": round(opportunity_total_cost, 2),
                "remaining_budget_USD": remaining(opportunity_total_cost),
            },
        }

    @classmethod
    def price_from_accumulated_total(
        cls,
        max_budget: float | None,
        total_base_price: float,
        total_stems: int,
    ) -> dict:
        """Turn one summed recipe cost into Premium, Standard and Opportunity.

        total_base_price already includes every species. Standard uses it as-is.
        Premium is 1.5 times that sum and Opportunity is 0.7 times that sum.
        """
        base = float(total_base_price)
        stems = int(total_stems)
        if base <= 0:
            raise ValueError("total_base_price debe ser mayor que cero.")
        if stems <= 0:
            raise ValueError("total_stems debe ser mayor que cero.")
        open_quote = max_budget is None or float(max_budget) <= 0
        budget_for_remainder = 0.0 if open_quote else float(max_budget)
        concepts = {
            "Premium": "Prioritizes maximum aesthetic volume & superior grading sorting.",
            "Standard": "Perfect commercial equilibrium between price and design fulfillment.",
            "Opportunity": "Leverages wholesaler surplus. Maximum savings on immediate overstock liquidation.",
        }
        packages = {}
        for name, multiplier in (("Premium", 1.5), ("Standard", 1.0), ("Opportunity", 0.7)):
            full_cost = base * multiplier
            if open_quote or full_cost <= budget_for_remainder:
                allocated = stems
                cost = full_cost
            else:
                allocated = min(stems, math.floor(stems * budget_for_remainder / full_cost))
                cost = full_cost * allocated / stems if stems else 0.0
            unit = round(cost / allocated, 2) if allocated else round((base / stems) * multiplier, 2)
            remaining = 0.0 if open_quote else round(budget_for_remainder - cost, 2)
            packages[name] = {
                "tier_concept": concepts[name],
                "unit_price_USD": unit,
                "allocated_stems": allocated,
                "package_total_USD": round(cost, 2),
                "remaining_budget_USD": remaining,
            }
        return packages

    @classmethod
    def quote_recipe(
        cls,
        max_budget: float | None,
        arrangements_quantity: int,
        items: list[dict],
        unit_prices: dict[str, float],
    ) -> dict:
        """Sum a mixed recipe. Each line is stems per arrangement times the arrangement count.

        unit_prices are the Standard wholesale figures already looked up. This method
        only multiplies them by 1.5, 1.0, and 0.7. It does not invent a price.
        """
        arrangements = int(arrangements_quantity)
        if arrangements <= 0:
            raise ValueError("arrangements_quantity debe ser mayor que cero.")
        total_base_price = 0.0
        total_stems = 0
        lines = []
        for item in list(items):
            name = str(item.get("flower_name") or "").strip().lower()
            per = int(item.get("stems_per_arrangement") or 0)
            if not name or per <= 0:
                continue
            if name not in unit_prices:
                raise ValueError(f"No hay precio mayorista para {name}.")
            item_base_price = float(unit_prices[name])
            if item_base_price <= 0:
                raise ValueError(f"El precio de {name} debe ser mayor que cero.")
            item_total_stems = per * arrangements
            total_base_price += item_base_price * item_total_stems
            total_stems += item_total_stems
            lines.append({
                "flower_name": name,
                "stems_per_arrangement": per,
                "base": item_base_price,
                "total_stems": item_total_stems,
            })
        if not lines or total_base_price <= 0:
            raise ValueError("La receta no tiene tallos.")

        open_quote = max_budget is None or float(max_budget) <= 0
        multipliers = (("Premium", 1.5), ("Standard", 1.0), ("Opportunity", 0.7))
        concepts = {
            "Premium": "Prioritizes maximum aesthetic volume & superior grading sorting.",
            "Standard": "Perfect commercial equilibrium between price and design fulfillment.",
            "Opportunity": "Leverages wholesaler surplus. Maximum savings on immediate overstock liquidation.",
        }

        def band(name: str, multiplier: float) -> dict:
            arrangement_cost = sum(item["stems_per_arrangement"] * item["base"] * multiplier for item in lines)
            if open_quote:
                count = arrangements
                budget_for_remainder = 0.0
            else:
                budget_for_remainder = float(max_budget)
                affordable = math.floor(budget_for_remainder / arrangement_cost) if arrangement_cost > 0 else 0
                count = min(max(affordable, 0), arrangements)
            detail = []
            total_stems = 0
            total_cost = 0.0
            for item in lines:
                stems = item["stems_per_arrangement"] * count
                subtotal = stems * item["base"] * multiplier
                total_stems += stems
                total_cost += subtotal
                detail.append({
                    "flower_name": item["flower_name"],
                    "stems_per_arrangement": item["stems_per_arrangement"],
                    "total_stems": stems,
                    "unit_price_usd": round(item["base"] * multiplier, 2),
                    "subtotal_usd": round(subtotal, 2),
                })
            blended = round(total_cost / total_stems, 2) if total_stems else 0.0
            remaining = 0.0 if open_quote else round(budget_for_remainder - total_cost, 2)
            return {
                "tier_concept": concepts[name],
                "unit_price_USD": blended,
                "allocated_stems": total_stems,
                "package_total_USD": round(total_cost, 2),
                "remaining_budget_USD": remaining,
                "arrangements_quoted": count,
                "lines": detail,
            }

        packages = {name: band(name, multiplier) for name, multiplier in multipliers}
        return {"packages": packages, "arrangements_quantity": arrangements}


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


class BloomTrustDesignConsultant:
    """Names one complementary botanical. Prices stay outside this class."""

    _CATALOG = {
        "italian ruscus": {
            "species": "Italian Ruscus",
            "image_url": "https://images.unsplash.com/photo-1632232812783-c774e55bbbe9?auto=format&fit=crop&w=1400&q=80",
            "unsplash_page": "https://unsplash.com/photos/a-close-up-of-a-bush-with-green-leaves-1L0kxXvXyj4",
        },
        "eucalyptus": {
            "species": "Eucalyptus",
            "image_url": "https://images.unsplash.com/photo-1744744041774-4485eb2492ec?auto=format&fit=crop&w=1400&q=80",
            "unsplash_page": "https://unsplash.com/photos/eucalyptus-leaves-curve-gracefully-against-a-cream-background-p3P4d6jSNUc",
        },
        "lisianthus": {
            "species": "Lisianthus",
            "image_url": "https://images.unsplash.com/photo-1617630970477-535b975bec53?auto=format&fit=crop&w=1400&q=80",
            "unsplash_page": "https://unsplash.com/photos/pink-and-white-roses-in-bloom-during-daytime-vJxeOtFyBPo",
        },
    }

    def catalog_entry(self, species: str) -> dict:
        folded = (species or "").lower().strip()
        for key, entry in self._CATALOG.items():
            if key in folded or folded in key:
                return dict(entry)
        return dict(self._CATALOG["eucalyptus"])

    def _respaldo(self, hero_species: str) -> dict:
        folded = (hero_species or "").lower()
        if "ruscus" in folded:
            chosen = "Eucalyptus"
        elif "eucalyptus" in folded:
            chosen = "Lisianthus"
        elif "lisianthus" in folded or "eustoma" in folded:
            chosen = "Italian Ruscus"
        else:
            chosen = "Eucalyptus"
        entry = self.catalog_entry(chosen)
        entry["advice"] = (
            f"{entry['species']} is the complementary lot I would place beside the hero bloom. "
            "Its texture keeps a New York table from reading flat, and the Chelsea bench can "
            "add it as greenery or a white-adjacent filler without changing the main flower."
        )
        return entry

    def generate_styling_advice(self, message: str, hero_species: str = "") -> dict:
        """Ask gpt-4o-mini for one catalog species, then attach that plant's Unsplash photo."""
        respaldo = self._respaldo(hero_species)
        try:
            import os

            from openai import OpenAI

            from app.services.agents.multimodal_agent import _VISION_MODEL, _cargar_env

            _cargar_env()
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        except OSError:
            return respaldo
        if not api_key:
            return respaldo
        system = (
            "You are the BloomTrust design consultant for the Chelsea Flower Market in New York. "
            "Recommend exactly one complementary greenery or filler. "
            "Return JSON with advice and species. "
            "species must be exactly one of: Italian Ruscus, Eucalyptus, Lisianthus. "
            "Italian Ruscus maps to its Unsplash botanical photograph, "
            "Eucalyptus maps to its Unsplash botanical photograph, "
            "and Lisianthus maps to its Unsplash botanical photograph. "
            "Do not invent another species and do not invent an image URL. "
            "advice is two or three English sentences on how that plant supports the event design. "
            "Do not calculate prices or stem counts. "
            "If the planner already named the hero flower, choose a different complement."
        )
        user = f"Hero flower: {hero_species or 'unspecified'}\nPlanner note:\n{message or 'The planner wants floral inspiration.'}"
        try:
            client = OpenAI(api_key=api_key, timeout=30.0)
            completion = client.chat.completions.create(
                model=_VISION_MODEL,
                temperature=0.4,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "styling_advice",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "advice": {"type": "string"},
                                "species": {
                                    "type": "string",
                                    "enum": ["Italian Ruscus", "Eucalyptus", "Lisianthus"],
                                },
                            },
                            "required": ["advice", "species"],
                        },
                    },
                },
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            import json

            crudo = json.loads(completion.choices[0].message.content or "{}")
        except Exception:
            return respaldo
        advice = str(crudo.get("advice") or "").strip()
        if not advice or "low packed runner" in advice.lower():
            return respaldo
        entry = self.catalog_entry(str(crudo.get("species") or ""))
        entry["advice"] = advice
        return entry
