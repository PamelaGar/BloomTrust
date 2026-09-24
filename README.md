# 🌹 BloomTrust — B2B Floral Sourcing Platform (NYC)
Plataforma de optimización de presupuestos y análisis de abastecimiento floral para el mercado B2B de eventos en New York City. Responde dos preguntas críticas para cualquier event planner o florista profesional:

1. **¿Qué hay disponible y cómo se ajusta a mi diseño?** — Identificación de especies, evaluación de estacionalidad, paletas estéticas y frescura, vía asistente conversacional multimodal con LLM (OpenAI) + Tools + Agentes.
2. **¿Cuánto cuesta realmente armar el evento?** — Análisis determinista de costos optimizado en tres bandas de presupuesto (Premium, Standard y Opportunity) evaluando el stock real de proveedores de NYC frente a inventarios excedentes.

⚠️ **BloomTrust es una herramienta de optimización logística y comercial de abastecimiento.** No asume directamente el transporte ni almacenamiento físico de las flores. Todo endpoint que devuelve una cotización o narrativa de costos va envuelto en un Envelope de cumplimiento de stock estacional.

---

## 🏗️ Arquitectura del Repositorio
Monorepo compuesto por un **Backend** (FastAPI + Uvicorn) y un **Frontend** que orquesta tres fuentes de datos: Catálogos de Distribuidores de NYC (datos de stock/precios), el SDK de OpenAI (Planificador, Crítico y Narrador del chat floral) y el motor determinista de cálculo de costos propio.

```text
📁 Tu Repositorio de GitHub (BloomTrust)
├── 📁 backend/               <-- FastAPI (Lógica, algoritmos de mermas y OpenAI)
│   ├── 📁 app/
│   │   ├── 📁 api/           <-- Endpoints de cotización, chat y proveedores
│   │   ├── 📁 services/      <-- Agentes de OpenAI (Planner, Critic) y conectores de inventario
│   │   └── 📁 valuation/     <-- Motor determinista de costos (Premium/Standard/Opportunity)
│   └── 📁 data/
│       └── bloomtrust.db     <-- Base de datos SQLite (Caché de catálogos y mermas)
├── 📁 frontend/              <-- Interfaz de Usuario para los Floristas Profesionales
└── notebook_eda.ipynb        <-- Cuaderno de Google Colab con el Análisis Exploratorio (EDA)
```

---

## 🔌 Componentes del Backend — FastAPI (`backend/`)

| Componente | Ruta | Responsabilidad |
| :--- | :--- | :--- |
| **API Routers** | `app/api/v1/` | Gestión de endpoints para: `auth`, `providers`, `sourcing`, `chat`, y `admin`. |
| **Sourcing Provider**| `app/services/market/` | Conector REST con el mercado mayorista (ej. Chelsea Flower Market). Manejo de errores, caché local de 2 niveles en SQLite y limitador de cuota. |
| **Floral Agents** | `app/services/agents/` | **Planner, Critic y Narrator:** Agentes de OpenAI (`gpt-4o-mini`). Extraen intenciones del florista, identifican flores en imágenes de Pinterest y estructuran la narrativa en Markdown. |
| **Cost Engine** | `app/valuation/` | **Motor Determinista:** Algoritmo en Python puro que calcula costos exactos por tallo. El LLM nunca calcula dinero; el motor distribuye el presupuesto en las bandas *Premium, Standard y Opportunity*. |

### 🎯 Decisiones Clave de Diseño (Alineadas a la Rúbrica)
* **D1 — El LLM nunca calcula números:** Las matemáticas de costos y disponibilidad se ejecutan de forma determinista en `app/valuation/` usando Python para garantizar reproducibilidad exacta y cero alucinaciones de presupuesto.
* **D2 — Cumplimiento Estricto de Temporada:** Si el florista solicita una flor fuera de su ventana estacional (ej. Peonías en invierno), el Agente Crítico detiene el flujo y propone sustitutos estéticos basándose en la base de datos local.
* **D3 — Enfoque Trilateral de Escenarios:** Las respuestas de cotización nunca muestran un precio único flotante; se envuelven obligatoriamente en bandas de decisión comercial: *Premium* (Máxima calidad), *Standard* (Equilibrio) y *Opportunity* (Aprovechamiento de excedentes/mermas).

---

## 🗄️ Datos — SQLite (`backend/data/bloomtrust.db`)
Única base de datos local con soporte WAL. Almacena de forma eficiente: usuarios (floristas/proveedores), resumen de catálogos botánicos, respuestas crudas de stock de los distribuidores (caché para no saturar servidores), históricos de precios de tallos en NYC, y telemetría de ejecuciones del chat.

---

## ⚙️ Configuración de Entorno (`.env`)

```ini
ENVIRONMENT=development
DEBUG=True
DATABASE_URL=sqlite+aiosqlite:///./data/bloomtrust.db
JWT_SECRET=tu_clave_secreta_jwt_para_floristas

# CONFIGURACIÓN DE APIS REQUERIDAS
OPENAI_API_KEY=sk-proj-TU_LLAVE_DE_OPENAI_DE_5_DOLARES
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.5

# RESTRICCIONES DE MERCADO
BLOOMTRUST_MIN_PEERS=3
NYC_MARKET_REQUESTS_PER_MINUTE=75
```

---

## 🚀 Puesta en Marcha (Paso a Paso)

### 1. Backend (Instalación Local)
```bash
cd backend
uv venv --python 3.12
source .venv/bin/activate
pip install -e ".[dev]"
python -m app.db.seed          # Crea la base de datos con el catálogo de flores de NYC
uvicorn app.main:app --reload --port 8000
```
*La API interactiva de documentación quedará lista en `http://localhost:8000/docs`.*

### 2. Frontend
```bash
cd frontend
npm install --force
npm start
```
*La interfaz visual del chat interactivo para los event planners correrá en `http://localhost:4200`.*
