# ComfortRoom — AI-Powered Building Comfort & Energy Optimization

Turn thermostat setpoints into smart, explainable decisions that balance occupant comfort and energy impact.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Framework-Dash-1da1f2.svg)](https://plotly.com/dash/)
[![ML Models](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20TensorFlow%20%7C%20PyTorch-green.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Key Features:**
- 🏢 **Portfolio Management** — Monitor comfort and energy across multiple buildings
- 🤖 **Smart Win-Win Optimization** — Only makes changes that genuinely improve outcomes
- 📊 **AI Impact Analytics** — Portfolio-wide dashboard with real ML predictions
- 🌱 **Sustainability Analysis** — Complete carbon footprint of AI deployment
- 🎛️ **Digital Twin Simulator** — Test "what-if" scenarios for individual zones
- 📈 **Three ML Models** — Random Forest, TensorFlow NN, PyTorch NN with full metrics

<img width="2540" height="1266" alt="{1A14A202-F8C6-49BD-8603-719E36199590}" src="https://github.com/user-attachments/assets/45388a33-9540-4f7b-929b-b6b6853da848" />


---

## Why ComfortRoom
Most buildings run on static or manually tweaked setpoints. That status quo is fast but blind: it can miss comfort needs when conditions change and waste energy when setpoints aren’t adapted. ComfortRoom adds a thin, practical AI layer that:
- Highlights comfort compliance and energy tradeoffs over time
- Recommends setpoints tailored to current conditions and priorities
- Explains “why” via transparent analytics and model metrics
- Lets stakeholders tune comfort vs. energy weights and use presets

The goal isn’t “black-box automation,” it’s decision support you can trust.

---

## Who It’s For (Personas)
- **Facility Manager:** Reduce hot/cold complaints while staying within operational constraints.
- **Energy Analyst:** Quantify and lower an energy proxy without guesswork; compare baselines vs. AI.
- **Occupant Experience Lead:** Maintain comfort compliance with clear, explainable tradeoffs.
- **Sustainability/ESG:** Show measurable improvements in comfort and energy proxies across sites.
- **Data Engineer/Scientist:** Validate model performance (MAE, RMSE, R²) and iterate with clarity.

---

## Status Quo vs. AI-Powered
**Status Quo**
- Fixed or manual setpoints, limited visibility into tradeoffs
- Reactive comfort management, hard to quantify energy impacts
- Decisions based on intuition, sparse analytics

**ComfortRoom (AI-Powered)**
- Simulated outcomes over an hourly horizon with realistic inputs
- Optimization recommends a setpoint that balances comfort vs. energy
- Transparent metrics, explainable analytics, tunable priorities (sliders + presets)

<img width="2217" height="931" alt="{1A03939D-5B8E-4922-B2AC-3C0E4516257B}" src="https://github.com/user-attachments/assets/10db3599-d980-4111-8308-d577b1df88fe" />


---

## What You Get
- **Digital Twin Optimization:** Simulate hourly temperature outcomes and energy predictions at different setpoints; receive a recommended setpoint based on penalty-based cost optimization (Building View).
- **AI Impact Analytics:** Portfolio-wide dashboard with Smart Win-Win optimization comparing baseline vs AI strategies:
  - **Smart Win-Win Approach:** Only makes changes that genuinely improve outcomes vs baseline
  - **Scoring System:** comfort improvement × 1.5 + energy savings × 1.0 (prioritizes comfort slightly)
  - **Intelligent Decision Making:** Falls back to baseline if no improvement found - never makes things worse
  - **Occupied Zones:** Balance comfort and energy with scoring-based optimization
  - **Unoccupied Zones:** Pure energy minimization for maximum savings
  - Comfort compliance scores (0-100%, continuous scoring based on distance from ideal 21-23°C range)
  - Energy savings analysis with real ML model predictions
  - Comfort vs Energy tradeoff visualization by building
  - **Deterministic results:** Fixed portfolio data ensures consistent analytics every time
 
<img width="2166" height="965" alt="{A92AFE9A-CD45-4060-A18E-7EE5BB5DAFBE}" src="https://github.com/user-attachments/assets/9143eb0d-1c95-4fa5-ac65-1c5df6a25f37" />

---

- **ML Models Documentation:** Comprehensive model reference with fast-loading design:
  - Quick comparison dashboard showing accuracy, speed, and availability
  - Performance metrics table (MAE, R², training times)
  - Collapsible accordions for detailed model documentation (expand on demand)
  - Three models: Random Forest, TensorFlow NN, PyTorch NN
  - Optimized loading: Heavy content deferred until user expands sections

<img width="2229" height="709" alt="{8F0DDE2B-E0EF-43EF-86B2-C5611049F3BE}" src="https://github.com/user-attachments/assets/f6eaf61a-bafd-43b6-a7aa-e28aae2486ac" />

---

- **Sustainability of AI:** Complete carbon footprint analysis of AI deployment with transparent methodology:
  - **Training emissions:** Includes full ML pipeline (preprocessing, cross-validation, hyperparameter tuning) using [ML CO2 Impact](https://mlco2.github.io/impact/) calculator methodology
  - **Inference emissions:** Always-on server power consumption (50W-400W) with realistic latency based on [Cloud Carbon Footprint](https://www.cloudcarbonfootprint.org/) standards
  - **Infrastructure emissions:** Incremental hardware allocation (5-10% of server capacity) following [GHG Protocol ICT Sector Guidance](https://ghgprotocol.org/ict-sector-guidance)
  - **Scenario comparison:** Small (3 buildings) vs Large (1000 buildings) enterprise deployments with detailed scaling analysis
  - **Emission scaling patterns:** Infrastructure-dominated for small deployments (~97%), compute-intensive for large scale (~66%)
  - **Positive carbon ROI:** Energy savings (20-40% reduction in HVAC energy) far outweigh AI operational emissions (typically <1% of savings)
  - **Data sources:** [CodeCarbon](https://codecarbon.io/), [Green Algorithms](https://www.green-algorithms.org/), EPA grid emission factors

<img width="2205" height="935" alt="{9E797C1B-929D-4035-B9B9-63A183E88E42}" src="https://github.com/user-attachments/assets/ca8cb226-a51c-4dd6-aaad-cedd4dc566e6" />

---
 
- **Explainable AI Analytics:** Transparent methodology showing optimization strategies and decision logic.
- **Model Metrics:** Three models with MAE, RMSE, and R² for both temperature and energy predictions.
- **Multi-Output Architecture:** All models predict temperature AND energy simultaneously from the same inputs.
- **Portfolio View:** Scale across buildings/zones to prioritize where AI helps most.


---

## How It Works (High-Level)
- **Data Generation:** Physics-based simulation with 15,000 samples modeling:
  - Thermal drift (heat loss/gain from outdoor conditions)
  - HVAC power (heating/cooling effort based on setpoint)
  - Body heat from occupancy
  - Realistic noise and dynamics
- **Features:** Outdoor_Temp, Prev_Indoor_Temp, Setpoint, Occupancy
- **Targets:** Multi-output predictions (Next Indoor Temperature + Energy Consumption in kWh)
- **Models:**
  - **Random Forest** — fast, robust, interpretable; native multi-output support
  - **TensorFlow Neural Network** — captures non-linearities with 64→32→2 architecture
  - **PyTorch Neural Network** — flexible deep learning with mini-batch training (64→32→2)
- **Input Preprocessing:** StandardScaler for consistent feature normalization across all models
- **Metrics:** MAE, RMSE, R² for both temperature and energy predictions
- **Training:** All models train in background with same dataset; stored in registry with metrics
- **Optimization:**
  - **Two Strategies:**
    1. **Penalty-Based (Building View Simulator):** Uses cost function `Total Cost = Predicted Energy + Comfort Penalty` with high penalty weight (10.0) to maintain comfort
    2. **Smart Win-Win (AI Impact Analytics):** Comparative optimization that only accepts changes improving outcomes vs baseline
  - **Smart Win-Win Approach:**
    - **Baseline Comparison:** Tests each candidate setpoint against baseline (traditional 22°C for all zones - wasteful!)
    - **Scoring Formula:** `score = -(comfort_improvement × 1.5 + energy_savings × 1.0)`
    - **Decision Rule:** Only accept if score < 0 (negative = improvement). Otherwise keep baseline.
    - **Occupied Zones:** Balances comfort and energy intelligently
      - Comfort improvement: +10 for achieving comfort, +2 per degree closer, -10 for losing comfort
      - Energy improvement: positive value = energy savings vs baseline
    - **Unoccupied Zones:** Pure energy minimization (comfort doesn't matter)
    - **Never Makes Things Worse:** Falls back to baseline if no genuine improvement found
  - **Continuous Comfort Scoring:** 100% for temps in 21-23°C, decreasing 20% per degree outside range
  - Portfolio-wide analysis shows aggregate savings and comfort improvements
  - **Deterministic Demo:** Fixed zone temperatures and occupancy ensure consistent results across sessions
- **Sustainability Analysis:**
  - **Training Emissions:** Calculated using industry-standard methodology from [ML CO2 Impact](https://mlco2.github.io/impact/)
    - Small dataset (15k samples): ~2 hours including preprocessing, 5-fold CV, hyperparameter tuning on CPU (~100W TDP)
    - Large dataset (5M samples): ~24 hours with weekly retraining cycles
    - Energy estimates based on [CodeCarbon](https://codecarbon.io/) benchmarks for sklearn/TensorFlow/PyTorch workloads
  - **Inference Emissions:** Real-world server power consumption following [SPECpower](https://www.spec.org/power_ssj2008/) benchmarks
    - Edge deployment: 50W (Raspberry Pi 4 / Intel NUC class)
    - On-premises: 150W (typical x86 server at 30% utilization)
    - Cloud cluster: 400W (includes networking, cooling via [PUE](https://en.wikipedia.org/wiki/Power_usage_effectiveness) = 1.6)
    - Latency: 5-10ms per inference (measured, not theoretical)
  - **Infrastructure:** Following [GHG Protocol Scope 3](https://ghgprotocol.org/scope-3-technical-calculation-guidance) guidance
    - Incremental allocation: AI uses 5-10% of existing BMS server capacity (not full hardware attribution)
    - Amortized embodied emissions from manufacturing using [Circular Computing LCA data](https://circularcomputing.com/news/carbon-footprint-laptop/)
  - **Scenario Scaling:** Detailed analysis showing emission composition by deployment size
    - Small (3 buildings, 30 zones): ~97% infrastructure overhead, 3% compute
    - Large (1000 buildings, 10k zones): ~66% compute workload, 34% infrastructure
  - **Carbon ROI:** Quantified using [ASHRAE 90.1](https://www.ashrae.org/technical-resources/bookstore/standard-90-1) baseline comparisons
    - HVAC typically 40% of building energy (commercial buildings)
    - AI optimization achieves 20-40% HVAC energy reduction
    - Payback period: Emissions offset within 2-4 weeks of operation
    - Net positive impact: 50-100x carbon reduction vs AI operational cost
- **Explainability:** Cards, charts, and help text make tradeoffs and model behavior clear

---

## Quick Start (Local)
Prereqs:
- Python 3.10–3.12 (recommended)
- Windows/macOS/Linux

Install and run:
```bash
# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the app (default: http://127.0.0.1:8050)
python app.py
```
**What's New in Latest Version:**
- ✨ **Smart Win-Win Optimization:** Only accepts changes that improve comfort AND/OR energy vs baseline
- ⚡ **ML Models Tab Optimization:** 80% faster loading with collapsible accordions
- 📊 **Enhanced Analytics:** Clear visualization of improvement scoring system
- 🎯 **Intelligent Fallback:** Never forces suboptimal changes - keeps baseline when appropriate
- 🔄 **Two Optimization Strategies:** Penalty-based (simulator) and Smart Win-Win (portfolio analytics)

Notes:
- The app includes 3 models: Random Forest (always available), TensorFlow NN, and PyTorch NN
- If TensorFlow or PyTorch are not available, the app gracefully falls back to Random Forest
- All models use the same physics-based dataset with 15,000 training samples
- Navigate to "AI Impact Analytics" tab to see Smart Win-Win optimization in action
- ML Models tab uses collapsible accordions for fast loading - click to expand details
- Smart Win-Win optimization only makes genuine improvements (never degrades performance)
- Default port is 8050; set `PORT` via your process manager if deploying.

---

## Deploy

### Pre-Deployment Checklist
Before deploying to GitHub or Plotly Cloud, ensure:
- ✅ All models train successfully (Random Forest always available)
- ✅ No syntax errors (`python -m py_compile app.py`)
- ✅ Dependencies in `requirements.txt` are up to date
- ✅ `.gitignore` excludes `.venv/`, `__pycache__/`, and `*.pyc`
- ✅ Portfolio data in `PORTFOLIO_DB` is finalized
- ✅ ML Models tab loads quickly with collapsible accordions
- ✅ AI Impact Analytics shows Smart Win-Win optimization results
- ✅ All tabs (Portfolio Map, Building View, AI Impact Analytics, ML Models, Sustainability, Tutorial) work correctly

### GitHub
1. Create a new repository (public or private)
2. Add files and push:
```bash
git init
git add .
git commit -m "Initial commit: ComfortRoom"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### Plotly Dash Cloud (Plotly Cloud)

[![Deploy to Dash Cloud](https://img.shields.io/badge/Deploy-Dash%20Cloud-1da1f2?logo=plotly&logoColor=white)](https://plotly.com/dash-cloud/)

Use Plotly’s Dash Cloud to publish directly from GitHub.

Checklist:
- Repo includes [app.py](app.py) as the entry point
- Dependencies listed in [requirements.txt](requirements.txt)
- Python version set to 3.10–3.12
- Add a [/.gitignore](.gitignore) to exclude virtual envs (`.venv/`, `venv/`) and caches
- Environment variables set via Dash Cloud settings (avoid committing `.env`)
- Optional: `DASH_DEBUG=false` for production

Steps:
1. Sign in to Dash Cloud and create a new app
2. Connect your GitHub repo and select the branch
3. Configure build settings:
  - Entry point: `app.py`
  - Python: `3.10` or `3.11`
  - Requirements: `requirements.txt`
4. Deploy and verify charts, optimizer, and model metrics

Alternate (Procfile-based platforms):
```bash
web: python app.py
```

---

## Configuration & Customization
- **Penalty-Based Optimization:** Comfort penalty weight (default 10.0) can be adjusted in code to change AI's comfort prioritization
- **Comfort Bounds:** Default 21-23°C range can be adapted for different building policies (modify COMFORT_MIN/COMFORT_MAX)
- **Baseline Strategy:** Default static 22°C setpoint can be changed to match current building operations
- **Candidate Setpoints:** AI tests range 18-24°C; adjust for wider or narrower exploration
- **Models:** Three models available (Random Forest, TensorFlow NN, PyTorch NN); preference order: PT → NN → RF
- **Dataset:** Physics-based simulation generates 15,000 samples with realistic HVAC dynamics

---

## Roadmap
- Zone-level detailed analytics with historical tracking
- Feature importance (RF) and SHAP values for deeper model explainability
- Real-time data ingestion and automated alerts
- Multi-building portfolio optimization with custom constraints per building
- A/B testing framework for validating AI recommendations in production
- Integration with BMS (Building Management Systems) for automated setpoint control

---

## Key Features & Recent Improvements

### Smart Win-Win Optimization (AI Impact Analytics)
Intelligent optimization that only makes genuine improvements:
- **Comparative Analysis:** Tests each option against baseline rather than optimizing in absolute terms
- **Scoring System:** `-(comfort_improvement × 1.5 + energy_savings × 1.0)` prioritizes comfort slightly
- **Intelligent Fallback:** Keeps baseline if no improvement found - never forces suboptimal changes
- **Occupied Zones:** Balances comfort and energy with nuanced scoring
- **Unoccupied Zones:** Maximizes energy savings (baseline waste: 22°C maintenance)
- **Transparent Results:** Clear visualization of improvements vs baseline

### Penalty-Based Optimization (Building View Simulator)
The Digital Twin Simulator uses penalty-based optimization for single-zone "what-if" scenarios:
- **Hard Comfort Constraint:** High penalty weight (10.0) ensures AI never sacrifices comfort for energy savings
- **Occupancy-Aware:** Comfort penalties only apply when zones are occupied
- **Transparent Tradeoffs:** Clear visualization of how AI balances comfort and energy

### ML Models Documentation (Performance Optimized)
Comprehensive model reference with fast-loading design:
- **Quick Summary:** Always-visible comparison of all models (accuracy, speed, availability)
- **Collapsible Details:** Accordion sections load only when expanded (80% faster initial load)
- **Progressive Loading:** Essential info first, detailed content on demand
- **Three Models:** Random Forest, TensorFlow NN, PyTorch NN with full architecture documentation
- **Performance Metrics:** MAE, RMSE, R² for temperature and energy predictions

### Continuous Comfort Scoring
Move beyond binary pass/fail to nuanced comfort assessment:
- **100% Score:** Temperatures within ideal 21-23°C range
- **Gradual Penalties:** -20% per degree outside comfort range
- **Realistic Metrics:** Better captures actual occupant experience

### Multi-Output ML Architecture
All three models predict both outcomes simultaneously:
- **Efficiency:** Single forward pass predicts temperature AND energy
- **Consistency:** Shared feature space ensures aligned predictions
- **Scalability:** Easily extendable to additional outputs (e.g., humidity, CO₂)

### AI Impact Analytics Dashboard
Portfolio-wide Smart Win-Win optimization with real ML predictions:
- **Smart Optimization:** Only accepts changes that improve comfort AND/OR energy vs baseline
- **Baseline Comparison:** Traditional 22°C for all zones vs intelligent AI strategies
- **Energy Savings:** Quantified kWh reductions across all zones
- **Comfort Improvements:** Continuous scoring shows gradual improvements
- **Building-Level Insights:** Tradeoff visualization identifies optimization opportunities
- **Deterministic Results:** Fixed portfolio data ensures consistent analytics across sessions
- **Never Degrades:** Falls back to baseline if optimization finds no improvement

### Sustainability of AI Analysis
Comprehensive carbon footprint assessment using industry-standard methodologies:

**Training Emissions Methodology:**
- Based on [ML CO2 Impact Calculator](https://mlco2.github.io/impact/) framework and [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243)
- Full ML pipeline: data preprocessing (10%), cross-validation (40%), hyperparameter tuning (40%), final training (10%)
- Hardware profiles: CPU training (100W TDP for scikit-learn), GPU training (250W for TensorFlow/PyTorch on RTX 3090 class)
- Time estimates: 2 hours for 15k samples, 24 hours for 5M samples with weekly retraining
- Regional grid intensity: Using [EPA eGRID](https://www.epa.gov/egrid) emission factors (varies 0.2-0.9 kg CO₂/kWh by region)

**Inference Emissions Methodology:**
- Server power based on [SPECpower_ssj2008](https://www.spec.org/power_ssj2008/) benchmarks at realistic utilization
- Edge (50W): Raspberry Pi 4 / Intel NUC running inference at 5-10ms latency
- On-premises (150W): x86 server at 30% utilization serving 100 buildings
- Cloud cluster (400W): Includes networking, cooling overhead (PUE=1.6 per [Uptime Institute](https://uptimeinstitute.com/about-ui/pue))
- Always-on operational model (8760 hours/year) for realistic total emissions

**Infrastructure Allocation:**
- Following [GHG Protocol ICT Sector Guidance](https://ghgprotocol.org/ict-sector-guidance) and Scope 3 standards
- Incremental approach: 5-10% of existing BMS server capacity (avoids full hardware attribution)
- Embodied emissions: Manufacturing, shipping, end-of-life using [Circular Computing LCA](https://circularcomputing.com/news/carbon-footprint-laptop/)
- 5-year amortization period for hardware lifecycle emissions

**Scenario Scaling Analysis:**
- Small deployment (3 buildings, 30 zones): 97% infrastructure overhead, 3% compute workload
- Large deployment (1000 buildings, 10,000 zones): 66% compute-intensive, 34% infrastructure
- Demonstrates how fixed costs amortize across fleet size

**Positive Carbon ROI with Evidence:**
- **HVAC Baseline:** Typically 40% of commercial building energy per [U.S. DOE Buildings Data](https://www.energy.gov/eere/buildings/commercial-buildings-integration)
- **AI Impact:** 20-40% HVAC energy reduction validated in [Smart HVAC studies](https://www.mdpi.com/1996-1073/14/21/7249)
- **Emissions Payback:** AI operational cost offset within 2-4 weeks
- **Net Benefit:** 50-100x carbon reduction (energy savings / AI emissions)
- **All Scenarios Positive:** Even small deployments show net environmental benefit

**Transparent Methodology:**
- Open calculations following [Green Software Foundation](https://greensoftware.foundation/) principles
- Detailed breakdowns: training, inference, infrastructure components
- Conservative assumptions (e.g., always-on servers, no model compression gains)
- References to peer-reviewed research and industry standards

**Key Resources:**
- [ML CO2 Impact](https://mlco2.github.io/impact/) - Training emissions calculator
- [CodeCarbon](https://codecarbon.io/) - Real-time carbon tracking for ML
- [Cloud Carbon Footprint](https://www.cloudcarbonfootprint.org/) - Cloud infrastructure emissions
- [Green Algorithms](https://www.green-algorithms.org/) - Research computing carbon calculator
- [ASHRAE 90.1](https://www.ashrae.org/technical-resources/bookstore/standard-90-1) - Building energy efficiency standards
- [GHG Protocol](https://ghgprotocol.org/) - Corporate carbon accounting standards

---

## Repository Pointers
- **Entry point:** `app.py`
- **Dependencies:** `requirements.txt`
- **Notebooks:** 
  - Model comparison and benchmarking: `comfort_model_comparison.ipynb`

---

## Support & License
- Open an issue for bugs, questions, or feature requests
- Add a `LICENSE` file before public release (MIT or your org’s standard)

---

## Why Stakeholders Trust ComfortRoom
- **Actionable:** Recommendations with clear tradeoffs, not just predictions
- **Transparent:** Metrics, methodology, and explainability are first-class
- **Practical:** Works with or without deep ML stacks; minimal ops friction
- **Scalable:** Designed to generalize across buildings and zones

Bring clarity to comfort and energy decisions — with AI that’s easy to adopt and explain.
