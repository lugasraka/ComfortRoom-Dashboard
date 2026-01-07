# ComfortRoom Technical Specification

**Version:** 1.0
**Last Updated:** 2026-01-06
**Status:** Production

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Data Models](#3-data-models)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Optimization Algorithms](#5-optimization-algorithms)
6. [User Interface](#6-user-interface)
7. [Performance Requirements](#7-performance-requirements)
8. [API Specifications](#8-api-specifications)
9. [Deployment](#9-deployment)
10. [Configuration](#10-configuration)
11. [Dependencies](#11-dependencies)

---

## 1. System Overview

### 1.1 Purpose

ComfortRoom is an AI-powered decision support system for building HVAC management that:
- Provides transparent, explainable recommendations for thermostat setpoints
- Balances occupant comfort with energy efficiency
- Tracks portfolio-wide performance across multiple buildings
- Quantifies environmental impact of AI deployment

### 1.2 Key Stakeholders

| Role | Use Case |
|------|----------|
| Facility Manager | Reduce temperature complaints while staying within operational constraints |
| Energy Analyst | Quantify and reduce energy consumption with data-driven decisions |
| Occupant Experience Lead | Maintain comfort compliance with explainable tradeoffs |
| Sustainability/ESG | Demonstrate measurable improvements in comfort and energy metrics |
| Data Scientist | Validate model performance and iterate on ML pipeline |

### 1.3 Core Principles

- **Transparency over automation** - Decision support, not black-box control
- **Explainability first** - Clear metrics and methodology
- **Never degrade performance** - Smart Win-Win optimization only accepts improvements
- **Practical deployment** - Works with minimal ML stack (Random Forest only)

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Dash Web Application                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Portfolio   │  │   Building   │  │     AI       │     │
│  │     Map      │  │     View     │  │   Impact     │     │
│  │              │  │  (Simulator) │  │  Analytics   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  ML Models   │  │Sustainability│  │   Tutorial   │     │
│  │     Docs     │  │  of AI       │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                      Callback Layer                          │
│   (Event handlers, state management, data flow control)     │
├─────────────────────────────────────────────────────────────┤
│                     Business Logic                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Optimization │  │  Prediction  │  │Sustainability│     │
│  │   Engine     │  │   Engine     │  │  Calculator  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                       ML Model Layer                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Random    │  │  TensorFlow  │  │   PyTorch    │     │
│  │    Forest    │  │     NN       │  │     NN       │     │
│  │  (Required)  │  │  (Optional)  │  │  (Optional)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                        Data Layer                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  training_data.csv (15,000 samples)              │      │
│  │  - Physics-based HVAC simulation                 │      │
│  │  - Features: Outdoor temp, indoor temp, setpoint │      │
│  │  - Targets: Next temp, energy consumption        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  PORTFOLIO_DB (In-memory)                        │      │
│  │  - 3 buildings × 5 zones = 15 zones             │      │
│  │  - Fixed values for deterministic demo           │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Plotly Dash | Latest |
| UI Components | Dash Bootstrap Components | Latest |
| ML - Random Forest | scikit-learn | Latest |
| ML - Neural Network | TensorFlow | 2.18.0+ (optional) |
| ML - Neural Network | PyTorch | 2.0.0+ (optional) |
| Data Processing | NumPy, Pandas | Latest |
| Visualization | Plotly | Latest |
| Runtime | Python | 3.10-3.12 |

### 2.3 Data Flow

```
User Interaction
      ↓
Dash Callback (app.py)
      ↓
Business Logic (optimization/prediction functions)
      ↓
ML Model Prediction (RF/NN/PT)
      ↓
Results Processing
      ↓
UI Component Update
      ↓
User Display
```

---

## 3. Data Models

### 3.1 Training Data Schema

**File:** `training_data.csv`
**Size:** 15,000 samples
**Generation:** `build_training_data.ipynb`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Outdoor_Temp` | float32 | -10°C to 40°C | Outdoor ambient temperature |
| `Prev_Indoor_Temp` | float32 | 15°C to 28°C | Current indoor temperature |
| `Setpoint` | float32 | 18°C to 24°C | Thermostat setpoint |
| `Occupancy` | int | 0 or 1 | Zone occupancy status (0=empty, 1=occupied) |
| `Target_Temp` | float32 | 15°C to 28°C | Predicted next indoor temperature |
| `Target_Energy` | float32 | 0+ kWh | Predicted energy consumption |

**Physics Model:**
```python
# Temperature dynamics
thermal_drift = 0.05 * (outdoor_temp - prev_indoor_temp)
hvac_power = 0.3 * (setpoint - prev_indoor_temp)
body_heat = 0.1 * occupancy
next_temp = prev_temp + thermal_drift + hvac_power + body_heat + noise

# Energy consumption
base_load = 0.5  # Fans, lights
hvac_kwh = abs(hvac_power) * 5
total_energy = base_load + hvac_kwh + (0.2 * occupancy) + noise
```

### 3.2 Portfolio Database Schema

**Location:** In-memory (app.py:330)
**Type:** Python dictionary

```python
PORTFOLIO_DB = {
    '<building_name>': {
        'meta': {
            'id': str,           # Building ID (e.g., "B1")
            'name': str,         # Building name
            'lat': float,        # Latitude
            'lon': float,        # Longitude
            'type': str          # Building type (Office/R&D Lab/Warehouse)
        },
        'zones': [
            {
                'name': str,     # Zone name (e.g., "Lobby")
                'temp': float,   # Current temperature (°C)
                'occupied': bool, # Occupancy status
                'status': str    # Status ("OK"/"Warning"/"Critical")
            },
            # ... 5 zones per building
        ]
    },
    # ... 3 buildings total
}
```

**Buildings:**
1. **Comfort Room - Zug** (Office) - 47.1662°N, 8.5155°E
2. **Tech Hub - Munich** (R&D Lab) - 48.1351°N, 11.5820°E
3. **Logistics - Milan** (Warehouse) - 45.4642°N, 9.1900°E

### 3.3 Model Registry Schema

**Location:** Global (app.py:338-342)

```python
MODEL_REGISTRY = {
    'RF': {  # Random Forest
        'model': RandomForestRegressor or None,
        'scaler': StandardScaler or None,
        'metrics': {
            'train_time_s': float,
            'mae_temp': float,
            'mae_energy': float,
            'rmse_temp': float,
            'rmse_energy': float,
            'r2_temp': float,
            'r2_energy': float,
            'n_estimators': int,
            'max_depth': int
        }
    },
    'NN': { /* TensorFlow model */ },
    'PT': { /* PyTorch model */ }
}
```

---

## 4. Machine Learning Models

### 4.1 Model Specifications

#### Random Forest (scikit-learn)

**Architecture:**
- Multi-output regressor (native support)
- 50 estimators
- Max depth: 10
- Random state: 42 (reproducibility)

**Training:**
- Input: 4 features (scaled with StandardScaler)
- Output: 2 targets (temperature, energy)
- Train/test split: 80/20
- Training time: ~1-2 seconds

**Metrics (typical):**
```
Temperature Prediction:
  MAE: 0.15-0.25°C
  RMSE: 0.20-0.30°C
  R²: 0.95-0.98

Energy Prediction:
  MAE: 0.08-0.12 kWh
  RMSE: 0.10-0.15 kWh
  R²: 0.92-0.96
```

#### TensorFlow Neural Network (optional)

**Architecture:**
```
Input Layer: 4 features (scaled)
Hidden Layer 1: 64 neurons, ReLU activation
Hidden Layer 2: 32 neurons, ReLU activation
Output Layer: 2 neurons (temp, energy)
```

**Training:**
- Optimizer: Adam
- Loss: MSE
- Epochs: 20
- Batch size: 32
- Training time: ~3-5 seconds

**Parameters:** ~2,500 trainable parameters

#### PyTorch Neural Network (optional)

**Architecture:**
```python
class MultiTaskNet(nn.Module):
    fc1: Linear(4, 64)
    fc2: Linear(64, 32)
    out: Linear(32, 2)
    activation: ReLU
```

**Training:**
- Optimizer: Adam (lr=0.005)
- Loss: MSELoss
- Epochs: 50
- Batch size: 32
- DataLoader: Shuffle enabled
- Training time: ~8-12 seconds

**Parameters:** ~2,500 trainable parameters

### 4.2 Model Selection Strategy

**Priority Order:** PyTorch → TensorFlow → Random Forest

**Fallback Logic:**
1. Try PyTorch (best accuracy)
2. If unavailable, try TensorFlow
3. If unavailable, use Random Forest (always available)

**Production Deployment:**
- Cloud (Plotly Cloud): Random Forest only
- Local/On-premises: All three models available

### 4.3 Input Preprocessing

**Standardization:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Each feature standardized to:
# mean = 0, std = 1
```

**Feature Ranges (pre-scaling):**
- Outdoor_Temp: μ=15°C, σ=8°C
- Prev_Indoor_Temp: μ=21°C, σ=2°C
- Setpoint: 18-24°C (discrete steps of 0.5°C)
- Occupancy: {0, 1}

---

## 5. Optimization Algorithms

### 5.1 Penalty-Based Optimization (Building View)

**Purpose:** Single-zone "what-if" scenarios in Digital Twin Simulator

**Cost Function:**
```python
total_cost = predicted_energy + comfort_penalty

where:
  comfort_penalty = PENALTY_WEIGHT * max(0, deviation_from_comfort)
  PENALTY_WEIGHT = 10.0  # High weight ensures comfort priority
  deviation_from_comfort = distance from [21°C, 23°C] range
```

**Algorithm:**
```python
def optimize_penalty_based(zone_state, outdoor_temp, occupancy):
    best_cost = infinity
    best_setpoint = None

    for setpoint in CANDIDATE_SETPOINTS:  # 16-26°C
        # Predict outcomes
        pred_temp, pred_energy = model.predict([outdoor_temp,
                                                 zone_state.temp,
                                                 setpoint,
                                                 occupancy])

        # Calculate comfort penalty
        if occupancy == 1:
            if pred_temp < COMFORT_MIN:
                penalty = (COMFORT_MIN - pred_temp) * PENALTY_WEIGHT
            elif pred_temp > COMFORT_MAX:
                penalty = (pred_temp - COMFORT_MAX) * PENALTY_WEIGHT
            else:
                penalty = 0
        else:
            penalty = 0  # No comfort requirement when empty

        # Total cost
        cost = pred_energy + penalty

        if cost < best_cost:
            best_cost = cost
            best_setpoint = setpoint

    return best_setpoint, best_cost
```

### 5.2 Smart Win-Win Optimization (AI Impact Analytics)

**Purpose:** Portfolio-wide optimization with baseline comparison

**Philosophy:** Only accept changes that genuinely improve outcomes vs baseline

**Baseline Strategy:**
- Occupied zones: 22°C (traditional comfort setpoint)
- Unoccupied zones: 22°C (wasteful - maintains temp 24/7)

**Scoring Function:**
```python
score = -(comfort_improvement × 1.5 + energy_savings × 1.0)

# Negative score = improvement
# Accept if: score < 0
# Otherwise: keep baseline
```

**Comfort Improvement Calculation (Occupied Zones):**
```python
baseline_comfort = (COMFORT_MIN <= baseline_temp <= COMFORT_MAX)
candidate_comfort = (COMFORT_MIN <= candidate_temp <= COMFORT_MAX)

if candidate_comfort and not baseline_comfort:
    comfort_improvement = +10.0  # Achieved comfort
elif candidate_comfort and baseline_comfort:
    comfort_improvement = 0.0    # Both comfortable
elif not candidate_comfort and not baseline_comfort:
    # Both uncomfortable - measure which is closer
    baseline_deviation = min(abs(baseline_temp - COMFORT_MIN),
                            abs(baseline_temp - COMFORT_MAX))
    candidate_deviation = min(abs(candidate_temp - COMFORT_MIN),
                             abs(candidate_temp - COMFORT_MAX))
    comfort_improvement = (baseline_deviation - candidate_deviation) × 2.0
else:
    comfort_improvement = -10.0  # Lost comfort
```

**Energy Improvement:**
```python
energy_improvement = baseline_energy - candidate_energy
# Positive = savings
# Negative = increased consumption
```

**Algorithm:**
```python
def optimize_smart_win_win(zone_state, outdoor_temp, occupancy):
    # Predict baseline outcomes
    baseline_setpoint = 22.0
    baseline_temp, baseline_energy = model.predict([outdoor_temp,
                                                     zone_state.temp,
                                                     baseline_setpoint,
                                                     occupancy])

    if occupancy == 0:
        # Unoccupied: Pure energy minimization
        best_idx = argmin(candidate_energies)
        return CANDIDATE_SETPOINTS[best_idx]

    # Occupied: Find best win-win scenario
    best_score = 0  # Baseline score
    best_setpoint = baseline_setpoint

    for setpoint in CANDIDATE_SETPOINTS:
        candidate_temp, candidate_energy = model.predict([...])

        comfort_improvement = calculate_comfort_improvement(
            baseline_temp, candidate_temp)
        energy_improvement = baseline_energy - candidate_energy

        score = -(comfort_improvement × 1.5 + energy_improvement × 1.0)

        if score < best_score:  # Improvement found
            best_score = score
            best_setpoint = setpoint

    return best_setpoint
```

### 5.3 Continuous Comfort Scoring

**Purpose:** Nuanced comfort assessment beyond binary pass/fail

**Formula:**
```python
def comfort_score(temperature):
    if COMFORT_MIN <= temperature <= COMFORT_MAX:
        return 100.0  # Perfect comfort

    # Calculate deviation
    if temperature < COMFORT_MIN:
        deviation = COMFORT_MIN - temperature
    else:  # temperature > COMFORT_MAX
        deviation = temperature - COMFORT_MAX

    # Penalty: -20% per degree
    score = 100.0 - (deviation * 20.0)

    return max(0.0, score)  # Floor at 0%
```

**Examples:**
- 22°C → 100%
- 20°C → 80% (1°C below comfort)
- 19°C → 60% (2°C below comfort)
- 26°C → 40% (3°C above comfort)
- 17°C → 0% (4°C+ below comfort)

---

## 6. User Interface

### 6.1 Application Layout

**Sidebar (Fixed Left, 20rem width):**
- Logo and branding
- View selector (6 tabs)
- Building/zone filters (contextual)
- Model selector (for analytics)

**Content Area (Main, Right of sidebar):**
- Dynamic content based on selected view
- Responsive cards and charts
- Interactive controls

### 6.2 Views and Features

#### 6.2.1 Portfolio Map

**Purpose:** Geographic overview of all buildings

**Components:**
- Interactive Plotly map (lat/lon coordinates)
- Building markers color-coded by status
- Summary statistics cards
- Alert indicators

**Data Displayed:**
- Building locations
- Zone count per building
- Comfort compliance rates
- Critical alerts

#### 6.2.2 Building View (Digital Twin Simulator)

**Purpose:** Single-zone "what-if" scenarios with penalty-based optimization

**Inputs:**
- Building selector (dropdown)
- Zone selector (dropdown)
- Outdoor temperature (slider: -10°C to 40°C)
- Occupancy toggle

**Outputs:**
- Recommended setpoint
- Predicted next temperature
- Predicted energy consumption
- Hourly simulation chart (12 hours)
- Cost breakdown (energy vs comfort penalty)

**Visualization:**
- Line chart: Temperature over time
- Bar chart: Energy consumption per setpoint
- Metrics cards: MAE, RMSE, R² for selected model

#### 6.2.3 AI Impact Analytics

**Purpose:** Portfolio-wide Smart Win-Win optimization results

**Key Metrics:**
- Baseline energy consumption
- Optimized energy consumption
- Energy savings (kWh, %)
- Comfort compliance (baseline vs optimized)
- Cost savings

**Visualizations:**
1. **Savings Overview Card**
   - Total kWh saved
   - Total cost saved
   - Percent improvement

2. **Comfort vs Energy Scatter Plot**
   - X-axis: Energy consumption
   - Y-axis: Comfort score
   - Points: Individual zones
   - Color: Building
   - Baseline vs Optimized comparison

3. **Building-Level Breakdown Table**
   - Columns: Building, Zones, Baseline Energy, Optimized Energy, Savings
   - Sortable and filterable

4. **Zone-Level Results Table**
   - Columns: Zone, Baseline Setpoint, AI Setpoint, Temp, Energy, Comfort
   - Decision explanation (why AI chose this setpoint)

**Model Selection:**
- Dropdown to choose RF/NN/PT
- Model availability indicator
- Performance metrics display

#### 6.2.4 ML Models Documentation

**Purpose:** Comprehensive model reference with fast loading

**Structure:**
- Quick comparison table (always visible)
  - Model name
  - Accuracy (R² temp/energy)
  - Speed (training time)
  - Availability status

- Collapsible accordions (expand on demand)
  - Model 1: Random Forest
  - Model 2: TensorFlow NN
  - Model 3: PyTorch NN

**Each Accordion Contains:**
- Architecture diagram/description
- Hyperparameters
- Performance metrics (MAE, RMSE, R²)
- Training details
- Use cases and recommendations
- Code snippets (optional)

**Performance Optimization:**
- Accordions collapsed by default
- Content loads only when expanded
- ~80% faster initial page load

#### 6.2.5 Sustainability of AI

**Purpose:** Complete carbon footprint analysis of AI deployment

**Scenarios:**
1. **Small (Demo):** 3 buildings, 5 zones each
2. **Large (Enterprise):** 1000 buildings, 10 zones each

**Emission Categories:**
- **Training:** ML pipeline emissions (preprocessing, CV, hyperparameter tuning)
- **Inference:** Server power consumption (always-on, per prediction)
- **Infrastructure:** Incremental hardware allocation + datacenter overhead

**Visualizations:**
1. **Emission Breakdown Stacked Bar Chart**
   - X-axis: Scenarios
   - Y-axis: kg CO₂/year
   - Stacks: Training, Inference, Infrastructure

2. **Carbon ROI Chart**
   - Emissions cost vs energy savings benefit
   - Payback period indicator
   - Net benefit calculation

3. **Scaling Analysis**
   - Shows how emission composition changes with scale
   - Small: 97% infrastructure, 3% compute
   - Large: 66% compute, 34% infrastructure

**Methodology Display:**
- Transparent calculation sources
- Links to industry standards (ML CO2 Impact, ASHRAE, GHG Protocol)
- Conservative assumptions documented

**Key Metrics:**
- Total AI emissions (kg CO₂/year)
- Energy savings from HVAC optimization (kWh/year)
- CO₂ savings from energy reduction (kg CO₂/year)
- Net benefit (kg CO₂/year)
- ROI ratio (savings / emissions)

#### 6.2.6 Tutorial

**Purpose:** User onboarding and feature explanation

**Content:**
- Quick start guide
- Feature walkthrough
- Use case examples
- Best practices
- FAQ

### 6.3 UI Component Library

**Dash Bootstrap Components:**
- Cards (dbc.Card)
- Tabs (dbc.Tabs)
- Buttons (dbc.Button)
- Dropdowns (dcc.Dropdown)
- Sliders (dcc.Slider)
- Radio items (dbc.RadioItems)
- Alerts (dbc.Alert)
- Accordions (dbc.Accordion)

**Plotly Charts:**
- Scatter plots (go.Scatter)
- Bar charts (go.Bar)
- Line charts (go.Scatter with mode='lines')
- Maps (px.scatter_mapbox)
- Heatmaps (go.Heatmap)

### 6.4 Theming

**Bootstrap Theme:** LUX (dbc.themes.LUX)

**Color Scheme:**
- Primary: Blue (#1da1f2)
- Success: Green (comfort achieved)
- Warning: Yellow (minor issues)
- Danger: Red (critical alerts)
- Info: Light blue (informational)

**Typography:**
- Headers: Bootstrap default
- Body: Monospace font (rendered in terminal)
- Code: Monospace with syntax highlighting

---

## 7. Performance Requirements

### 7.1 Response Time Targets

| Operation | Target | Measured |
|-----------|--------|----------|
| App startup | < 5s | ~2-3s |
| Model training (all 3) | < 30s | ~15-20s |
| Page navigation | < 200ms | ~100-150ms |
| Single prediction | < 10ms | ~2-5ms |
| Batch prediction (15 zones) | < 30ms | ~10-20ms |
| Analytics tab load | < 500ms | ~200-300ms |
| Sustainability tab load | < 300ms | ~100-150ms (cached) |

### 7.2 Optimization Techniques (Implemented)

**P1: Shared Training Data (100-300ms savings)**
```python
# Load once at startup
_GLOBAL_TRAINING_X, _GLOBAL_TRAINING_Y = _load_training_data()

# Shared across all model training functions
```

**P2: Batch Predictions (30-50ms savings)**
```python
# Batch all baseline predictions (15 zones → 1 call)
baseline_inputs = np.array([...all zones...])
baseline_preds = model.predict(baseline_inputs_scaled)
```

**P3: Module Constants (5ms savings)**
```python
# Define once at module level
CANDIDATE_SETPOINTS = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
COMFORT_MIN, COMFORT_MAX = 21.0, 23.0
```

**P4: Function Caching (50-100ms savings)**
```python
# Cache CO2 calculations for 5 minutes
_CACHED_CO2_DATA = None
_CACHED_CO2_TIMESTAMP = None

def calculate_co2_impacts():
    if _CACHED_CO2_DATA and (time.time() - _CACHED_CO2_TIMESTAMP < 300):
        return _CACHED_CO2_DATA
    # ... expensive calculation ...
```

### 7.3 Scalability Considerations

**Current Scale:**
- 3 buildings
- 5 zones per building = 15 total zones
- 11 candidate setpoints per optimization
- 15,000 training samples

**Expected Scale (Enterprise):**
- 100-1000 buildings
- 5-10 zones per building = 500-10,000 zones
- Same candidate setpoints
- 1M-5M training samples (periodic retraining)

**Scaling Strategies:**
- Vectorized NumPy operations (avoid Python loops)
- Batch predictions (reduce API overhead)
- Result caching (avoid redundant calculations)
- Asynchronous model training (background threads)
- Database for persistent storage (future)

### 7.4 Memory Usage

| Component | Memory Footprint |
|-----------|------------------|
| Training data (15k samples) | ~5 MB |
| Random Forest model | ~10 MB |
| TensorFlow model | ~1 MB |
| PyTorch model | ~1 MB |
| Portfolio database | < 1 KB |
| Dash app overhead | ~50 MB |
| **Total (all models)** | **~70 MB** |

---

## 8. API Specifications

### 8.1 Internal Function APIs

#### Model Training Functions

**`train_models_rf(X=None, y=None)`**

```python
Parameters:
  X: np.ndarray (optional) - Training features (n_samples, 4)
  y: np.ndarray (optional) - Training targets (n_samples, 2)

Returns:
  model: RandomForestRegressor - Trained model
  scaler: StandardScaler - Fitted scaler
  metrics: dict - Performance metrics
    {
      'train_time_s': float,
      'mae_temp': float,
      'mae_energy': float,
      'rmse_temp': float,
      'rmse_energy': float,
      'r2_temp': float,
      'r2_energy': float,
      'n_estimators': int,
      'max_depth': int
    }
```

**`train_models_nn(X=None, y=None)`**

```python
Parameters:
  X: np.ndarray (optional) - Training features
  y: np.ndarray (optional) - Training targets

Returns:
  model: tf.keras.Sequential - Trained TensorFlow model
  scaler: StandardScaler - Fitted scaler
  metrics: dict - Performance metrics (same structure as RF, plus 'params')

Note: Returns (None, None, {'error': 'TensorFlow not available'}) if TF unavailable
```

**`train_models_pt(X=None, y=None)`**

```python
Parameters:
  X: np.ndarray (optional) - Training features
  y: np.ndarray (optional) - Training targets

Returns:
  model: torch.nn.Module - Trained PyTorch model
  scaler: StandardScaler - Fitted scaler
  metrics: dict - Performance metrics (same structure as RF, plus 'params')

Note: Returns (None, None, {'error': 'PyTorch not available'}) if PT unavailable
```

#### Optimization Functions

**`calculate_actual_savings_from_demo()`**

```python
Returns:
  savings_pct: float - Energy savings percentage (0.0 to 1.0)

Side Effects:
  - Caches result in _CACHED_SAVINGS_PCT
  - Uses Random Forest model from MODEL_REGISTRY

Performance:
  - Cached after first call
  - ~50-100ms uncached, ~1ms cached
```

**`calculate_co2_impacts()`**

```python
Returns:
  results: dict - Comprehensive CO2 analysis
    {
      'Random Forest': {
        'Small (Demo)': {
          'training_co2_kg': float,
          'inference_co2_kg': float,
          'infrastructure_co2_kg': float,
          'total_ai_co2_kg': float,
          'energy_saved_kwh': float,
          'co2_saved_kg': float,
          'net_benefit_kg': float,
          'roi_ratio': float,
          'predictions_per_year': int,
          'retraining_per_year': float,
          'total_zones': int,
          'training_percentage': float,
          'inference_percentage': float,
          'infrastructure_percentage': float
        },
        'Large (Enterprise)': { /* same structure */ }
      },
      '_baseline': {
        'hvac_energy_kwh': float,
        'hvac_co2_kg': float,
        'total_building_energy_kwh': float,
        'carbon_intensity': float,
        'actual_savings_pct': float
      },
      '_scenarios': { /* scenario definitions */ }
    }

Performance:
  - Cached for 5 minutes (300s)
  - ~100-200ms uncached, ~1ms cached
```

#### Prediction Functions

**Model Prediction (all models)**

```python
# Random Forest
predictions = model.predict(X_scaled)

# TensorFlow
predictions = model.predict(X_scaled, verbose=0)

# PyTorch
with torch.no_grad():
    predictions = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

Input:
  X_scaled: np.ndarray (n_samples, 4) - Scaled features

Output:
  predictions: np.ndarray (n_samples, 2)
    [:, 0] - Predicted temperature
    [:, 1] - Predicted energy
```

### 8.2 Dash Callback Specifications

**Primary Callback:**

```python
@app.callback(
    Output("page-content", "children"),
    [
        Input("view-selector", "value"),
        Input("building-dropdown", "value"),
        Input("zone-dropdown", "value"),
        Input("model-selector", "value")
    ]
)
def render_page(view_mode, selected_building, selected_zone, selected_model):
    """
    Main routing callback for page content.

    Parameters:
      view_mode: str - Selected view ("portfolio", "analytics", "building", etc.)
      selected_building: str - Building name
      selected_zone: str - Zone name
      selected_model: str - Model key ("RF", "NN", "PT")

    Returns:
      children: dash.html.Div - Page content components
    """
```

**Building/Zone Filter Callbacks:**

```python
@app.callback(
    Output("zone-dropdown", "options"),
    Input("building-dropdown", "value")
)
def update_zone_options(selected_building):
    """Update zone dropdown based on selected building."""

@app.callback(
    Output("building-filters", "style"),
    Input("view-selector", "value")
)
def toggle_filters(view_mode):
    """Show/hide filters based on view."""
```

---

## 9. Deployment

### 9.1 Local Development

**Prerequisites:**
- Python 3.10-3.12
- pip package manager
- Virtual environment (recommended)

**Setup:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

**Access:**
- URL: http://127.0.0.1:8050
- Default port: 8050 (configurable via PORT env var)

### 9.2 Plotly Cloud Deployment

**Requirements:**
- GitHub repository
- Plotly Cloud account
- `requirements.txt` with dependencies

**Configuration:**
- Entry point: `app.py`
- Python version: 3.10 or 3.11
- Build command: `pip install -r requirements.txt`

**Deployment Steps:**
1. Push code to GitHub
2. Connect repo to Plotly Cloud
3. Configure build settings
4. Deploy and verify

**Production Optimizations:**
- TensorFlow/PyTorch excluded from requirements (optional)
- Random Forest only (smaller footprint)
- DASH_DEBUG=false

**Live Demo:**
https://7a2b294d-8130-4503-8e04-d2e73333da12.plotly.app

### 9.3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8050 | Server port |
| `DASH_DEBUG` | False | Enable debug mode |
| `HOST` | 127.0.0.1 | Server host |

### 9.4 File Structure

```
ComfortRoom/
├── app.py                            # Main application
├── training_data.csv                 # Pre-generated training data
├── requirements.txt                  # Python dependencies
├── README.md                         # User documentation
├── spec.md                          # Technical specification (this file)
├── TRAINING_DATA_SETUP.md           # Data generation docs
├── plotly-cloud.toml                # Plotly Cloud config
├── .gitignore                       # Git ignore rules
├── build_training_data.ipynb        # Data generation notebook
└── comfort_model_comparison.ipynb   # Model benchmarking notebook
```

---

## 10. Configuration

### 10.1 Optimization Parameters

**Module Constants (app.py:352-354):**

```python
# Candidate setpoints tested during optimization
CANDIDATE_SETPOINTS = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

# Comfort temperature range
COMFORT_MIN = 21.0  # °C
COMFORT_MAX = 23.0  # °C
```

**Penalty-Based Optimization:**

```python
PENALTY_WEIGHT = 10.0  # High weight ensures comfort is prioritized
```

**Smart Win-Win Scoring:**

```python
# Scoring formula
score = -(comfort_improvement × 1.5 + energy_savings × 1.0)

# Weights:
COMFORT_WEIGHT = 1.5  # Slightly prioritize comfort
ENERGY_WEIGHT = 1.0
```

**Baseline Strategy:**

```python
BASELINE_SETPOINT_OCCUPIED = 22.0    # Traditional comfort setpoint
BASELINE_SETPOINT_UNOCCUPIED = 22.0  # Wasteful (maintains temp 24/7)
```

### 10.2 Model Hyperparameters

**Random Forest:**

```python
RandomForestRegressor(
    n_estimators=50,      # Number of trees
    max_depth=10,         # Maximum tree depth
    random_state=42       # Reproducibility
)
```

**TensorFlow Neural Network:**

```python
tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Training
optimizer='adam'
loss='mse'
epochs=20
batch_size=32
```

**PyTorch Neural Network:**

```python
# Architecture (same as TensorFlow)
# Training
optimizer=Adam(lr=0.005)
criterion=MSELoss()
epochs=50
batch_size=32
```

### 10.3 Cache Configuration

```python
# CO2 calculation cache TTL
CO2_CACHE_TTL = 300  # seconds (5 minutes)

# Savings calculation cache (persistent until restart)
# Uses _CACHED_SAVINGS_PCT (no TTL)
```

### 10.4 Portfolio Configuration

**Building Definitions (app.py:46-49):**

```python
buildings = [
    {"id": "B1", "name": "Comfort Room - Zug",
     "lat": 47.1662, "lon": 8.5155, "type": "Office"},
    {"id": "B2", "name": "Tech Hub - Munich",
     "lat": 48.1351, "lon": 11.5820, "type": "R&D Lab"},
    {"id": "B3", "name": "Logistics - Milan",
     "lat": 45.4642, "lon": 9.1900, "type": "Warehouse"}
]
```

**Zone Configurations (app.py:57-79):**

Hardcoded for deterministic demo. Modify `zone_configs` dictionary to change zone properties.

### 10.5 Sustainability Calculation Parameters

**Carbon Intensity:**

```python
CARBON_INTENSITY = 475  # g CO₂/kWh (global grid average)
```

**Training Power Consumption:**

```python
# Dataset size → Power consumption
if dataset_size <= 15000:
    train_power_w = 250   # Multi-core CPU
elif dataset_size <= 100000:
    train_power_w = 600   # High-performance server
else:
    train_power_w = 2000  # Multi-node cluster with GPUs
```

**Inference Power Consumption:**

```python
# Building count → Server configuration
if buildings <= 10:
    inference_power_w = 50   # Edge server
elif buildings <= 100:
    inference_power_w = 150  # Inference cluster
else:
    inference_power_w = 400  # Multi-server cluster
```

---

## 11. Dependencies

### 11.1 Core Dependencies (Required)

```txt
dash>=2.0.0
dash-bootstrap-components>=1.0.0
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### 11.2 Optional Dependencies

```txt
tensorflow>=2.18.0  # TensorFlow Neural Network
torch>=2.0.0        # PyTorch Neural Network
```

### 11.3 Python Version

**Supported:** Python 3.10, 3.11, 3.12
**Recommended:** Python 3.11

### 11.4 System Requirements

**Minimum:**
- 2 CPU cores
- 4 GB RAM
- 100 MB disk space

**Recommended:**
- 4+ CPU cores
- 8 GB RAM
- 500 MB disk space (with all models)

---

## 12. Testing and Validation

### 12.1 Model Validation

**Train/Test Split:**
- 80% training, 20% testing
- Random state: 42 (reproducible)

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

**Cross-Validation:**
- Not implemented in production app
- Performed during development in notebooks

### 12.2 Optimization Validation

**Deterministic Results:**
- Fixed portfolio data (`PORTFOLIO_DB`)
- Fixed outdoor temperatures per building
- Consistent results across sessions

**Validation Criteria:**
- Smart Win-Win never degrades performance vs baseline
- Comfort violations minimized
- Energy savings quantifiable

### 12.3 Performance Testing

**Syntax Validation:**
```bash
python -m py_compile app.py
```

**Load Testing:**
- Manual testing of all 6 views
- Verify response times meet targets
- Check cache effectiveness

**Integration Testing:**
- Model training completes successfully
- All predictions return valid results
- UI components render correctly

---

## 13. Future Enhancements

### 13.1 Planned Features (Roadmap)

**Short-term (1-3 months):**
- Zone-level detailed analytics with historical tracking
- Feature importance (Random Forest) visualization
- SHAP values for model explainability

**Medium-term (3-6 months):**
- Real-time data ingestion from BMS
- Automated alerts and notifications
- A/B testing framework for AI recommendations
- Multi-building portfolio optimization with constraints

**Long-term (6-12 months):**
- Integration with Building Management Systems (BMS)
- Automated setpoint control (with human override)
- Mobile app for facility managers
- Advanced scheduling and forecasting

### 13.2 Performance Optimizations (Future)

**P5: Vectorize Analytics Scoring Loop** (8-15ms savings)
- Replace Python loops with NumPy operations
- Batch compute comfort scores across all zones

**P6: Cache Analytics Results by Model** (50-100ms savings)
- Implement `_ANALYTICS_CACHE[(model, building)]`
- Invalidate on portfolio data change

**P7: Implement Result Memoization**
- Cache frequently-called helper functions
- LRU cache for prediction results

### 13.3 Architectural Improvements

**Database Integration:**
- PostgreSQL or MongoDB for persistent storage
- Historical data tracking
- User preferences and settings

**API Layer:**
- RESTful API for external integrations
- WebSocket for real-time updates
- Authentication and authorization

**Microservices:**
- Separate prediction service
- Dedicated optimization engine
- Independent data ingestion pipeline

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **HVAC** | Heating, Ventilation, and Air Conditioning |
| **BMS** | Building Management System |
| **Setpoint** | Target temperature for thermostat |
| **Comfort Range** | Temperature range considered comfortable (21-23°C) |
| **Baseline** | Traditional HVAC strategy (22°C for all zones) |
| **Smart Win-Win** | Optimization that only accepts genuine improvements |
| **Digital Twin** | Virtual model for "what-if" scenario testing |
| **Penalty-Based** | Optimization using cost function with comfort penalties |
| **Multi-Output** | ML model predicting multiple targets (temp + energy) |
| **Batch Prediction** | Predicting multiple samples in single API call |
| **Carbon Intensity** | g CO₂ per kWh of electricity |
| **PUE** | Power Usage Effectiveness (datacenter efficiency) |

---

## 15. References

### 15.1 ML and Sustainability Methodologies

- [ML CO2 Impact Calculator](https://mlco2.github.io/impact/) - Training emissions
- [CodeCarbon](https://codecarbon.io/) - Real-time carbon tracking
- [Cloud Carbon Footprint](https://www.cloudcarbonfootprint.org/) - Infrastructure emissions
- [Green Algorithms](https://www.green-algorithms.org/) - Research computing carbon
- [ASHRAE 90.1](https://www.ashrae.org/technical-resources/bookstore/standard-90-1) - Building energy standards
- [GHG Protocol](https://ghgprotocol.org/) - Corporate carbon accounting

### 15.2 Technical Documentation

- [Plotly Dash Documentation](https://dash.plotly.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

**Document Version:** 1.0
**Maintained By:** ComfortRoom Development Team
**Last Review:** 2026-01-06
**Next Review:** 2026-04-06
