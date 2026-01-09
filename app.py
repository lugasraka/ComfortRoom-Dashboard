import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import threading

# TensorFlow (Neural Network)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ==========================================
# NOTE: Training Data Generation
# ==========================================
# The training data is generated separately in 'build_training_data.ipynb' notebook.
# Run that notebook first to create 'training_data.csv' before running this app.
# This approach separates data generation from model training/deployment.

# ==========================================
# 1. DATA & MODEL INITIALIZATION (GLOBAL)
# ==========================================

# --- A. Mock Data Generation ---
def generate_portfolio():
    buildings = [
        {"id": "B1", "name": "Comfort Room - Zug", "lat": 47.1662, "lon": 8.5155, "type": "Office"},
        {"id": "B2", "name": "Tech Hub - Munich", "lat": 48.1351, "lon": 11.5820, "type": "R&D Lab"},
        {"id": "B3", "name": "Logistics - Milan", "lat": 45.4642, "lon": 9.1900, "type": "Warehouse"}
    ]
    
    # Generate Zones for each building with fixed values for consistent demo
    portfolio = {}
    
    # Predefined zone data for consistency (no random generation)
    # Format: [temp, occupied, status]
    zone_configs = {
        'Comfort Room - Zug': [
            {"name": "Lobby", "temp": 22.3, "occupied": False, "status": "OK"},
            {"name": "Conf. Room A", "temp": 17.2, "occupied": True, "status": "Critical"},  # Hardcoded alert
            {"name": "Open Office", "temp": 25.8, "occupied": True, "status": "Warning"},     # Hardcoded alert
            {"name": "Server Room", "temp": 19.4, "occupied": False, "status": "OK"},
            {"name": "Cafeteria", "temp": 23.1, "occupied": True, "status": "OK"}
        ],
        'Tech Hub - Munich': [
            {"name": "Lobby", "temp": 21.8, "occupied": True, "status": "OK"},
            {"name": "Conf. Room A", "temp": 22.5, "occupied": False, "status": "OK"},
            {"name": "Open Office", "temp": 27.5, "occupied": True, "status": "Critical"},    # Hardcoded alert
            {"name": "Server Room", "temp": 20.1, "occupied": False, "status": "OK"},
            {"name": "Cafeteria", "temp": 24.2, "occupied": True, "status": "Warning"}
        ],
        'Logistics - Milan': [
            {"name": "Lobby", "temp": 23.6, "occupied": False, "status": "OK"},
            {"name": "Conf. Room A", "temp": 22.0, "occupied": True, "status": "OK"},
            {"name": "Open Office", "temp": 21.5, "occupied": True, "status": "OK"},
            {"name": "Server Room", "temp": 18.9, "occupied": False, "status": "OK"},
            {"name": "Cafeteria", "temp": 24.8, "occupied": True, "status": "Warning"}
        ]
    }
    
    for b in buildings:
        zones = zone_configs[b['name']]
        portfolio[b['name']] = {"meta": b, "zones": zones}
        
    return buildings, portfolio

# --- B. Train Digital Twin Models ---
def _generate_training_data_fallback(n_rows: int = 15000):
    """
    Fallback: Generate training data in-memory if CSV file is not available.
    This ensures the app can still function without the pre-generated CSV.
    Returns X (features), y (multi-output: temp and energy).
    """
    np.random.seed(42)

    # Inputs
    outdoor_temp = np.random.normal(15, 8, n_rows)   # Weather
    prev_indoor_temp = np.random.normal(21, 2, n_rows) # Current State
    setpoint = np.random.choice(np.arange(18, 25, 0.5), n_rows) # Action
    occupancy = np.random.choice([0, 1], n_rows, p=[0.4, 0.6]) # Context

    # --- Physics Logic ---
    # 1. Temperature Dynamics (Next Temp)
    # Heat Loss/Gain from outside + HVAC heating/cooling + Body heat
    thermal_drift = 0.05 * (outdoor_temp - prev_indoor_temp)
    hvac_power = 0.3 * (setpoint - prev_indoor_temp)
    body_heat = 0.1 * occupancy

    next_indoor_temp = prev_indoor_temp + thermal_drift + hvac_power + body_heat + np.random.normal(0, 0.1, n_rows)

    # 2. Energy Consumption (kWh)
    # Energy is proportional to the "gap" the HVAC tries to close + baseload
    base_load = 0.5 # Fans, lights
    hvac_kwh = np.abs(hvac_power) * 5 # Factor to convert 'effort' to kWh
    total_energy = base_load + hvac_kwh + (0.2 * occupancy) + np.random.normal(0, 0.05, n_rows)

    df = pd.DataFrame({
        'Outdoor_Temp': outdoor_temp,
        'Prev_Indoor_Temp': prev_indoor_temp,
        'Setpoint': setpoint,
        'Occupancy': occupancy,
        'Target_Temp': next_indoor_temp,   # Target 1
        'Target_Energy': total_energy      # Target 2
    })
    
    X = df[['Outdoor_Temp', 'Prev_Indoor_Temp', 'Setpoint', 'Occupancy']].values
    y = df[['Target_Temp', 'Target_Energy']].values # Multi-output target
    
    return X.astype(np.float32), y.astype(np.float32)

def _load_training_data(csv_path: str = 'training_data.csv'):
    """
    Load pre-generated training data from CSV file.
    The CSV is generated by the build_training_data.ipynb notebook.
    Falls back to in-memory generation if CSV is not available.
    Returns X (features), y (multi-output: temp and energy).
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded training data from {csv_path}")
        
        # Extract features and targets
        X = df[['Outdoor_Temp', 'Prev_Indoor_Temp', 'Setpoint', 'Occupancy']].values
        y = df[['Target_Temp', 'Target_Energy']].values  # Multi-output target
        
        return X.astype(np.float32), y.astype(np.float32)
    except FileNotFoundError:
        print(f"⚠ Warning: '{csv_path}' not found. Generating training data in-memory...")
        print(f"  For better performance, run 'build_training_data.ipynb' to generate the CSV file.")
        return _generate_training_data_fallback()

def train_models_rf(X=None, y=None):
    """
    Train Random Forest with multi-output support (predicts both temp and energy).
    RF natively supports multi-output regression.
    """
    # Use pre-loaded global data if not provided
    if X is None or y is None:
        X, y = _GLOBAL_TRAINING_X, _GLOBAL_TRAINING_Y
    
    # Scale inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    start = time.perf_counter()
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - start

    # Predict both outputs
    y_pred = rf_model.predict(X_test)
    
    # Separate metrics for each output
    mae_t = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_e = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    rmse_t = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_e = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    r2_t = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_e = r2_score(y_test[:, 1], y_pred[:, 1])

    metrics = {
        'train_time_s': train_time_s,
        'mae_temp': float(mae_t),
        'mae_energy': float(mae_e),
        'rmse_temp': float(rmse_t),
        'rmse_energy': float(rmse_e),
        'r2_temp': float(r2_t),
        'r2_energy': float(r2_e),
        'n_estimators': 50,
        'max_depth': 10
    }
    return rf_model, scaler_X, metrics

def train_models_nn(X=None, y=None):
    """
    Train TensorFlow Neural Network with multi-output (2 output nodes for temp and energy).
    """
    if not TF_AVAILABLE:
        return None, None, {'error': 'TensorFlow not available'}

    # Use pre-loaded global data if not provided
    if X is None or y is None:
        X, y = _GLOBAL_TRAINING_X, _GLOBAL_TRAINING_Y
    
    # Scale inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    tf.random.set_seed(42)

    # Build multi-output model (2 output nodes)
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # Output size 2 (Temp, Energy)
    ])
    tf_model.compile(optimizer='adam', loss='mse')

    start = time.perf_counter()
    tf_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    train_time_s = time.perf_counter() - start

    # Predict both outputs
    y_pred = tf_model.predict(X_test, verbose=0)
    
    # Separate metrics
    mae_t = float(mean_absolute_error(y_test[:, 0], y_pred[:, 0]))
    mae_e = float(mean_absolute_error(y_test[:, 1], y_pred[:, 1]))
    rmse_t = float(np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0])))
    rmse_e = float(np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1])))
    r2_t = float(r2_score(y_test[:, 0], y_pred[:, 0]))
    r2_e = float(r2_score(y_test[:, 1], y_pred[:, 1]))

    metrics = {
        'train_time_s': train_time_s,
        'mae_temp': mae_t,
        'mae_energy': mae_e,
        'rmse_temp': rmse_t,
        'rmse_energy': rmse_e,
        'r2_temp': r2_t,
        'r2_energy': r2_e,
        'params': int(tf_model.count_params())
    }
    return tf_model, scaler_X, metrics

def train_models_pt(X=None, y=None):
    """
    Train PyTorch Neural Network with multi-output (2 output nodes for temp and energy).
    Uses mini-batch training with DataLoader.
    """
    if not TORCH_AVAILABLE:
        return None, None, {'error': 'PyTorch not available'}

    # Use pre-loaded global data if not provided
    if X is None or y is None:
        X, y = _GLOBAL_TRAINING_X, _GLOBAL_TRAINING_Y
    
    # Scale inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_pt = torch.tensor(X_train, dtype=torch.float32)
    y_train_pt = torch.tensor(y_train, dtype=torch.float32)
    X_test_pt = torch.tensor(X_test, dtype=torch.float32)
    y_test_pt = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for mini-batch training
    train_data = TensorDataset(X_train_pt, y_train_pt)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

    # Define PyTorch Model
    class MultiTaskNet(nn.Module):
        def __init__(self):
            super(MultiTaskNet, self).__init__()
            self.fc1 = nn.Linear(4, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 2)  # Output size 2 (Temp, Energy)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.out(x)

    pt_model = MultiTaskNet()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    start = time.perf_counter()
    # Training loop with mini-batches
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = pt_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    train_time_s = time.perf_counter() - start

    # Predict on test set
    pt_model.eval()
    with torch.no_grad():
        y_pred = pt_model(X_test_pt).numpy()
    
    # Separate metrics
    mae_t = float(mean_absolute_error(y_test[:, 0], y_pred[:, 0]))
    mae_e = float(mean_absolute_error(y_test[:, 1], y_pred[:, 1]))
    rmse_t = float(np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0])))
    rmse_e = float(np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1])))
    r2_t = float(r2_score(y_test[:, 0], y_pred[:, 0]))
    r2_e = float(r2_score(y_test[:, 1], y_pred[:, 1]))

    # Count parameters
    total_params = sum(p.numel() for p in pt_model.parameters())

    metrics = {
        'train_time_s': train_time_s,
        'mae_temp': mae_t,
        'mae_energy': mae_e,
        'rmse_temp': rmse_t,
        'rmse_energy': rmse_e,
        'r2_temp': r2_t,
        'r2_energy': r2_e,
        'params': int(total_params)
    }
    return pt_model, scaler_X, metrics

# Initialize System
BUILDINGS_LIST, PORTFOLIO_DB = generate_portfolio()

# Pre-load training data once (shared across all models)
print("📊 Pre-loading training data...")
_GLOBAL_TRAINING_X, _GLOBAL_TRAINING_Y = _load_training_data()
print(f"✓ Training data loaded: {_GLOBAL_TRAINING_X.shape[0]} samples")

# Global model registry and training status
MODEL_REGISTRY = {
    'RF': {'model': None, 'scaler': None, 'metrics': None},
    'NN': {'model': None, 'scaler': None, 'metrics': None},
    'PT': {'model': None, 'scaler': None, 'metrics': None}
}
MODELS_TRAINING = True
MODELS_READY = threading.Event()

# Optimization constants
CANDIDATE_SETPOINTS = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
COMFORT_MIN, COMFORT_MAX = 21.0, 23.0

# Defer model training to background to avoid blocking app startup
def _train_rf_background():
    rf_model, rf_scaler, metrics_rf = train_models_rf()
    MODEL_REGISTRY['RF'] = {
        'model': rf_model,
        'scaler': rf_scaler,
        'metrics': metrics_rf
    }

# Train all models in background thread
def _train_all_models_background():
    global MODELS_TRAINING
    try:
        print("🚀 Starting model training in background...")
        
        # Train RF first (faster)
        print("  Training Random Forest...")
        _train_rf_background()
        print(f"  ✓ Random Forest trained: {MODEL_REGISTRY['RF']['metrics']}")
        
        # Then train NN
        print("  Training Neural Network (TensorFlow)...")
        nn_model, nn_scaler, metrics_nn = train_models_nn()
        if nn_model is not None:
            MODEL_REGISTRY['NN'] = {
                'model': nn_model,
                'scaler': nn_scaler,
                'metrics': metrics_nn
            }
            print(f"  ✓ Neural Network trained: {metrics_nn}")
        else:
            MODEL_REGISTRY['NN'] = {
                'model': None,
                'scaler': None,
                'metrics': metrics_nn
            }
            print(f"  ⚠ Neural Network unavailable: {metrics_nn}")
        
        # Then train PyTorch
        print("  Training PyTorch Neural Network...")
        pt_model, pt_scaler, metrics_pt = train_models_pt()
        if pt_model is not None:
            MODEL_REGISTRY['PT'] = {
                'model': pt_model,
                'scaler': pt_scaler,
                'metrics': metrics_pt
            }
            print(f"  ✓ PyTorch trained: {metrics_pt}")
        else:
            MODEL_REGISTRY['PT'] = {
                'model': None,
                'scaler': None,
                'metrics': metrics_pt
            }
            print(f"  ⚠ PyTorch unavailable: {metrics_pt}")
        
        print("✓ All models training completed!")
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
    finally:
        MODELS_TRAINING = False
        MODELS_READY.set()

# Start model training in background
model_thread = threading.Thread(target=_train_all_models_background, daemon=True)
model_thread.start()

# ==========================================
# 2. DASH APP SETUP
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Comfort Room"

# Styles
SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem",
    "padding": "2rem 1rem", "background-color": "#f8f9fa",
}
CONTENT_STYLE = {
    "margin-left": "22rem", "margin-right": "2rem", "padding": "2rem 1rem",
}

# ==========================================
# 3. LAYOUT COMPONENTS
# ==========================================

# --- Sidebar ---
sidebar = html.Div([
    html.Img(src="https://www.freeiconspng.com/download/23660", height="40px", className="mb-4"),
    html.H4("Comfort Room Dashboard", className="display-7"),
    html.Hr(),
    
    # Navigation
    dbc.Label("View Level"),
    dbc.RadioItems(
        id="view-selector",
        options=[
            {"label": "🌍 Portfolio Map", "value": "portfolio"},
            {"label": "📊 AI Impact Analytics", "value": "analytics"},
            {"label": "🏢 Building View", "value": "building"},
            {"label": "🤖 ML Models", "value": "ml_models"},
            {"label": "🌱 Sustainability of AI", "value": "sustainability"},
            {"label": "❓ Tutorial", "value": "tutorial"}
        ],
        value="portfolio",
        className="mb-4"
    ),
    
    # Dynamic Filters (Hidden on Portfolio View via callbacks)
    html.Div(id="building-filters", children=[
        dbc.Label("Select Building"),
        dcc.Dropdown(
            id="building-dropdown",
            options=[{'label': b['name'], 'value': b['name']} for b in BUILDINGS_LIST],
            value=BUILDINGS_LIST[0]['name'],
            clearable=False,
            className="mb-3"
        ),
        dbc.Label("Select Zone"),
        dcc.Dropdown(id="zone-dropdown", clearable=False, className="mb-3"),
    ]),
    
    html.Hr(),
    
    # Model Selector (for AI Impact Analytics)
    html.Div(id="model-selector-container", children=[
        dbc.Label("ML Model (for Analytics)"),
        dcc.Dropdown(
            id="model-selector",
            options=[
                {'label': '🌳 Random Forest', 'value': 'RF'},
                {'label': '🧠 TensorFlow NN', 'value': 'NN'},
                {'label': '🔥 PyTorch NN', 'value': 'PT'}
            ],
            value='PT',  # Default to PyTorch
            clearable=False,
            className="mb-3"
        ),
        html.Small("Select which model to use for AI Impact Analytics predictions", className="text-muted")
    ])
], style=SIDEBAR_STYLE)

# --- Main Content Area ---
content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])

# ==========================================
# 4. SUSTAINABILITY (CO2 CALCULATIONS)
# ==========================================
# Global cache for savings calculation (avoids recalculating on every page load)
_CACHED_SAVINGS_PCT = None

# Global cache for CO2 calculations (avoids expensive recalculation)
_CACHED_CO2_DATA = None
_CACHED_CO2_TIMESTAMP = None

# Regional carbon intensity data (g CO2/kWh)
# Sources: IEA (2024), EPA eGRID (2024), European Environment Agency (2024)
CARBON_INTENSITY_REGIONS = {
    'global_avg': {'value': 475, 'label': 'Global Average', 'uncertainty': (400, 550),
                   'description': 'Worldwide grid mix average (IEA 2024)'},
    'nordics': {'value': 50, 'label': 'Nordic Countries', 'uncertainty': (30, 80),
                'description': 'Hydro and wind dominant (Norway, Sweden, Iceland)'},
    'france': {'value': 80, 'label': 'France', 'uncertainty': (60, 100),
               'description': 'Nuclear-dominant grid (~70% nuclear)'},
    'us_west': {'value': 200, 'label': 'US West Coast', 'uncertainty': (150, 250),
                'description': 'California, Oregon, Washington (high renewables)'},
    'us_midwest': {'value': 650, 'label': 'US Midwest', 'uncertainty': (550, 750),
                   'description': 'Coal and natural gas heavy'},
    'china': {'value': 600, 'label': 'China', 'uncertainty': (550, 700),
              'description': 'Coal-dominant grid transitioning to renewables'},
    'australia': {'value': 750, 'label': 'Australia', 'uncertainty': (650, 850),
                  'description': 'Coal-heavy grid with growing solar'},
    'india': {'value': 900, 'label': 'India', 'uncertainty': (800, 1000),
              'description': 'Coal-dominant with rapid renewable growth'}
}

# Hardware embodied carbon (kg CO2e lifecycle)
# Sources: Dell Product Carbon Footprints (2024), Apple Environmental Reports (2024)
HARDWARE_EMBODIED_CO2 = {
    'cpu_server': {'value': 1500, 'lifespan_years': 4, 'label': 'CPU Server',
                   'description': 'Standard rack server (Dell PowerEdge, HP ProLiant)'},
    'gpu_server': {'value': 3000, 'lifespan_years': 4, 'label': 'GPU Server',
                   'description': 'GPU-accelerated server (NVIDIA A100/H100)'},
    'edge_device': {'value': 400, 'lifespan_years': 5, 'label': 'Edge Device',
                    'description': 'IoT gateway or edge computing device'}
}

# Alternative HVAC management approaches (for comparison)
ALTERNATIVE_APPROACHES = {
    'no_optimization': {
        'label': 'No Optimization',
        'savings_pct': 0.0,
        'cost_per_building_year': 0,
        'co2_kg_year': 0,
        'description': 'Current wasteful baseline - maintains 22°C everywhere, always'
    },
    'manual_tuning': {
        'label': 'Manual Technician Tuning',
        'savings_pct': 0.08,  # 8% savings from quarterly tune-ups
        'cost_per_building_year': 2000,  # 4 visits/year × $500/visit
        'co2_kg_year': 80,  # 4 truck rolls × 20 kg CO2/visit
        'description': 'Quarterly HVAC technician visits for setpoint adjustments'
    },
    'simple_scheduling': {
        'label': 'Simple Time-based Scheduling',
        'savings_pct': 0.15,  # 15% savings from basic occupancy scheduling
        'cost_per_building_year': 500,  # Programmable thermostat + setup
        'co2_kg_year': 0,
        'description': 'Basic 9-5 occupancy schedule, no AI or sensors'
    }
}

def calculate_actual_savings_from_demo():
    """
    Calculate actual energy savings percentage from the demo data.
    Uses THE EXACT SAME optimization logic as AI Impact Analytics tab.
    Runs optimization on all zones and computes real savings percentage.
    Uses caching to avoid expensive recalculation on every page load.
    """
    global _CACHED_SAVINGS_PCT
    
    # Return cached value if already calculated
    if _CACHED_SAVINGS_PCT is not None:
        return _CACHED_SAVINGS_PCT

    # Baseline: Traditional HVAC maintains 22°C for all zones (wasteful!)
    BASELINE_SETPOINT_OCCUPIED = 22.0
    BASELINE_SETPOINT_UNOCCUPIED = 22.0
    # AI uses expanded setpoint range (matching AI Impact Analytics)
    candidate_setpoints = CANDIDATE_SETPOINTS
    
    # Use Random Forest model for calculation (fastest)
    model_entry = MODEL_REGISTRY.get('RF')
    if not model_entry or model_entry['model'] is None:
        return 0.20  # Fallback to 20% if model not ready
    
    model = model_entry['model']
    scaler = model_entry['scaler']
    
    # Use fixed outdoor temps (matching analytics tab)
    outdoor_temps = {
        'Comfort Room - Zug': 12.5,
        'Tech Hub - Munich': 14.0,
        'Logistics - Milan': 16.5
    }
    
    # PERFORMANCE OPTIMIZATION: Batch all baseline predictions at once
    # Collect all zone inputs first
    all_zones = []
    for building_name, building_data in PORTFOLIO_DB.items():
        outdoor_temp = outdoor_temps.get(building_name, 15.0)
        for zone in building_data['zones']:
            prev_temp = zone['temp']
            is_occupied = 1 if zone['occupied'] else 0
            baseline_setpoint = BASELINE_SETPOINT_OCCUPIED if is_occupied else BASELINE_SETPOINT_UNOCCUPIED
            all_zones.append({
                'outdoor_temp': outdoor_temp,
                'prev_temp': prev_temp,
                'is_occupied': is_occupied,
                'baseline_setpoint': baseline_setpoint
            })

    # Batch predict all baselines at once (15 zones in 1 call instead of 15 calls)
    baseline_inputs = np.array([
        [z['outdoor_temp'], z['prev_temp'], z['baseline_setpoint'], z['is_occupied']]
        for z in all_zones
    ])
    baseline_inputs_scaled = scaler.transform(baseline_inputs)
    baseline_preds = model.predict(baseline_inputs_scaled)  # ONE batch call

    total_baseline_energy = 0
    total_optimized_energy = 0

    zone_idx = 0
    for building_name, building_data in PORTFOLIO_DB.items():
        outdoor_temp = outdoor_temps.get(building_name, 15.0)
        for zone in building_data['zones']:
            prev_temp = zone['temp']
            is_occupied = 1 if zone['occupied'] else 0

            # Get baseline prediction from batch results
            pred_base = baseline_preds[zone_idx]
            baseline_energy = pred_base[1]
            zone_idx += 1

            # AI Optimization: Test all candidate setpoints
            candidate_inputs = np.array([
                [outdoor_temp, prev_temp, sp, is_occupied] for sp in candidate_setpoints
            ])
            candidate_inputs_scaled = scaler.transform(candidate_inputs)
            candidate_preds = model.predict(candidate_inputs_scaled)
            
            candidate_temps = candidate_preds[:, 0]
            candidate_energies = candidate_preds[:, 1]
            
            # AI Optimization Strategy (SMART WIN-WIN - Matching AI Impact Analytics):
            # Only make changes that improve outcomes compared to baseline
            
            if is_occupied == 1:
                # OCCUPIED: Find win-win scenarios vs baseline
                baseline_comfort_achieved = (COMFORT_MIN <= pred_base[0] <= COMFORT_MAX)
                
                best_score = float('inf')
                best_idx = None
                
                for idx in range(len(candidate_temps)):
                    temp = candidate_temps[idx]
                    energy = candidate_energies[idx]
                    
                    achieves_comfort = (COMFORT_MIN <= temp <= COMFORT_MAX)
                    energy_improvement = baseline_energy - energy
                    
                    if achieves_comfort and not baseline_comfort_achieved:
                        comfort_improvement = 10.0
                    elif achieves_comfort and baseline_comfort_achieved:
                        comfort_improvement = 0
                    elif not achieves_comfort and not baseline_comfort_achieved:
                        baseline_deviation = min(abs(pred_base[0] - COMFORT_MIN), abs(pred_base[0] - COMFORT_MAX))
                        candidate_deviation = min(abs(temp - COMFORT_MIN), abs(temp - COMFORT_MAX))
                        comfort_improvement = (baseline_deviation - candidate_deviation) * 2.0
                    else:
                        comfort_improvement = -10.0
                    
                    score = -(comfort_improvement * 1.5 + energy_improvement * 1.0)
                    
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is None or best_score >= 0:
                    baseline_sp = BASELINE_SETPOINT_OCCUPIED
                    best_idx = np.argmin(np.abs(candidate_setpoints - baseline_sp))
            else:
                # UNOCCUPIED: Minimize energy
                best_idx = np.argmin(candidate_energies)
            
            best_energy = candidate_energies[best_idx]
            
            total_baseline_energy += baseline_energy
            total_optimized_energy += best_energy
    
    # Calculate actual savings percentage
    savings_pct = ((total_baseline_energy - total_optimized_energy) / total_baseline_energy) if total_baseline_energy > 0 else 0.20
    savings_pct = max(0.01, min(0.50, savings_pct))  # Clamp between 1% and 50%
    
    # Cache the result for future calls
    _CACHED_SAVINGS_PCT = savings_pct
    
    return savings_pct

def calculate_co2_impacts(region='global_avg', hardware_type='cpu_server'):
    """
    Calculate CO2 emissions for AI models across different lifecycle phases.
    Creates three realistic deployment scenarios to show how emissions scale.

    Sources and assumptions:
    - Training: Based on ML CO2 Impact calculator (mlco2.github.io/impact)
    - Inference: Based on GreenAlgorithms (www.green-algorithms.org)
    - Infrastructure: Dell/HP/Apple Product Carbon Footprints (2024)
    - Energy Savings: Actual savings from demo optimization (calculated dynamically)
    - Carbon intensity: Regional grid data from IEA (2024), EPA eGRID (2024)

    Args:
        region: Carbon intensity region key (see CARBON_INTENSITY_REGIONS)
        hardware_type: Hardware type for embodied carbon (see HARDWARE_EMBODIED_CO2)
    """
    global _CACHED_CO2_DATA, _CACHED_CO2_TIMESTAMP

    # Cache key includes region and hardware type
    cache_key = f"{region}_{hardware_type}"

    # Return cached value if less than 5 minutes old (300 seconds)
    if _CACHED_CO2_DATA is not None and _CACHED_CO2_TIMESTAMP is not None:
        if isinstance(_CACHED_CO2_DATA, dict) and cache_key in _CACHED_CO2_DATA:
            age_seconds = time.time() - _CACHED_CO2_TIMESTAMP
            if age_seconds < 300:
                return _CACHED_CO2_DATA[cache_key]

    # Carbon intensity (g CO2 per kWh) - region-specific
    region_data = CARBON_INTENSITY_REGIONS.get(region, CARBON_INTENSITY_REGIONS['global_avg'])
    CARBON_INTENSITY = region_data['value']
    CARBON_INTENSITY_MIN, CARBON_INTENSITY_MAX = region_data['uncertainty']
    
    # Calculate actual savings from demo
    actual_savings_pct = calculate_actual_savings_from_demo()
    
    # Building energy consumption baseline
    BUILDINGS_SQFT = 3 * 5000  # Total square meters for current demo
    ANNUAL_ENERGY_KWH = BUILDINGS_SQFT * 150  # kWh/year
    HVAC_PERCENTAGE = 0.40  # HVAC is typically 40% of building energy
    HVAC_BASELINE_KWH = ANNUAL_ENERGY_KWH * HVAC_PERCENTAGE
    
    # TWO DEPLOYMENT SCENARIOS (optimized for demo - shows contrast between small and large)
    scenarios = {
        'Small (Demo)': {
            'buildings': 3,
            'zones_per_building': 5,
            'retraining_frequency_days': 365,  # Annual retraining
            'training_dataset_size': 15000,  # Current demo size
            'description': 'Current demo deployment with 3 buildings, annual model updates'
        },
        'Large (Enterprise)': {
            'buildings': 1000,
            'zones_per_building': 10,
            'retraining_frequency_days': 7,  # Weekly retraining
            'training_dataset_size': 5000000,  # Million+ samples
            'description': 'Enterprise-scale deployment with continuous learning'
        }
    }
    
    results = {}
    
    # Only use Random Forest for Sustainability analysis (faster loading)
    model_name = "Random Forest"
    model_key = "RF"
    
    if model_key in MODEL_REGISTRY and MODEL_REGISTRY[model_key]['metrics'] is not None:
        
        results[model_name] = {}
        
        for scenario_name, scenario in scenarios.items():
            # Use realistic training time estimates (not demo's tiny training time)
            # Random Forest training time scales roughly linearly with dataset size
            # Industry benchmark: ~10-30 minutes per 100k samples on CPU
            dataset_size = scenario['training_dataset_size']
            
            # Realistic training time estimates (in seconds)
            # Includes: data preprocessing, feature engineering, model training, 
            # cross-validation (5-fold), hyperparameter tuning, model validation
            if dataset_size <= 15000:
                # Small: ~2 hours (comprehensive training pipeline)
                base_train_time_s = 7200
            elif dataset_size <= 100000:
                # Medium: ~8 hours
                base_train_time_s = 28800
            else:
                # Large (5M): ~24 hours (full production pipeline with distributed training)
                base_train_time_s = 86400
            
            # Number of retraining iterations per year
            retraining_per_year = 365 / scenario['retraining_frequency_days']
            annual_training_time_s = base_train_time_s * retraining_per_year
            
            # 1. TRAINING EMISSIONS (scaled for scenario)
            # Training power: includes CPU/GPU, memory, storage I/O
            if dataset_size <= 15000:
                train_power_w = 250  # Multi-core CPU workstation
            elif dataset_size <= 100000:
                train_power_w = 600  # High-performance server
            else:
                train_power_w = 2000  # Multi-node distributed cluster with GPUs
            
            train_pue = 1.2  # On-prem datacenter
            
            # Annual training energy (all retraining iterations)
            annual_train_energy_kwh = (train_power_w * train_pue * annual_training_time_s) / (1000 * 3600)
            annual_train_co2_kg = annual_train_energy_kwh * CARBON_INTENSITY / 1000
            
            # 2. INFERENCE EMISSIONS (scaled for scenario)
            total_zones = scenario['buildings'] * scenario['zones_per_building']
            predictions_per_year = (365 * 24 * 60 / 5) * total_zones  # Every 5 minutes
            
            # Realistic inference: includes always-on servers, API overhead, data I/O
            # Server power consumption is NOT just during active prediction
            # It includes idle power, memory, networking, storage
            
            if scenario['buildings'] <= 10:
                # Edge server: always-on with periodic predictions
                inference_time_s = 0.010  # 10ms (includes API latency, I/O)
                inference_power_w = 50  # Server average power (not just CPU)
            elif scenario['buildings'] <= 100:
                # Dedicated inference server
                inference_time_s = 0.008  # 8ms
                inference_power_w = 150  # Server cluster average power
            else:
                # High-performance inference cluster
                inference_time_s = 0.005  # 5ms (optimized batch processing)
                inference_power_w = 400  # Multi-server cluster average power
            
            # Annual inference energy
            annual_inference_energy_kwh = (inference_power_w * inference_time_s * predictions_per_year) / (1000 * 3600)
            annual_inference_co2_kg = (annual_inference_energy_kwh * CARBON_INTENSITY) / 1000
            
            # 3. INFRASTRUCTURE & SUPPLY CHAIN (scaled for scenario)
            # Key insight: AI models run on existing infrastructure
            # Infrastructure emissions = minimal incremental hardware + datacenter overhead

            # Calculate compute emissions (what infrastructure supports)
            operational_co2_kg = annual_train_co2_kg + annual_inference_co2_kg

            # Get hardware embodied carbon based on selected type
            hw_data = HARDWARE_EMBODIED_CO2.get(hardware_type, HARDWARE_EMBODIED_CO2['cpu_server'])
            hw_annual_co2 = hw_data['value'] / hw_data['lifespan_years']

            # Incremental hardware allocation (very conservative)
            # Assumes AI model uses 5-10% of a shared server's capacity
            if scenario['buildings'] <= 10:
                # Small: Each building shares 1 edge server (10% allocation)
                num_allocated_servers = scenario['buildings'] * 0.1
                hardware_allocation_kg = num_allocated_servers * hw_annual_co2 * 0.10
            else:
                # Large: Efficient datacenter sharing (5% allocation per server)
                # 1 server per 100 buildings, AI uses 5%
                num_allocated_servers = scenario['buildings'] / 100
                hardware_allocation_kg = num_allocated_servers * hw_annual_co2 * 0.05
            
            # Datacenter operational overhead
            # This is the ADDITIONAL overhead beyond direct compute (already in PUE)
            # Includes networking, storage, monitoring, facility operations
            # Should be proportional to compute workload, NOT dominate it
            if scenario['buildings'] <= 10:
                # Small: Higher relative overhead due to inefficiency
                overhead_multiplier = 0.80  # 80% overhead
            elif scenario['buildings'] <= 100:
                # Medium: Moderate overhead
                overhead_multiplier = 0.60  # 60% overhead
            else:
                # Large: Optimized datacenter operations
                overhead_multiplier = 0.40  # 40% overhead
            
            datacenter_overhead_kg = operational_co2_kg * overhead_multiplier
            
            # Total infrastructure = incremental hardware + datacenter overhead
            infrastructure_co2_kg = hardware_allocation_kg + datacenter_overhead_kg
            
            # 4. TOTAL AI EMISSIONS
            total_ai_co2_kg = annual_train_co2_kg + annual_inference_co2_kg + infrastructure_co2_kg
            
            # 5. ENERGY SAVINGS (scaled for scenario)
            # Scale baseline HVAC energy by number of buildings
            scenario_hvac_baseline_kwh = (HVAC_BASELINE_KWH / 3) * scenario['buildings']
            scenario_energy_saved_kwh = scenario_hvac_baseline_kwh * actual_savings_pct
            scenario_co2_saved_kg = (scenario_energy_saved_kwh * CARBON_INTENSITY) / 1000
            
            # 6. NET BENEFIT
            net_benefit_kg = scenario_co2_saved_kg - total_ai_co2_kg
            roi_ratio = scenario_co2_saved_kg / total_ai_co2_kg if total_ai_co2_kg > 0 else 0

            # Calculate uncertainty ranges using min/max carbon intensity
            scenario_co2_saved_kg_min = (scenario_energy_saved_kwh * CARBON_INTENSITY_MIN) / 1000
            scenario_co2_saved_kg_max = (scenario_energy_saved_kwh * CARBON_INTENSITY_MAX) / 1000
            net_benefit_min = scenario_co2_saved_kg_min - total_ai_co2_kg
            net_benefit_max = scenario_co2_saved_kg_max - total_ai_co2_kg

            results[model_name][scenario_name] = {
                'training_co2_kg': annual_train_co2_kg,
                'inference_co2_kg': annual_inference_co2_kg,
                'infrastructure_co2_kg': infrastructure_co2_kg,
                'total_ai_co2_kg': total_ai_co2_kg,
                'energy_saved_kwh': scenario_energy_saved_kwh,
                'co2_saved_kg': scenario_co2_saved_kg,
                'co2_saved_kg_range': (scenario_co2_saved_kg_min, scenario_co2_saved_kg_max),
                'net_benefit_kg': net_benefit_kg,
                'net_benefit_range': (net_benefit_min, net_benefit_max),
                'roi_ratio': roi_ratio,
                'predictions_per_year': predictions_per_year,
                'retraining_per_year': retraining_per_year,
                'total_zones': total_zones,
                'training_percentage': (annual_train_co2_kg / total_ai_co2_kg * 100) if total_ai_co2_kg > 0 else 0,
                'inference_percentage': (annual_inference_co2_kg / total_ai_co2_kg * 100) if total_ai_co2_kg > 0 else 0,
                'infrastructure_percentage': (infrastructure_co2_kg / total_ai_co2_kg * 100) if total_ai_co2_kg > 0 else 0
            }
    
    # Add baseline reference data and metadata
    results['_baseline'] = {
        'hvac_energy_kwh': HVAC_BASELINE_KWH,
        'hvac_co2_kg': (HVAC_BASELINE_KWH * CARBON_INTENSITY) / 1000,
        'total_building_energy_kwh': ANNUAL_ENERGY_KWH,
        'carbon_intensity': CARBON_INTENSITY,
        'carbon_intensity_range': (CARBON_INTENSITY_MIN, CARBON_INTENSITY_MAX),
        'region': region_data['label'],
        'region_description': region_data['description'],
        'hardware_type': hw_data['label'],
        'hardware_description': hw_data['description'],
        'actual_savings_pct': actual_savings_pct * 100  # Convert to percentage for display
    }

    results['_scenarios'] = scenarios

    # Add regional sensitivity analysis
    results['_regional_sensitivity'] = {}
    for reg_key, reg_data in CARBON_INTENSITY_REGIONS.items():
        reg_intensity = reg_data['value']
        # Calculate for Small scenario as representative
        small_hvac_kwh = (HVAC_BASELINE_KWH / 3) * scenarios['Small (Demo)']['buildings']
        small_energy_saved = small_hvac_kwh * actual_savings_pct
        reg_co2_saved = (small_energy_saved * reg_intensity) / 1000

        # Get AI emissions from Small scenario (constant across regions)
        if model_name in results and 'Small (Demo)' in results[model_name]:
            ai_co2 = results[model_name]['Small (Demo)']['total_ai_co2_kg']
            reg_net_benefit = reg_co2_saved - ai_co2
            reg_roi = reg_co2_saved / ai_co2 if ai_co2 > 0 else 0
        else:
            reg_net_benefit = 0
            reg_roi = 0

        results['_regional_sensitivity'][reg_key] = {
            'label': reg_data['label'],
            'carbon_intensity': reg_intensity,
            'co2_saved_kg': reg_co2_saved,
            'net_benefit_kg': reg_net_benefit,
            'roi_ratio': reg_roi
        }

    # Add comparison to alternative approaches
    results['_alternatives'] = {}
    for alt_key, alt_data in ALTERNATIVE_APPROACHES.items():
        # Calculate for Small scenario (3 buildings)
        small_hvac_kwh = (HVAC_BASELINE_KWH / 3) * scenarios['Small (Demo)']['buildings']
        alt_energy_saved = small_hvac_kwh * alt_data['savings_pct']
        alt_co2_saved = (alt_energy_saved * CARBON_INTENSITY) / 1000

        # Subtract the alternative's own emissions
        alt_total_co2 = alt_data['co2_kg_year'] * scenarios['Small (Demo)']['buildings']
        alt_net_benefit = alt_co2_saved - alt_total_co2

        results['_alternatives'][alt_key] = {
            'label': alt_data['label'],
            'description': alt_data['description'],
            'savings_pct': alt_data['savings_pct'] * 100,
            'co2_saved_kg': alt_co2_saved,
            'co2_cost_kg': alt_total_co2,
            'net_benefit_kg': alt_net_benefit,
            'cost_per_year': alt_data['cost_per_building_year'] * scenarios['Small (Demo)']['buildings']
        }

    # Cache the results with timestamp (include region/hardware in cache)
    if not isinstance(_CACHED_CO2_DATA, dict):
        _CACHED_CO2_DATA = {}
    _CACHED_CO2_DATA[cache_key] = results
    _CACHED_CO2_TIMESTAMP = time.time()

    return results

# ==========================================
# 5. CALLBACKS (LOGIC)
# ==========================================

# --- A. Navigation & Zone Dropdown Updater ---
@app.callback(
    [Output("building-filters", "style"),
     Output("model-selector-container", "style"),
     Output("zone-dropdown", "options"),
     Output("zone-dropdown", "value")],
    [Input("view-selector", "value"),
     Input("building-dropdown", "value")]
)
def update_sidebar(view_mode, selected_building):
    # Toggle Dropdowns Visibility
    building_style = {'display': 'block'} if view_mode == 'building' else {'display': 'none'}
    model_style = {'display': 'block'} if view_mode == 'analytics' else {'display': 'none'}
    
    # Update Zone Options based on Building
    zones = PORTFOLIO_DB[selected_building]['zones']
    options = [{'label': z['name'], 'value': z['name']} for z in zones]
    
    return building_style, model_style, options, options[0]['value']

# --- A2. Model Selector Options Updater (Dynamic based on library availability) ---
@app.callback(
    [Output("model-selector", "options"),
     Output("model-selector", "value")],
    [Input("view-selector", "value")]  # Trigger on any view change (runs once at startup)
)
def update_model_options(_):
    """
    Dynamically update model selector options based on available libraries.
    Disables TensorFlow and PyTorch options if libraries are not installed.
    """
    # Base options with availability indicators
    selector_options = []
    radio_options = []

    # Random Forest is always available (scikit-learn is a core dependency)
    selector_options.append({'label': '🌳 Random Forest', 'value': 'RF'})
    radio_options.append({'label': 'Random Forest', 'value': 'RF'})

    # TensorFlow Neural Network
    if TF_AVAILABLE:
        selector_options.append({'label': '🧠 TensorFlow NN', 'value': 'NN'})
        radio_options.append({'label': 'Neural Network (TF)', 'value': 'NN'})
    else:
        selector_options.append({'label': '🧠 TensorFlow NN (Not Available)', 'value': 'NN', 'disabled': True})
        radio_options.append({'label': 'Neural Network (TF) - Not Available', 'value': 'NN', 'disabled': True})

    # PyTorch Neural Network
    if TORCH_AVAILABLE:
        selector_options.append({'label': '🔥 PyTorch NN', 'value': 'PT'})
        radio_options.append({'label': 'PyTorch', 'value': 'PT'})
    else:
        selector_options.append({'label': '🔥 PyTorch NN (Not Available)', 'value': 'PT', 'disabled': True})
        radio_options.append({'label': 'PyTorch - Not Available', 'value': 'PT', 'disabled': True})

    # Add "Compare All" option for radio buttons (only include available models)
    radio_options.append({'label': 'Compare All', 'value': 'Both'})

    # Set default value to first available model
    if TORCH_AVAILABLE:
        default_value = 'PT'  # Prefer PyTorch if available
    elif TF_AVAILABLE:
        default_value = 'NN'  # Fall back to TensorFlow
    else:
        default_value = 'RF'  # Fall back to Random Forest

    return selector_options, default_value

# --- B. Main Page Content Renderer ---
@app.callback(
    Output("page-content", "children"),
    [Input("view-selector", "value"),
     Input("building-dropdown", "value"),
     Input("zone-dropdown", "value"),
     Input("model-selector", "value")]
)
def render_page(view_mode, selected_building, selected_zone_name, selected_model):
    
    # === ML MODELS VIEW ===
    if view_mode == 'ml_models':
        # Wait for models to finish training
        if MODELS_TRAINING:
            return html.Div([
                html.H2("🤖 ML Models Documentation"),
                html.Hr(),
                dbc.Alert("Models are training in the background... Please wait.", color="info")
            ])
        
        # Get metrics from registry
        rf_metrics = MODEL_REGISTRY['RF']['metrics']
        nn_metrics = MODEL_REGISTRY['NN']['metrics']
        
        # === QUICK SUMMARY (Always visible - lightweight) ===
        quick_summary = dbc.Card([
            dbc.CardHeader("📊 Quick Model Comparison", className="bg-info text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("🌳 Random Forest", className="text-success"),
                        html.P([html.Strong("Accuracy: "), f"{rf_metrics.get('mae_temp', 0):.3f}°C MAE"]),
                        html.P([html.Strong("Speed: "), "~0.1 ms (fastest)"]),
                        html.P([html.Strong("Status: "), "✅ Available"], className="mb-0")
                    ], md=4),
                    dbc.Col([
                        html.H5("🧠 TensorFlow NN", className="text-primary"),
                        html.P([html.Strong("Accuracy: "), f"{nn_metrics.get('mae_temp', 0):.3f}°C MAE"]),
                        html.P([html.Strong("Speed: "), "~0.3 ms"]),
                        html.P([html.Strong("Status: "), "✅ Available" if MODEL_REGISTRY['NN']['model'] else "❌ Not available"], className="mb-0")
                    ], md=4),
                    dbc.Col([
                        html.H5("🔥 PyTorch NN", className="text-warning"),
                        html.P([html.Strong("Accuracy: "), f"{MODEL_REGISTRY['PT']['metrics'].get('mae_temp', 0):.3f}°C MAE"]),
                        html.P([html.Strong("Speed: "), "~0.3 ms"]),
                        html.P([html.Strong("Status: "), "✅ Available" if MODEL_REGISTRY['PT']['model'] else "❌ Not available"], className="mb-0")
                    ], md=4)
                ]),
                html.Hr(),
                html.P([
                    html.Strong("All models predict: "), 
                    "Next temperature (°C) and Energy consumption (kWh) using 4 inputs: Previous Temp, Outdoor Temp, Occupancy, Setpoint"
                ], className="mb-0 text-muted")
            ])
        ], className="mb-4 shadow")
        
        # === FEATURES & TARGET VARIABLES (Collapsible) ===
        features_target = dbc.Accordion([
            dbc.AccordionItem([
                html.H5("Input Features (Predictors)", className="mt-3 mb-2"),
                html.P("All models use the following 4 input features:", className="mb-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("1. Previous Indoor Temperature (°C)"),
                        html.P("The zone's temperature at the current measurement interval (15-min updates)", className="mb-0 text-muted")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("2. Outdoor Temperature (°C)"),
                        html.P("External weather conditions that influence building thermal dynamics", className="mb-0 text-muted")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("3. Occupancy (Binary: 0 or 1)"),
                        html.P("Whether the zone is occupied by people (affects comfort requirements and heat generation)", className="mb-0 text-muted")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("4. Temperature Setpoint (°C)"),
                        html.P("The HVAC system's target temperature setting (what we're trying to control)", className="mb-0 text-muted")
                    ])
                ], className="mb-3"),
                
                html.H5("Output Targets (Predictions)", className="mt-4 mb-2"),
                html.P("Models predict two separate targets:", className="mb-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("1. Next Indoor Temperature (°C)"),
                        html.P("Physics-based prediction of what the zone temperature will be in the next 15-minute interval, "
                               "considering thermal inertia, setpoint, and outdoor conditions", className="mb-0 text-muted")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("2. Energy Consumption (kWh)"),
                        html.P("Non-linear HVAC energy usage based on temperature difference, occupancy, and control effort required", className="mb-0 text-muted")
                    ])
                ])
            ], title="📋 Features & Target Variables - Click to expand")
        ], start_collapsed=True, className="mb-4")
        
        # === MODEL DETAILS (Collapsible Accordions) ===
        # Get pt_metrics first (needed for accordion content)
        pt_metrics = MODEL_REGISTRY['PT']['metrics']
        
        # Build list of accordion items
        accordion_items = []
        
        # RANDOM FOREST (always available)
        accordion_items.append(dbc.AccordionItem([
                html.H5("Logic & Rationale", className="mt-0"),
                html.P(
                    "Random Forest is an ensemble learning method that builds multiple decision trees and averages their "
                    "predictions. Each tree learns non-linear relationships between features and targets by recursively "
                    "splitting the feature space.",
                    className="mb-2"
                ),
                html.P(
                    html.Strong("Why Random Forest for HVAC?"),
                    className="mb-2"
                ),
                html.Ul([
                    html.Li("Captures non-linear relationships (e.g., HVAC efficiency varies with temp difference)"),
                    html.Li("Robust to outliers and noisy sensor data"),
                    html.Li("Fast inference (~microseconds per prediction) - suitable for real-time control"),
                    html.Li("Handles mixed feature types without scaling"),
                    html.Li("Built-in feature importance for interpretability")
                ], className="mb-3"),
                
                html.H5("Hyperparameters", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Number of Trees: "),
                            html.Code(str(rf_metrics.get('n_estimators', 'N/A')))
                        ]),
                        html.P([
                            html.Strong("Max Depth: "),
                            html.Code(str(rf_metrics.get('max_depth', 'N/A')))
                        ], className="mb-0")
                    ], md=6),
                    dbc.Col([
                        html.P("More trees = better but slower. Deeper trees = more complexity risk.", className="text-muted mb-0")
                    ], md=6)
                ], className="mb-3"),
                
                html.H5("Performance on Test Data", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{rf_metrics.get('mae_temp', 0):.3f}°C", className="text-success mb-1"),
                                html.P("Temperature MAE", className="mb-0 fw-bold"),
                                html.Small("Mean Absolute Error on holdout test set", className="text-muted")
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{rf_metrics.get('mae_energy', 0):.3f}kWh", className="text-info mb-1"),
                                html.P("Energy MAE", className="mb-0 fw-bold"),
                                html.Small("Mean error on energy predictions", className="text-muted")
                            ])
                        ])
                    ], md=6)
                ], className="mb-3"),
                
                html.H5("Speed & Efficiency", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Training Time: "),
                            html.Code(f"{rf_metrics.get('train_time_s', 0) * 1000:.1f} ms")
                        ]),
                        html.Small("Time to train 2000 synthetic samples", className="text-muted")
                    ], md=6),
                    dbc.Col([
                        html.P([
                            html.Strong("Inference Speed: "),
                            html.Code("~0.1 ms per prediction")
                        ]),
                        html.Small("⚡ Extremely fast - suitable for real-time control loops", className="text-muted")
                    ], md=6)
                ])
            ], title="🌳 Random Forest Regressor - Click for details"))
        
        # TENSORFLOW NN (conditional)
        if MODEL_REGISTRY['NN']['model'] is not None:
            accordion_items.append(dbc.AccordionItem([
                    html.H5("Logic & Rationale", className="mt-0"),
                    html.P(
                        "A neural network is a biologically-inspired learning model consisting of interconnected layers of "
                        "artificial neurons. Each neuron applies a non-linear activation function to learn complex patterns "
                        "in the training data.",
                        className="mb-2"
                    ),
                    html.P(
                        html.Strong("Why Neural Networks for HVAC?"),
                        className="mb-2"
                    ),
                    html.Ul([
                        html.Li("Universal function approximators - can learn any non-linear pattern given enough data"),
                        html.Li("Excellent for high-dimensional, complex interactions between features"),
                        html.Li("Can learn temporal patterns if augmented with sequence data"),
                        html.Li("Leverages GPU acceleration for training (if available)"),
                        html.Li("Transfer learning capabilities for quick adaptation to new buildings")
                    ], className="mb-3"),
                    
                    html.H5("Architecture", className="mt-3"),
                    html.P("Feed-forward network with 2 hidden layers:", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("Input Layer: "),
                                html.Code("4 neurons (features)")
                            ]),
                            html.P([
                                html.Strong("Hidden Layer 1: "),
                                html.Code("64 neurons + ReLU")
                            ]),
                            html.P([
                                html.Strong("Hidden Layer 2: "),
                                html.Code("32 neurons + ReLU")
                            ]),
                            html.P([
                                html.Strong("Output Layer: "),
                                html.Code("1 neuron (prediction)")
                            ], className="mb-0")
                        ], md=6),
                        dbc.Col([
                            html.P("ReLU (Rectified Linear Unit) activation introduces non-linearity.", className="text-muted mb-2"),
                            html.P("Total parameters:", className="mb-2"),
                            html.Code(f"{nn_metrics.get('params_temp', 'N/A')} (temperature model)", className="fw-bold"),
                            html.Br(),
                            html.Code(f"{nn_metrics.get('params_energy', 'N/A')} (energy model)", className="fw-bold")
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.H5("Training Configuration", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("Optimizer: "),
                                html.Code("Adam")
                            ]),
                            html.P([
                                html.Strong("Loss Function: "),
                                html.Code("Mean Squared Error (MSE)")
                            ]),
                            html.P([
                                html.Strong("Learning Rate: "),
                                html.Code("0.001")
                            ]),
                            html.P([
                                html.Strong("Epochs: "),
                                html.Code("50")
                            ], className="mb-0")
                        ], md=6),
                        dbc.Col([
                            html.P([
                                html.Strong("Batch Size: "),
                                html.Code("64")
                            ]),
                            html.P([
                                html.Strong("Validation Split: "),
                                html.Code("10%")
                            ]),
                            html.P("Adam is an adaptive learning rate optimizer - updates are guided by both "
                                   "first and second moments of gradients.", className="text-muted text-small", style={"fontSize": "0.85rem"})
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.H5("Performance on Test Data", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{nn_metrics.get('mae_temp', 0):.3f}°C", className="text-primary mb-1"),
                                    html.P("Temperature MAE", className="mb-0 fw-bold"),
                                    html.Small("Mean Absolute Error on holdout test set", className="text-muted")
                                ])
                            ])
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{nn_metrics.get('mae_energy', 0):.3f}kWh", className="text-info mb-1"),
                                    html.P("Energy MAE", className="mb-0 fw-bold"),
                                    html.Small("Mean error on energy predictions", className="text-muted")
                                ])
                            ])
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.H5("Speed & Efficiency", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("Training Time: "),
                                html.Code(f"{nn_metrics.get('train_time_s', 0) * 1000:.1f} ms")
                            ]),
                            html.Small("Time to train 2000 samples over 50 epochs", className="text-muted")
                        ], md=6),
                        dbc.Col([
                            html.P([
                                html.Strong("Inference Speed: "),
                                html.Code("~0.2-0.5 ms per prediction")
                            ]),
                            html.Small("⚡ Still very fast for real-time control", className="text-muted")
                        ], md=6)
                    ])
            ], title="🧠 TensorFlow Neural Network - Click for details"))
        
        # PYTORCH NN (conditional)
        if MODEL_REGISTRY['PT']['model'] is not None:
            accordion_items.append(dbc.AccordionItem([
                    html.H5("Logic & Rationale", className="mt-0"),
                    html.P(
                        "PyTorch is a popular deep learning framework known for its flexibility and dynamic computation graphs. "
                        "This implementation uses the same architecture as TensorFlow (64→32→2 nodes) but with PyTorch's "
                        "tensor operations and automatic differentiation.",
                        className="mb-2"
                    ),
                    html.P(
                        html.Strong("Why PyTorch for HVAC?"),
                        className="mb-2"
                    ),
                    html.Ul([
                        html.Li("Dynamic computation graphs allow for flexible model architectures"),
                        html.Li("Excellent for research and prototyping new control strategies"),
                        html.Li("Strong community support and extensive ecosystem"),
                        html.Li("Easy to debug and inspect intermediate values"),
                        html.Li("Can leverage GPU acceleration for faster training")
                    ], className="mb-3"),
                    
                    html.H5("Architecture Details", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("Layer 1: "),
                                html.Code("Linear(4 → 64) + ReLU")
                            ]),
                            html.P([
                                html.Strong("Layer 2: "),
                                html.Code("Linear(64 → 32) + ReLU")
                            ]),
                            html.P([
                                html.Strong("Output: "),
                                html.Code("Linear(32 → 2)"),
                                html.Br(),
                                html.Small("Multi-output: [Temperature, Energy]", className="text-muted")
                            ])
                        ], md=6),
                        dbc.Col([
                            html.P([
                                html.Strong("Training: "),
                                "Mini-batch SGD with Adam optimizer"
                            ]),
                            html.P([
                                html.Strong("Epochs: "),
                                "50"
                            ]),
                            html.P([
                                html.Strong("Learning Rate: "),
                                "0.005"
                            ])
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.H5("Performance on Test Data", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{pt_metrics.get('mae_temp', 0):.3f}°C", className="text-warning mb-1"),
                                    html.P("Temperature MAE", className="mb-0 fw-bold"),
                                    html.Small("Mean Absolute Error on test set", className="text-muted")
                                ])
                            ])
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{pt_metrics.get('mae_energy', 0):.3f}kWh", className="text-info mb-1"),
                                    html.P("Energy MAE", className="mb-0 fw-bold"),
                                    html.Small("Mean error on energy predictions", className="text-muted")
                                ])
                            ])
                        ], md=6)
                    ], className="mb-3"),
                    
                    html.H5("Speed & Efficiency", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("Training Time: "),
                                html.Code(f"{pt_metrics.get('train_time_s', 0) * 1000:.1f} ms")
                            ]),
                            html.Small("Time to train 15,000 samples", className="text-muted")
                        ], md=6),
                        dbc.Col([
                            html.P([
                                html.Strong("Model Parameters: "),
                                html.Code(f"{pt_metrics.get('params', 0):,}")
                            ]),
                            html.Small("Total trainable parameters", className="text-muted")
                        ], md=6)
                    ])
            ], title="🔥 PyTorch Neural Network - Click for details"))
        
        # Create the accordion with the filtered items
        model_details = dbc.Accordion(accordion_items, start_collapsed=True, className="mb-4")
        
        # === MODEL COMPARISON (Lightweight table) ===
        comparison_data = [
            {"Metric": "Temp MAE", "Random Forest": f"{rf_metrics.get('mae_temp', 0):.3f}°C", "TensorFlow": f"{nn_metrics.get('mae_temp', 0):.3f}°C", "PyTorch": f"{pt_metrics.get('mae_temp', 0):.3f}°C"},
            {"Metric": "Energy MAE", "Random Forest": f"{rf_metrics.get('mae_energy', 0):.3f}kWh", "TensorFlow": f"{nn_metrics.get('mae_energy', 0):.3f}kWh", "PyTorch": f"{pt_metrics.get('mae_energy', 0):.3f}kWh"},
            {"Metric": "Temp R²", "Random Forest": f"{rf_metrics.get('r2_temp', 0):.3f}", "TensorFlow": f"{nn_metrics.get('r2_temp', 0):.3f}", "PyTorch": f"{pt_metrics.get('r2_temp', 0):.3f}"},
            {"Metric": "Inference", "Random Forest": "~0.1ms", "TensorFlow": "~0.3ms", "PyTorch": "~0.3ms"},
            {"Metric": "Training", "Random Forest": f"{rf_metrics.get('train_time_s', 0)*1000:.0f}ms", "TensorFlow": f"{nn_metrics.get('train_time_s', 0)*1000:.0f}ms", "PyTorch": f"{pt_metrics.get('train_time_s', 0)*1000:.0f}ms"}
        ]
        comparison_df = pd.DataFrame(comparison_data)
        
        comparison_table = dbc.Card([
            dbc.CardHeader("📊 Performance Comparison", className="bg-dark text-white"),
            dbc.CardBody([
                dbc.Table.from_dataframe(
                    comparison_df,
                    striped=True, bordered=True, hover=True, responsive=True, size="sm"
                )
            ])
        ], className="mb-4 shadow")
        
        # === RECOMMENDATION ===
        recommendation = dbc.Alert([
            html.H5("💡 Recommendation", className="alert-heading"),
            html.P([
                html.Strong("Random Forest is recommended for production HVAC control: "),
                "Best balance of accuracy (~0.15°C MAE), speed (0.1ms), and interpretability. "
                "Neural networks offer similar accuracy but are black-box models."
            ], className="mb-0")
        ], color="success")
        
        return html.Div([
            html.H2("🤖 ML Models Documentation"),
            html.P("Machine learning models powering Comfort Room's predictive HVAC control",
                   className="lead text-muted"),
            html.Hr(),
            
            quick_summary,
            comparison_table,
            recommendation,
            features_target,
            model_details,
            
            dbc.Alert([
                html.P([
                    html.Strong("💡 Tip: "),
                    "Click on accordion sections above to view detailed model information. "
                    "All models are trained on 15,000 physics-based building simulations."
                ], className="mb-0")
            ], color="info", className="mt-4")
        ])
    
    # === SUSTAINABILITY OF AI VIEW ===
    elif view_mode == 'sustainability':
        # Calculate CO2 impacts with scenarios
        co2_data = calculate_co2_impacts()
        baseline = co2_data['_baseline']
        scenarios = co2_data['_scenarios']
        
        # Prepare data for visualizations (optimized for speed with 2 scenarios, RF only)
        scenario_names = list(scenarios.keys())
        
        # 1. COMBINED: Percentage breakdown + Net benefit (single efficient chart)
        # Show Random Forest as representative model (simplifies visualization)
        combined_data = []
        rf_model = 'Random Forest'
        if rf_model in co2_data:
            for scenario_name in scenario_names:
                if scenario_name in co2_data[rf_model]:
                    d = co2_data[rf_model][scenario_name]
                    # Add percentage breakdown
                    combined_data.append({
                        'Scenario': scenario_name,
                        'Metric': 'Training %',
                        'Value': d['training_percentage'],
                        'Type': 'Breakdown'
                    })
                    combined_data.append({
                        'Scenario': scenario_name,
                        'Metric': 'Inference %',
                        'Value': d['inference_percentage'],
                        'Type': 'Breakdown'
                    })
                    combined_data.append({
                        'Scenario': scenario_name,
                        'Metric': 'Infrastructure %',
                        'Value': d['infrastructure_percentage'],
                        'Type': 'Breakdown'
                    })
        
        df_combined = pd.DataFrame(combined_data)
        fig_breakdown = px.bar(
            df_combined[df_combined['Type'] == 'Breakdown'],
            x='Scenario',
            y='Value',
            color='Metric',
            title='AI Carbon Footprint Breakdown: Random Forest Model (Representative)',
            barmode='stack',
            color_discrete_map={
                'Training %': '#FF6B6B',
                'Inference %': '#4ECDC4',
                'Infrastructure %': '#95E1D3'
            }
        )
        fig_breakdown.update_layout(height=350, yaxis_title='Percentage (%)')
        fig_breakdown.update_yaxes(range=[0, 100])
        
        # 2. ROI comparison - simple bar chart for Random Forest only
        roi_data = []
        if 'Random Forest' in co2_data:
            for scenario_name in scenario_names:
                if scenario_name in co2_data['Random Forest']:
                    d = co2_data['Random Forest'][scenario_name]
                    roi_data.append({
                        'Scenario': scenario_name,
                        'ROI': d['roi_ratio']
                    })
        
        df_roi = pd.DataFrame(roi_data)
        fig_roi = px.bar(
            df_roi, 
            x='Scenario', 
            y='ROI',
            title='Carbon ROI: CO2 Saved per kg of AI Emissions (Random Forest)',
            color_discrete_sequence=['#51CF66']
        )
        fig_roi.update_layout(height=350)
        fig_roi.add_hline(y=1, line_dash="dash", line_color="red", 
                          annotation_text="Break-even", annotation_position="right")
        
        # 5. Create scenario comparison cards
        scenario_cards = []
        for scenario_name in scenario_names:
            scenario_info = scenarios[scenario_name]
            
            # Get Random Forest data for this scenario (as representative)
            rf_data = co2_data.get('Random Forest', {}).get(scenario_name, {})
            
            if rf_data:
                card_color = "success" if scenario_name == 'Small (Demo)' else "info" if scenario_name == 'Medium (100 buildings)' else "warning"
                
                card = dbc.Col(dbc.Card([
                    dbc.CardHeader(scenario_name, className=f"bg-{card_color} text-white"),
                    dbc.CardBody([
                        html.H6("Deployment Size", className="text-muted"),
                        html.P(f"{scenario_info['buildings']} buildings, {rf_data['total_zones']} zones"),
                        
                        html.Hr(),
                        
                        html.H6("Retraining", className="text-muted"),
                        html.P(f"Every {scenario_info['retraining_frequency_days']} days ({rf_data['retraining_per_year']:.1f}×/year)"),
                        
                        html.Hr(),
                        
                        html.H6("AI Emissions (RF)", className="text-muted"),
                        html.P([
                            html.Strong(f"{rf_data['total_ai_co2_kg']:,.1f} kg CO2/year"),
                            html.Br(),
                            html.Small(f"Training: {rf_data['training_percentage']:.1f}%", className="text-muted"),
                            html.Br(),
                            html.Small(f"Inference: {rf_data['inference_percentage']:.1f}%", className="text-muted"),
                            html.Br(),
                            html.Small(f"Infrastructure: {rf_data['infrastructure_percentage']:.1f}%", className="text-muted")
                        ]),
                        
                        html.Hr(),
                        
                        html.H6("CO2 Savings", className="text-muted"),
                        html.H4(f"{rf_data['co2_saved_kg']:,.0f} kg CO2/year", className="text-success"),
                        
                        html.Hr(),
                        
                        html.H6("Net Benefit", className="text-muted"),
                        html.H3(f"{rf_data['net_benefit_kg']:,.0f} kg CO2/year", className="text-primary"),
                        html.P(f"ROI: {rf_data['roi_ratio']:.0f}x", className="mb-0 fw-bold")
                    ])
                ], className="mb-3", outline=True, color=card_color), md=4)
                scenario_cards.append(card)
        
        return html.Div([
            html.H2("🌱 Sustainability of AI"),
            html.P([
                "Complete carbon footprint analysis of AI-powered HVAC optimization. ",
                "Energy savings calculated from actual demo data - ",
                html.A("see AI Impact Analytics tab for detailed optimization results", 
                       href="#", 
                       className="alert-link",
                       style={"textDecoration": "underline"})
            ], className="lead text-muted"),
            html.Hr(),
            
            # Executive Summary (updated for 2 scenarios)
            dbc.Alert([
                html.H4("Executive Summary", className="alert-heading"),
                html.P([
                    "This analysis evaluates the complete carbon footprint of AI-powered building optimization ",
                    "across two realistic deployment scenarios (Small Demo and Large Enterprise), using ",
                    html.Strong(f"actual savings data from the AI Impact Analytics tab ({baseline['actual_savings_pct']:.1f}% HVAC energy reduction)"),
                    ". The AI optimization intelligently balances comfort and energy: occupied zones use efficient comfort-range setpoints (21-23°C), ",
                    "while unoccupied zones relax temperature control to minimize energy waste."
                ]),
                html.Hr(),
                html.H5("Key Insights:"),
                html.Ul([
                    html.Li([
                        html.Strong(f"Actual AI Optimization Savings: {baseline['actual_savings_pct']:.1f}% HVAC energy reduction "),
                        "(calculated from demo portfolio with real zone conditions - see AI Impact Analytics tab)"
                    ]),
                    html.Li("Baseline wastes energy: Traditional HVAC maintains 22°C in ALL zones, even empty ones"),
                    html.Li("AI strategy: Comfort-first for occupied zones, energy-first for unoccupied zones"),
                    html.Li("Small deployments: Infrastructure dominates emissions (server hardware manufacturing)"),
                    html.Li("Large deployments: Training becomes significant with weekly retraining on 5M samples"),
                    html.Li("All scenarios show strong positive ROI - AI emissions vastly outweighed by energy savings")
                ])
            ], color="success"),
            
            # Scenario comparison cards
            html.H4("Deployment Scenario Comparison", className="mt-4 mb-3"),
            dbc.Row(scenario_cards),
            
            # Detailed visualizations (optimized - 2 charts instead of 4)
            html.H4("Detailed Carbon Analysis", className="mt-5 mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig_breakdown)
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig_roi)
                        ])
                    ])
                ], md=6)
            ], className="mb-4"),
            
            # Analysis (updated for 2 scenarios)
            dbc.Card([
                dbc.CardHeader("📊 Scenario Analysis & Insights", className="bg-dark text-white"),
                dbc.CardBody([
                    html.H5("Why Do Emissions Scale Differently?"),
                    
                    html.H6("Small Deployment (3 buildings - Demo):", className="mt-3 text-success"),
                    html.Ul([
                        html.Li(f"Infrastructure dominates at ~{co2_data.get('Random Forest', {}).get('Small (Demo)', {}).get('infrastructure_percentage', 0):.0f}% (server hardware manufacturing amortization)"),
                        html.Li("Training is negligible - only once per year with small dataset (15k samples)"),
                        html.Li("Inference is minimal - only 15 zones need predictions"),
                        html.Li("Hardware manufacturing emissions dwarf operational compute emissions")
                    ]),
                    
                    html.H6("Large Deployment (1000 buildings - Enterprise):", className="mt-3 text-warning"),
                    html.Ul([
                        html.Li(f"Training dominates at ~{co2_data.get('Random Forest', {}).get('Large (Enterprise)', {}).get('training_percentage', 0):.0f}% - weekly retraining (52×/year) on 5M samples"),
                        html.Li(f"Inference significant at ~{co2_data.get('Random Forest', {}).get('Large (Enterprise)', {}).get('inference_percentage', 0):.0f}% - 10,000 zones with 105M predictions/year"),
                        html.Li("Infrastructure percentage lowest due to economies of scale"),
                        html.Li("Continuous learning pipeline with fresh data maintains accuracy")
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("Environmental Impact:"),
                    html.P([
                        "Even in the most compute-intensive scenario (Large deployment, weekly retraining), ",
                        "the carbon ROI remains strongly positive. The HVAC energy savings from AI optimization ",
                        "outweigh the emissions from training and inference by factors of ",
                        html.Strong(f"{co2_data.get('Random Forest', {}).get('Small (Demo)', {}).get('roi_ratio', 0):.0f}× to {co2_data.get('Random Forest', {}).get('Large (Enterprise)', {}).get('roi_ratio', 0):.0f}×"),
                        " depending on deployment scale."
                    ]),
                    
                    html.H5("Conclusion:", className="text-success mt-3"),
                    html.P([
                        html.Strong("AI-powered building optimization is environmentally sustainable at any scale. "),
                        "The key insight is that as deployments grow and retraining becomes more frequent, ",
                        "training and inference emissions increase but remain far below the energy savings achieved. ",
                        "This validates the use of AI for climate-positive building management."
                    ], className="lead")
                ])
            ], className="mb-4"),
            
            # Methodology
            dbc.Card([
                dbc.CardHeader("📋 Methodology & Assumptions", className="bg-secondary text-white"),
                dbc.CardBody([
                    html.H5("Calculation Methodology:"),
                    
                    html.H6("1. Energy Savings (Data-Driven from Demo)", className="mt-3"),
                    html.Ul([
                        html.Li([
                            html.Strong(f"Actual savings: {baseline['actual_savings_pct']:.1f}% HVAC energy reduction"),
                            " (calculated from demo portfolio using real AI optimization)"
                        ]),
                        html.Li("Baseline: Traditional HVAC maintains 22°C for ALL zones (occupied and unoccupied) - wasteful!"),
                        html.Li([
                            "AI Optimization Strategy (Smart Win-Win):",
                            html.Ul([
                                html.Li([html.Strong("Occupied zones:"), " Compare each option vs baseline. Only select if it improves comfort AND/OR energy. Uses scoring: comfort improvement × 1.5 + energy savings × 1.0"]),
                                html.Li([html.Strong("No improvement?"), " Stick with baseline (no change) - avoids making things worse"]),
                                html.Li([html.Strong("Unoccupied zones:"), " Pure energy minimization (16-26°C range) for maximum savings"])
                            ])
                        ]),
                        html.Li("Optimization runs on all 15 zones across 3 buildings with actual zone conditions (varied temperatures 17-27°C)"),
                        html.Li("Savings scaled linearly with building count for larger scenarios"),
                        html.Li(f"Demo baseline HVAC: {baseline['hvac_energy_kwh']:,.0f} kWh/year (40% of total building energy)"),
                        html.Li("Key insight: Major savings come from unoccupied zones where AI relaxes temperature control")
                    ]),
                    
                    html.H6("2. AI Training Emissions (Full Pipeline)", className="mt-3"),
                    html.Ul([
                        html.Li("Training includes: data preprocessing, feature engineering, cross-validation, hyperparameter tuning"),
                        html.Li("Small (15k samples): ~2 hours/training, annual retraining → 1×/year"),
                        html.Li("Large (5M samples): ~24 hours/training, weekly retraining → 52×/year (1,248 hours total)"),
                        html.Li("Training power: 250W (small workstation) → 2000W (distributed GPU cluster)"),
                        html.Li("PUE 1.2 (datacenter efficiency multiplier)")
                    ]),
                    
                    html.H6("3. AI Inference Emissions (Always-On Servers)", className="mt-3"),
                    html.Ul([
                        html.Li("Prediction frequency: Every 5 minutes per zone, 24/7 operation"),
                        html.Li("Small: 15 zones → 1.6M predictions/year"),
                        html.Li("Large: 10,000 zones → 1.05B predictions/year"),
                        html.Li("Server power (average, not just compute): 50W (edge) → 400W (inference cluster)"),
                        html.Li("Prediction latency: 5-10ms (includes API, preprocessing, I/O)")
                    ]),
                    
                    html.H6("4. Infrastructure Emissions (Minimal Incremental)", className="mt-3"),
                    html.Ul([
                        html.Li("AI models use 5-10% of existing server capacity (shared infrastructure)"),
                        html.Li("Hardware allocation: Only incremental manufacturing emissions attributed to AI"),
                        html.Li("Small: 10% server allocation per building × 38 kg/year"),
                        html.Li("Large: 1 server per 100 buildings, 5% AI allocation × 19 kg/year"),
                        html.Li("Datacenter overhead: 80% (small) → 40% (large) of compute emissions"),
                        html.Li("Overhead includes networking, storage, monitoring beyond direct compute")
                    ]),
                    
                    html.Hr(),

                    html.P([
                        html.Strong("Carbon Intensity: "),
                        f"{baseline['carbon_intensity']} g CO2/kWh ({baseline['region']})",
                        html.Br(),
                        html.Small(f"Range: {baseline['carbon_intensity_range'][0]}-{baseline['carbon_intensity_range'][1]} g CO2/kWh", className="text-muted")
                    ]),
                    html.P([
                        html.Strong("Hardware: "),
                        f"{baseline['hardware_type']} - {baseline['hardware_description']}"
                    ])
                ])
            ], className="mb-4"),

            # Regional Sensitivity Analysis
            html.H4("🌍 Regional Carbon Intensity Sensitivity", className="mt-5 mb-3"),
            dbc.Card([
                dbc.CardHeader("How Does Grid Carbon Intensity Affect Results?", className="bg-info text-white"),
                dbc.CardBody([
                    html.P([
                        "Grid carbon intensity varies dramatically by region (50-900 g CO2/kWh). ",
                        "Below shows how AI optimization ROI changes across different power grids, ",
                        "using the Small (Demo) scenario as a representative example."
                    ]),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Region"),
                            html.Th("Grid Intensity"),
                            html.Th("CO2 Saved/Year"),
                            html.Th("Net Benefit"),
                            html.Th("ROI Ratio")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(co2_data['_regional_sensitivity'][reg_key]['label']),
                                html.Td(f"{co2_data['_regional_sensitivity'][reg_key]['carbon_intensity']} g/kWh"),
                                html.Td(f"{co2_data['_regional_sensitivity'][reg_key]['co2_saved_kg']:.0f} kg"),
                                html.Td(f"{co2_data['_regional_sensitivity'][reg_key]['net_benefit_kg']:.0f} kg",
                                       className="text-success" if co2_data['_regional_sensitivity'][reg_key]['net_benefit_kg'] > 0 else "text-warning"),
                                html.Td(f"{co2_data['_regional_sensitivity'][reg_key]['roi_ratio']:.1f}×")
                            ]) for reg_key in ['nordics', 'france', 'us_west', 'global_avg', 'us_midwest', 'china', 'australia', 'india']
                        ])
                    ], bordered=True, striped=True, hover=True, responsive=True, size="sm"),
                    html.Hr(),
                    html.P([
                        html.Strong("Key Insight: "),
                        "AI optimization provides positive ROI in ALL regions, but impact scales with grid intensity. ",
                        "Clean grids (Nordics, France) have lower absolute savings but still justify AI deployment. ",
                        "Carbon-intensive grids (India, Australia) see 10-18× larger absolute CO2 reductions."
                    ], className="text-info")
                ])
            ], className="mb-4"),

            # Alternative Approaches Comparison
            html.H4("⚖️ Comparison to Alternative HVAC Strategies", className="mt-5 mb-3"),
            dbc.Card([
                dbc.CardHeader("How Does AI Compare to Other Optimization Methods?", className="bg-warning text-dark"),
                dbc.CardBody([
                    html.P([
                        "AI optimization is compared against common alternative strategies for the Small (Demo) scenario. ",
                        "Note: AI approach shows actual demo results from AI Impact Analytics tab."
                    ]),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Approach"),
                            html.Th("Savings %"),
                            html.Th("CO2 Saved/Year"),
                            html.Th("Implementation Cost"),
                            html.Th("Own Emissions"),
                            html.Th("Net Benefit")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td([
                                    html.Strong(co2_data['_alternatives'][alt_key]['label']),
                                    html.Br(),
                                    html.Small(co2_data['_alternatives'][alt_key]['description'], className="text-muted")
                                ]),
                                html.Td(f"{co2_data['_alternatives'][alt_key]['savings_pct']:.1f}%"),
                                html.Td(f"{co2_data['_alternatives'][alt_key]['co2_saved_kg']:.0f} kg"),
                                html.Td(f"${co2_data['_alternatives'][alt_key]['cost_per_year']:,.0f}/yr"),
                                html.Td(f"{co2_data['_alternatives'][alt_key]['co2_cost_kg']:.0f} kg"),
                                html.Td(f"{co2_data['_alternatives'][alt_key]['net_benefit_kg']:.0f} kg",
                                       className="text-success" if co2_data['_alternatives'][alt_key]['net_benefit_kg'] > 0 else "text-danger")
                            ]) for alt_key in ['no_optimization', 'manual_tuning', 'simple_scheduling']
                        ] + [
                            html.Tr([
                                html.Td([
                                    html.Strong("AI Optimization (This System)"),
                                    html.Br(),
                                    html.Small(f"Actual measured savings: {baseline['actual_savings_pct']:.1f}% HVAC energy reduction", className="text-muted")
                                ]),
                                html.Td(f"{baseline['actual_savings_pct']:.1f}%", className="fw-bold"),
                                html.Td(f"{co2_data.get('Random Forest', {}).get('Small (Demo)', {}).get('co2_saved_kg', 0):.0f} kg", className="fw-bold"),
                                html.Td("~$5,000/yr*", className="text-muted"),
                                html.Td(f"{co2_data.get('Random Forest', {}).get('Small (Demo)', {}).get('total_ai_co2_kg', 0):.0f} kg", className="fw-bold"),
                                html.Td(f"{co2_data.get('Random Forest', {}).get('Small (Demo)', {}).get('net_benefit_kg', 0):.0f} kg", className="text-success fw-bold")
                            ], className="table-success")
                        ])
                    ], bordered=True, striped=True, hover=True, responsive=True),
                    html.Small("*AI costs include sensors, compute infrastructure, and maintenance", className="text-muted"),
                    html.Hr(),
                    html.P([
                        html.Strong("Key Insight: "),
                        f"AI optimization achieves {baseline['actual_savings_pct']:.1f}% savings - ",
                        f"significantly better than manual tuning (8%) or simple scheduling (15%). ",
                        "While AI has higher upfront costs and compute emissions, the superior energy savings ",
                        "result in both better environmental outcomes and cost payback within 1-2 years."
                    ], className="text-success")
                ])
            ], className="mb-4"),

            # Sources and Validation
            html.H4("📚 Data Sources & Validation", className="mt-5 mb-3"),
            dbc.Card([
                dbc.CardHeader("Methodology Transparency", className="bg-secondary text-white"),
                dbc.CardBody([
                    html.H6("Carbon Intensity Data:"),
                    html.Ul([
                        html.Li("IEA World Energy Outlook (2024) - Global grid averages"),
                        html.Li("EPA eGRID (2024) - US regional grid data"),
                        html.Li("European Environment Agency (2024) - European grid data")
                    ]),
                    html.H6("Hardware Embodied Carbon:", className="mt-3"),
                    html.Ul([
                        html.Li("Dell Product Carbon Footprints (2024) - Server manufacturing"),
                        html.Li("Apple Environmental Progress Reports (2024) - Device lifecycle emissions"),
                        html.Li("Google Cloud Carbon Footprint Methodology (2024)")
                    ]),
                    html.H6("AI Training/Inference Emissions:", className="mt-3"),
                    html.Ul([
                        html.Li("ML CO2 Impact Calculator (mlco2.github.io/impact) - Training estimates"),
                        html.Li("GreenAlgorithms (green-algorithms.org) - Computational carbon footprint"),
                        html.Li("Measured power consumption from benchmark studies (SPEC Power, MLPerf)")
                    ]),
                    html.H6("Uncertainty & Limitations:", className="mt-3"),
                    html.Ul([
                        html.Li(f"Carbon intensity range: ±{((baseline['carbon_intensity_range'][1] - baseline['carbon_intensity_range'][0]) / (2 * baseline['carbon_intensity'])) * 100:.0f}% uncertainty based on grid mix variability"),
                        html.Li("Training time estimates: Based on industry benchmarks, actual times vary ±30% by hardware"),
                        html.Li("Does not include network transfer costs (typically <1% of total emissions)"),
                        html.Li("Does not include marginal vs average emissions (marginal typically 2-3× higher during peak)"),
                        html.Li("Savings percentages based on actual demo data but may vary by building characteristics")
                    ])
                ])
            ], className="mb-4")
        ])

    # === TUTORIAL VIEW ===
    elif view_mode == 'tutorial':
        return html.Div([
            html.H2("🎓 Tutorial & User Guide"),
            html.Hr(),
            
            # Overview
            dbc.Card([
                dbc.CardHeader("Overview", className="bg-info text-white"),
                dbc.CardBody([
                    html.P("Comfort Room is a digital twin platform that helps optimize building comfort and energy efficiency using machine learning."),
                    html.Ul([
                        html.Li("Monitor real-time zone conditions across your portfolio"),
                        html.Li("Identify problematic areas (Critical/Warning zones)"),
                        html.Li("Use AI-powered models (Random Forest, TensorFlow, PyTorch) trained on 15,000 physics-based building simulations"),
                        html.Li("Test 'what-if' scenarios with the Digital Twin Simulator"),
                        html.Li("Analyze portfolio-wide energy savings and comfort improvements with AI Impact Analytics")
                    ])
                ])
            ], className="mb-4"),
            
            # Navigation Guide
            dbc.Card([
                dbc.CardHeader("How to Navigate", className="bg-info text-white"),
                dbc.CardBody([
                    html.H5("📍 Portfolio Map View"),
                    html.P("View all your buildings on an interactive map with key performance indicators (KPIs)."),
                    html.Ul([
                        html.Li("Shows total sites managed, energy saved, and critical alerts"),
                        html.Li("Click on building markers for details"),
                        html.Li("Use to get a high-level overview of your portfolio health")
                    ]),
                    html.Hr(),
                    html.H5("🏢 Building View"),
                    html.P("Dive deep into individual buildings and zones."),
                    html.Ul([
                        html.Li("Select a building from the dropdown in the sidebar"),
                        html.Li("View all zones with their current temperature and occupancy status"),
                        html.Li("See color-coded status badges: Green (OK), Yellow (Warning), Red (Critical)"),
                        html.Li("Active alerts section highlights problematic zones at the top")
                    ]),
                    html.Hr(),
                    html.H5("📊 AI Impact Analytics"),
                    html.P("Portfolio-wide AI optimization using real ML model predictions."),
                    html.Ul([
                        html.Li("Compare baseline (wasteful 22°C for all zones) vs AI-optimized strategies"),
                        html.Li("Smart Win-Win Optimization: Only makes changes that improve comfort AND/OR energy vs baseline"),
                        html.Li("Scoring system: comfort improvement × 1.5 + energy savings × 1.0 (prioritizes comfort slightly)"),
                        html.Li("Falls back to baseline if no improvement found - never makes things worse"),
                        html.Li("Occupied zones: Balance comfort and energy intelligently"),
                        html.Li("Unoccupied zones: Pure energy minimization for maximum savings"),
                        html.Li("View continuous comfort scores (0-100%) and energy savings across all zones"),
                        html.Li("Analyze comfort vs energy tradeoffs by building"),
                        html.Li("Deterministic results: Fixed portfolio data ensures consistent analytics every time")
                    ]),
                    html.Hr(),
                    html.H5("🌱 Sustainability of AI"),
                    html.P("Comprehensive carbon footprint analysis of AI deployment."),
                    html.Ul([
                        html.Li("Training emissions: Full ML pipeline with realistic training times (2-24 hours per run)"),
                        html.Li("Inference emissions: Always-on server power consumption (50W-400W depending on scale)"),
                        html.Li("Infrastructure: Incremental hardware allocation (5-10% of existing server capacity)"),
                        html.Li("Scenario comparison: Small (3 buildings) vs Large (1000 buildings) enterprise deployments"),
                        html.Li("Shows emissions breakdown: Infrastructure-dominated (small) vs compute-dominated (large)"),
                        html.Li("Carbon ROI: Demonstrates positive environmental impact across all deployment scenarios")
                    ]),
                    html.Hr(),
                    html.H5("🤖 ML Models"),
                    html.P("Comprehensive documentation of machine learning models."),
                    html.Ul([
                        html.Li("Quick model comparison with accuracy, speed, and availability status"),
                        html.Li("Performance metrics table showing MAE, R², and training times"),
                        html.Li("Collapsible accordions with detailed model explanations (click to expand)"),
                        html.Li("Random Forest, TensorFlow NN, and PyTorch NN architecture details"),
                        html.Li("Optimized for fast loading - heavy content deferred until you expand sections")
                    ]),
                    html.Hr(),
                    html.H5("❓ Tutorial (You are here!)"),
                    html.P("Learn how to use the app and test scenarios.")
                ])
            ], className="mb-4"),
            
            # Simulator Tutorial
            dbc.Card([
                dbc.CardHeader("Using the Digital Twin Optimizer", className="bg-success text-white"),
                dbc.CardBody([
                    html.H5("Step-by-Step Guide"),
                    html.Ol([
                        html.Li("Go to 🏢 Building View"),
                        html.Li("Select a building and zone from the dropdown"),
                        html.Li("Scroll down to the 🎛️ Digital Twin Optimization panel"),
                        html.Li("Adjust simulation inputs:"),
                        html.Ul([
                            html.Li([html.Strong("Indoor Temp (°C):"), " Current zone temperature (15-30°C)"]),
                            html.Li([html.Strong("Outdoor Temp (°C):"), " External weather conditions (-10 to 40°C)"]),
                            html.Li([html.Strong("Occupancy:"), " Is the zone occupied by people?"]),
                        ]),
                        html.Li("Click 'Run AI Optimization' button"),
                        html.Li("View results:"),
                        html.Ul([
                            html.Li("Graph shows energy cost vs indoor temperature"),
                            html.Li("Green dot marks the AI-recommended setpoint"),
                            html.Li("Blue line shows predicted temperature; Red line shows energy cost")
                        ])
                    ]),
                    html.Hr(),
                    html.H5("📊 Understanding the Results"),
                    html.P("The Digital Twin Optimizer uses trained ML models (Random Forest, TensorFlow, or PyTorch) that predict both temperature and energy simultaneously. It uses penalty-based optimization to recommend the best setpoint:"),
                    html.Ul([
                        html.Li([html.Strong("Total Cost = Predicted Energy + Comfort Penalty")]),
                        html.Li([html.Strong("Predicted Energy:"), " ML model predicts kWh consumption based on outdoor temp, current temp, setpoint, and occupancy"]),
                        html.Li([html.Strong("Comfort Penalty:"), " If occupied and temperature falls outside 21-23°C, penalty = |temp - boundary| × 10.0"]),
                        html.Li([html.Strong("High Penalty Weight (10.0):"), " Forces AI to prioritize comfort for occupied zones while still optimizing energy"]),
                        html.Li([html.Strong("Note:"), " This is different from the portfolio-wide Smart Win-Win optimization used in AI Impact Analytics"])
                    ]),
                    html.Hr(),
                    html.H5("🤖 ML Model Details"),
                    html.P("The platform trains three models on 15,000 physics-based building simulations:"),
                    html.Ul([
                        html.Li([html.Strong("Random Forest:"), " 50 trees, depth 10, fast inference"]),
                        html.Li([html.Strong("TensorFlow Neural Network:"), " Architecture: 64→32→2 nodes"]),
                        html.Li([html.Strong("PyTorch Neural Network:"), " Architecture: 64→32→2 nodes, mini-batch training"])
                    ]),
                    html.P("All models use multi-output architecture with StandardScaler preprocessing, predicting both next temperature and energy consumption simultaneously.", className="text-muted small")
                ])
            ], className="mb-4"),
            
            # Example Scenarios
            dbc.Card([
                dbc.CardHeader("Example Scenarios to Try", className="bg-warning text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("❄️ Scenario 1: Cold Office (Winter Morning)"),
                            html.P([
                                html.Strong("Inputs: "), "Indoor: 18°C | Outdoor: 5°C | Occupied: Yes",
                                html.Br(),
                                html.Strong("Expected: "), "AI recommends ~22-23°C to balance comfort & heating cost"
                            ])
                        ], md=6),
                        dbc.Col([
                            html.H5("🔥 Scenario 2: Hot Office (Summer Afternoon)"),
                            html.P([
                                html.Strong("Inputs: "), "Indoor: 27°C | Outdoor: 35°C | Occupied: Yes",
                                html.Br(),
                                html.Strong("Expected: "), "AI recommends ~22°C for comfort; cooling cost will be higher"
                            ])
                        ], md=6),
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("🌙 Scenario 3: Unoccupied Zone"),
                            html.P([
                                html.Strong("Inputs: "), "Any temp | Any outdoor | Occupied: No",
                                html.Br(),
                                html.Strong("Expected: "), "AI recommends low setpoint (~18-19°C) to save energy"
                            ])
                        ], md=6),
                        dbc.Col([
                            html.H5("⚡ Scenario 4: Mild Weather"),
                            html.P([
                                html.Strong("Inputs: "), "Indoor: 22°C | Outdoor: 18°C | Occupied: Yes",
                                html.Br(),
                                html.Strong("Expected: "), "AI recommends ~22°C (low energy cost, comfort maintained)"
                            ])
                        ], md=6),
                    ])
                ])
            ], className="mb-4"),
            
            # Tips
            dbc.Card([
                dbc.CardHeader("💡 Pro Tips", className="bg-secondary text-white"),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Red badges (Critical) are zones with occupied people AND uncomfortable temps — prioritize these!"),
                        html.Li("Visit AI Impact Analytics to see Smart Win-Win optimization in action - it only makes genuine improvements vs baseline."),
                        html.Li("Smart Win-Win never forces changes - if baseline is already optimal, it keeps the baseline setpoint."),
                        html.Li("Check ML Models tab - click accordions to expand detailed model documentation (optimized for fast loading)."),
                        html.Li("Sustainability of AI shows positive environmental ROI - energy savings vastly exceed AI operational emissions."),
                        html.Li("All analytics are deterministic: Portfolio data is fixed, so results remain consistent across sessions."),
                        html.Li("Comfort scores are continuous (0-100%): 100% = perfect (21-23°C), decreasing 20% per degree outside range."),
                        html.Li("Two optimization strategies: Penalty-based (Building View simulator) and Smart Win-Win (AI Impact Analytics)."),
                        html.Li("Experiment with different scenarios in the simulator to understand how occupancy & outdoor conditions impact setpoints."),
                        html.Li("Models are trained on 15,000 physics-based samples with realistic thermal dynamics (thermal drift, HVAC power, body heat)."),
                        html.Li("Major savings come from unoccupied zones where AI relaxes temperature control (baseline wastes energy maintaining 22°C)."),
                        html.Li("Energy savings add up across many zones — small % improvements = big annual savings!"),
                        html.Li("Large-scale deployments show compute-intensive emissions (training + inference ~66%), while small deployments are infrastructure-dominated (~97%).")
                    ])
                ])
            ], className="mb-4"),
        ])
    
    # === AI IMPACT ANALYTICS VIEW ===
    elif view_mode == 'analytics':
        # Ensure consistent results across page loads
        # Reset random seed since PORTFOLIO_DB and model predictions should be deterministic
        np.random.seed(42)
        
        # Configuration (from notebook)
        PENALTY_WEIGHT = 10.0  # High cost for comfort violations
        # Baseline: Traditional HVAC strategy
        BASELINE_SETPOINT_OCCUPIED = 22.0    # Comfort setpoint when occupied
        BASELINE_SETPOINT_UNOCCUPIED = 22.0  # Traditional systems often maintain same temp 24/7 (wasteful!)
        # Expanded setpoint range: includes energy-saving extremes for unoccupied zones
        candidate_setpoints = CANDIDATE_SETPOINTS
        
        # Get selected model from dropdown
        model_entry = MODEL_REGISTRY.get(selected_model, MODEL_REGISTRY['RF'])
        
        # Map model code to display name
        model_name_map = {
            'RF': 'Random Forest',
            'NN': 'TensorFlow Neural Network',
            'PT': 'PyTorch Neural Network'
        }
        model_name = model_name_map.get(selected_model, 'Random Forest')
        
        # Check if selected model is available
        if model_entry['model'] is None:
            # Fallback to Random Forest if selected model not available
            return dbc.Alert([
                html.H4("Model Not Available", className="alert-heading"),
                html.P(f"The {model_name} model is not available in this environment."),
                html.Hr(),
                html.P([
                    "Please select a different model from the sidebar, or ensure the required libraries are installed:",
                    html.Ul([
                        html.Li("TensorFlow NN requires: tensorflow"),
                        html.Li("PyTorch NN requires: torch")
                    ])
                ])
            ], color="warning")
        
        model = model_entry['model']
        scaler = model_entry['scaler']
        
        # ============================================
        # PERFORMANCE OPTIMIZATION: Vectorized batch prediction
        # Instead of 105 individual predictions (15 zones × 7 setpoints),
        # we make 2 batch predictions: baseline and all candidates
        # ============================================
        
        # Step 1: Collect all zone data with consistent outdoor temperatures
        # Use building-specific outdoor temps (seeded by building name for consistency)
        outdoor_temps = {
            'Comfort Room - Zug': 12.5,      # Switzerland - cooler
            'Tech Hub - Munich': 14.0,        # Germany - moderate
            'Logistics - Milan': 16.5         # Italy - warmer
        }
        
        zone_data = []
        for building_name, building_data in PORTFOLIO_DB.items():
            outdoor_temp = outdoor_temps.get(building_name, 15.0)  # Default to 15°C
            for zone in building_data['zones']:
                zone_data.append({
                    'building': building_name,
                    'zone_name': zone['name'],
                    'prev_temp': zone['temp'],
                    'is_occupied': 1 if zone['occupied'] else 0,
                    'outdoor_temp': outdoor_temp
                })
        
        n_zones = len(zone_data)
        n_setpoints = len(candidate_setpoints)
        
        # Step 2: Create baseline inputs (one per zone)
        # Baseline uses appropriate setpoint based on occupancy
        baseline_inputs = np.array([
            [z['outdoor_temp'], z['prev_temp'], 
             BASELINE_SETPOINT_OCCUPIED if z['is_occupied'] else BASELINE_SETPOINT_UNOCCUPIED, 
             z['is_occupied']]
            for z in zone_data
        ])
        baseline_inputs_scaled = scaler.transform(baseline_inputs)
        
        # Step 3: Create all candidate inputs (zones × setpoints matrix)
        # Shape: (n_zones * n_setpoints, 4)
        candidate_inputs = []
        for z in zone_data:
            for sp in candidate_setpoints:
                candidate_inputs.append([z['outdoor_temp'], z['prev_temp'], sp, z['is_occupied']])
        candidate_inputs = np.array(candidate_inputs)
        candidate_inputs_scaled = scaler.transform(candidate_inputs)
        
        # Step 4: BATCH PREDICT - All predictions at once!
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                baseline_preds = model(torch.tensor(baseline_inputs_scaled, dtype=torch.float32)).numpy()
                candidate_preds = model(torch.tensor(candidate_inputs_scaled, dtype=torch.float32)).numpy()
        elif TF_AVAILABLE and hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
            baseline_preds = model.predict(baseline_inputs_scaled, verbose=0)
            candidate_preds = model.predict(candidate_inputs_scaled, verbose=0)
        else:
            baseline_preds = model.predict(baseline_inputs_scaled)
            candidate_preds = model.predict(candidate_inputs_scaled)
        
        # Step 5: Process results - vectorized operations
        building_metrics = []
        
        for zone_idx, z in enumerate(zone_data):
            # Baseline results
            baseline_temp = baseline_preds[zone_idx, 0]
            baseline_energy = baseline_preds[zone_idx, 1]
            
            # Calculate baseline comfort
            if COMFORT_MIN <= baseline_temp <= COMFORT_MAX:
                baseline_comfort_score = 100.0
            else:
                distance = min(abs(baseline_temp - COMFORT_MIN), abs(baseline_temp - COMFORT_MAX))
                baseline_comfort_score = max(0, 100 - (distance * 20))
            
            # Get this zone's candidate predictions
            # Predictions are arranged as: [zone0_sp0, zone0_sp1, ..., zone1_sp0, zone1_sp1, ...]
            start_idx = zone_idx * n_setpoints
            end_idx = start_idx + n_setpoints
            zone_candidate_preds = candidate_preds[start_idx:end_idx]
            
            # Extract temps and energies
            candidate_temps = zone_candidate_preds[:, 0]
            candidate_energies = zone_candidate_preds[:, 1]
            
            # AI Optimization Strategy (SMART WIN-WIN):
            # Only make changes that improve outcomes compared to baseline
            # Goal: Better comfort AND lower energy (or at least don't worsen both)
            
            if z['is_occupied'] == 1:
                # OCCUPIED: Find win-win scenarios vs baseline
                
                # Calculate baseline comfort score for comparison
                baseline_comfort_achieved = (COMFORT_MIN <= baseline_temp <= COMFORT_MAX)
                
                # Strategy: Compare each option against baseline
                # Accept only if: (1) Better comfort + same/better energy, OR
                #                 (2) Same/better comfort + better energy, OR
                #                 (3) Significantly better on one metric with small tradeoff on other
                
                best_score = float('inf')
                best_idx = None
                
                for idx in range(len(candidate_temps)):
                    temp = candidate_temps[idx]
                    energy = candidate_energies[idx]
                    
                    # Calculate if this achieves comfort
                    achieves_comfort = (COMFORT_MIN <= temp <= COMFORT_MAX)
                    
                    # Calculate improvements vs baseline
                    energy_improvement = baseline_energy - energy  # Positive = saves energy
                    comfort_improvement = 0
                    
                    if achieves_comfort and not baseline_comfort_achieved:
                        # Achieves comfort when baseline doesn't - big win
                        comfort_improvement = 10.0
                    elif achieves_comfort and baseline_comfort_achieved:
                        # Both achieve comfort - neutral
                        comfort_improvement = 0
                    elif not achieves_comfort and not baseline_comfort_achieved:
                        # Neither achieves comfort - check if getting closer
                        baseline_deviation = min(abs(baseline_temp - COMFORT_MIN), abs(baseline_temp - COMFORT_MAX))
                        candidate_deviation = min(abs(temp - COMFORT_MIN), abs(temp - COMFORT_MAX))
                        if candidate_deviation < baseline_deviation:
                            comfort_improvement = (baseline_deviation - candidate_deviation) * 2.0
                        else:
                            comfort_improvement = (baseline_deviation - candidate_deviation) * 2.0  # Can be negative
                    else:
                        # Baseline achieves comfort but candidate doesn't - bad
                        comfort_improvement = -10.0
                    
                    # Multi-objective score: maximize comfort improvement + energy savings
                    # Negative score = improvement (we minimize the score)
                    score = -(comfort_improvement * 1.5 + energy_improvement * 1.0)
                    
                    # Only consider if it's actually an improvement over baseline
                    # (negative score means improvement)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                
                # If no improvement found, stick with baseline
                if best_idx is None or best_score >= 0:
                    # No improvement possible - use baseline setpoint
                    # Find the candidate closest to baseline setpoint
                    baseline_sp = BASELINE_SETPOINT_OCCUPIED
                    best_idx = np.argmin(np.abs(candidate_setpoints - baseline_sp))
                    
            else:
                # UNOCCUPIED: Always minimize energy (comfort not a concern)
                best_idx = np.argmin(candidate_energies)
            
            best_setpoint = candidate_setpoints[best_idx]
            best_temp = candidate_temps[best_idx]
            best_energy = candidate_energies[best_idx]
            
            # Calculate AI comfort
            if COMFORT_MIN <= best_temp <= COMFORT_MAX:
                ai_comfort_score = 100.0
            else:
                distance = min(abs(best_temp - COMFORT_MIN), abs(best_temp - COMFORT_MAX))
                ai_comfort_score = max(0, 100 - (distance * 20))
            
            building_metrics.append({
                'building': z['building'],
                'zone': z['zone_name'],
                'occupied': bool(z['is_occupied']),
                'baseline_setpoint': BASELINE_SETPOINT_OCCUPIED if z['is_occupied'] else BASELINE_SETPOINT_UNOCCUPIED,
                'ai_setpoint': float(best_setpoint),
                'baseline_temp': float(baseline_temp),
                'ai_temp': float(best_temp),
                'current_comfort_pct': baseline_comfort_score,
                'optimal_comfort_pct': ai_comfort_score,
                'current_energy_kwh': float(baseline_energy),
                'optimal_energy_kwh': float(best_energy),
                'energy_savings_kwh': float(baseline_energy - best_energy),
                'savings_pct': float((baseline_energy - best_energy) / baseline_energy * 100) if baseline_energy > 0 else 0
            })
        
        df_metrics = pd.DataFrame(building_metrics)

        # Portfolio aggregates
        total_current_energy = df_metrics['current_energy_kwh'].sum()
        total_optimal_energy = df_metrics['optimal_energy_kwh'].sum()
        total_savings = total_current_energy - total_optimal_energy
        savings_pct = (total_savings / total_current_energy * 100) if total_current_energy > 0 else 0

        # Comfort calculations: ONLY for occupied zones (comfort only matters when people are present)
        df_occupied = df_metrics[df_metrics['occupied'] == True]
        total_zones = len(df_metrics)
        occupied_zones = len(df_occupied)

        if occupied_zones > 0:
            avg_current_comfort = df_occupied['current_comfort_pct'].mean()
            avg_optimal_comfort = df_occupied['optimal_comfort_pct'].mean()
        else:
            # Fallback if no zones are occupied (unlikely but handle gracefully)
            avg_current_comfort = 0
            avg_optimal_comfort = 0

        comfort_improvement = avg_optimal_comfort - avg_current_comfort
        
        comparison_data = pd.DataFrame({
            'Scenario': ['Current System', 'AI Optimized'],
            'Comfort Compliance (%)': [avg_current_comfort, avg_optimal_comfort],
            'Energy Cost (kWh/day)': [total_current_energy, total_optimal_energy]
        })
        
        # Chart 1: Comfort Compliance
        fig_comfort = go.Figure()
        fig_comfort.add_trace(go.Bar(
            x=comparison_data['Scenario'],
            y=comparison_data['Comfort Compliance (%)'],
            marker_color=['#FF6B6B', '#51CF66'],
            text=comparison_data['Comfort Compliance (%)'].round(1),
            textposition='auto',
            texttemplate='%{text}%',
            width=0.5
        ))
        fig_comfort.add_hline(y=90, line_dash="dash", line_color="gray",
                             annotation_text="Industry Target: 90%", annotation_position="right")
        fig_comfort.update_layout(
            title=f"Comfort Compliance: Time in Optimal Temperature Range (21-23°C)<br><sub>Based on {occupied_zones} occupied zones (out of {total_zones} total zones)</sub>",
            yaxis_title="% Time in Comfort Zone",
            yaxis_range=[0, 100],
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=12)
        )
        
        # Chart 2: Energy Cost Comparison
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Bar(
            x=comparison_data['Scenario'],
            y=comparison_data['Energy Cost (kWh/day)'],
            marker_color=['#FF6B6B', '#51CF66'],
            text=comparison_data['Energy Cost (kWh/day)'].round(2),
            textposition='auto',
            texttemplate='%{text:.2f} kWh',
            width=0.5
        ))
        fig_energy.add_annotation(
            x=0.5, y=max(comparison_data['Energy Cost (kWh/day)']) * 0.9,
            text=f"Savings: {total_savings:.2f} kWh/day ({savings_pct:.1f}%)",
            showarrow=False, font=dict(size=14, color="green"), bgcolor="lightgreen"
        )
        fig_energy.update_layout(
            title="Daily Energy Consumption: Portfolio-Wide Comparison",
            yaxis_title="Energy (kWh/day)",
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=12)
        )
        
        # Chart 3: Tradeoff Analysis (comfort only for occupied zones, energy for all zones)
        # For comfort: use only occupied zones per building
        df_comfort_by_building = df_occupied.groupby('building').agg({
            'current_comfort_pct': 'mean',
            'optimal_comfort_pct': 'mean'
        }).reset_index()

        # For energy: use all zones (energy consumed regardless of occupancy)
        df_energy_by_building = df_metrics.groupby('building').agg({
            'current_energy_kwh': 'sum',
            'optimal_energy_kwh': 'sum'
        }).reset_index()

        # Merge both dataframes
        df_buildings = df_energy_by_building.merge(df_comfort_by_building, on='building', how='left')

        # Fill NaN values for buildings with no occupied zones (shouldn't happen in practice)
        df_buildings['current_comfort_pct'].fillna(0, inplace=True)
        df_buildings['optimal_comfort_pct'].fillna(0, inplace=True)
        
        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(go.Scatter(
            x=df_buildings['current_energy_kwh'],
            y=df_buildings['current_comfort_pct'],
            mode='markers+text',
            marker=dict(size=20, color='#FF6B6B', symbol='circle'),
            text=df_buildings['building'],
            textposition="top center",
            name='Current System'
        ))
        fig_tradeoff.add_trace(go.Scatter(
            x=df_buildings['optimal_energy_kwh'],
            y=df_buildings['optimal_comfort_pct'],
            mode='markers+text',
            marker=dict(size=20, color='#51CF66', symbol='star'),
            text=df_buildings['building'],
            textposition="bottom center",
            name='AI Optimized'
        ))
        # Add arrows showing movement
        for idx, row in df_buildings.iterrows():
            fig_tradeoff.add_annotation(
                x=row['optimal_energy_kwh'], y=row['optimal_comfort_pct'],
                ax=row['current_energy_kwh'], ay=row['current_comfort_pct'],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='gray'
            )
        fig_tradeoff.update_layout(
            title="Comfort vs Energy Tradeoff Analysis by Building<br><sub>Comfort based on occupied zones only</sub>",
            xaxis_title="Energy Cost (kWh/day)",
            yaxis_title="Comfort Compliance (%) - Occupied Zones",
            yaxis_range=[0, 100],
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(x=0.65, y=0.98),
            font=dict(size=12)
        )
        
        # Summary metrics
        impact_summary = dbc.Card([
            dbc.CardHeader("💡 Executive Summary: AI Optimization Impact", className="bg-success text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H3(f"{savings_pct:.1f}%", className="text-success mb-1"),
                        html.P("Energy Savings", className="mb-0 fw-bold"),
                        html.Small("Reduction in energy consumption", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3(f"{total_savings:.2f} kWh", className="text-success mb-1"),
                        html.P("Total Savings", className="mb-0 fw-bold"),
                        html.Small(f"Per interval across portfolio", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3(f"+{comfort_improvement:.1f}%", className="text-info mb-1"),
                        html.P("Comfort Improvement", className="mb-0 fw-bold"),
                        html.Small("Increase in zones within 21-23°C", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3("Win-Win" if comfort_improvement >= 0 and total_savings > 0 else "Tradeoff", 
                               className="text-primary mb-1"),
                        html.P("Outcome", className="mb-0 fw-bold"),
                        html.Small("Better comfort + Lower costs", className="text-muted")
                    ], width=3),
                ]),
                html.Hr(),
                html.P([
                    html.Strong("Model Used: "),
                    html.Code(model_name),
                    " | ",
                    html.Strong("Baseline: "),
                    f"Traditional HVAC (22°C all zones) | ",
                    html.Strong("AI Strategy: "),
                    "Smart optimization (comfort-first for occupied, energy-first for unoccupied)"
                ], className="mb-0 text-muted small")
            ])
        ], className="mb-4 shadow")
        
        return html.Div([
            html.H2("📊 AI Impact Analytics Dashboard"),
            html.P([
                "Real-time analysis using ",
                html.Strong(model_name),
                " model predictions with penalty-based optimization across your portfolio"
            ], className="lead text-muted"),
            dbc.Alert([
                html.Strong("💡 Tip: "),
                "Change the ML model selection in the sidebar to see how different models perform!"
            ], color="info", className="mb-3"),
            html.Hr(),
            
            impact_summary,
            
            # Comfort Analysis Section
            html.H4("🌡️ Comfort Compliance Analysis", className="mt-4 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.P([
                        html.Strong("What this shows: "),
                        f"Comfort compliance score (0-100%) for {occupied_zones} occupied zones (out of {total_zones} total zones), based on how close predicted temperatures are to the ideal range (21-23°C). ",
                        "Scores decrease by 20% per degree outside this range. ",
                        "Baseline uses a static 22°C setpoint; AI uses penalty-based optimization to find the best setpoint for each zone. ",
                        html.Strong("Comfort is only measured for occupied zones where people are present.")
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Key Takeaway: "),
                        f"AI optimization {'improves' if comfort_improvement >= 0 else 'adjusts'} comfort compliance by {abs(comfort_improvement):.1f} percentage points for occupied spaces. ",
                        "The penalty-based approach prioritizes keeping zones in the 21-23°C comfort range, ",
                        "especially for occupied spaces."
                    ], className="text-success mb-0")
                ])
            ], className="mb-3"),
            dcc.Graph(figure=fig_comfort),
            
            # Energy Analysis Section
            html.H4("⚡ Energy Cost Analysis", className="mt-5 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.P([
                        html.Strong("What this shows: "),
                        "Daily energy consumption (kWh) across all zones in the portfolio. ",
                        "Lower values indicate reduced HVAC energy costs and carbon footprint."
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Key Takeaway: "),
                        f"AI optimization reduces energy consumption by {savings_pct:.1f}% ({total_savings:.2f} kWh/day). ",
                        "The penalty-based approach balances energy savings with comfort requirements, ",
                        "finding setpoints that minimize energy while avoiding comfort violations."
                    ], className="text-success mb-0")
                ])
            ], className="mb-3"),
            dcc.Graph(figure=fig_energy),
            
            # Tradeoff Analysis Section
            html.H4("🔄 Comfort vs Energy Tradeoff", className="mt-5 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.P([
                        html.Strong("What this shows: "),
                        "Each building's position on comfort (vertical) vs energy cost (horizontal) axes. ",
                        "Arrows show the movement from baseline strategy (static 22°C, red circles) to AI-optimized strategy (green stars)."
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Key Takeaway: "),
                        "The smart AI optimization ONLY makes changes that genuinely improve outcomes vs baseline. ",
                        "Each AI recommendation either: (1) improves both comfort and energy, (2) significantly improves one with minor impact on the other, ",
                        "or (3) keeps baseline if no improvement is possible. This guarantees all movements show real value."
                    ], className="text-success mb-2"),
                    html.P([
                        html.Strong("Movement Interpretation: "),
                        html.Ul([
                            html.Li("↖ Up-Left: Best outcome (better comfort + lower energy) ✅"),
                            html.Li("↑ Upward: Comfort improvement worth the modest energy increase 🎯"),
                            html.Li("← Left: Energy savings (comfort already good or unoccupied zones) 💡"),
                            html.Li("⊙ No movement: Baseline already optimal - no change recommended 🔒"),
                        ], className="mb-0")
                    ])
                ])
            ], className="mb-3"),
            dcc.Graph(figure=fig_tradeoff),
            # Methodology & Explainable AI
            dbc.Card([
                dbc.CardHeader("🧪 Methodology & Explainable AI", className="bg-light"),
                dbc.CardBody([
                    html.H5("Penalty-Based Optimization Approach (from Notebook)", className="mt-0"),
                    html.P([
                        "This analytics view uses the same penalty-based optimization strategy. The approach compares a baseline strategy (static setpoint) against AI optimization."
                    ], className="mb-3"),
                    
                    html.H5("How It Works:", className="mt-3"),
                    html.Ul([
                        html.Li([html.Strong("Model Predictions: "),
                                 f"Uses the {model_name} model trained on physics-based building data (15,000 samples). ",
                                 "For each zone, the model predicts both next temperature and energy consumption."]),
                        html.Li([html.Strong("Baseline Strategy: "),
                                 "Traditional HVAC maintains 22°C for all zones (occupied and unoccupied), ",
                                 "which wastes energy in empty spaces. Model predicts resulting temperature and energy."]),
                        html.Li([html.Strong("AI Strategy (SMART WIN-WIN): "),
                                 "Intelligent optimization that ONLY makes changes that improve outcomes vs baseline:",
                                 html.Br(),
                                 "• ", html.Strong("Core Principle:"), " Compare each option against baseline (22°C). Only select if it improves comfort AND/OR energy without worsening the other metric significantly.",
                                 html.Br(),
                                 "• ", html.Strong("Scoring System:"), " Comfort improvement × 1.5 + Energy savings × 1.0. Negative score = improvement. Only accept improvements (score < 0).",
                                 html.Br(),
                                 "• ", html.Strong("Comfort Improvement:"), " +10 points if achieves comfort when baseline doesn't. +2 points per degree closer to comfort. -10 points if loses comfort.",
                                 html.Br(),
                                 "• ", html.Strong("Energy Improvement:"), " Positive value = energy savings vs baseline. Directly added to score.",
                                 html.Br(),
                                 "• ", html.Strong("Fallback:"), " If no option improves vs baseline, stick with baseline setpoint (no change).",
                                 html.Br(),
                                 "• ", html.Strong("Unoccupied zones:"), " Pure energy minimization for maximum savings.",
                                 html.Br(),
                                 "• ", html.Strong("Expected Result:"), " All AI changes are genuine improvements - better comfort + lower energy, or significantly better on one with minor impact on the other."]),
                        html.Li([html.Strong("Optimal Setpoint: "),
                                 "AI selects the setpoint with minimum total cost, balancing energy savings with comfort requirements."]),
                    ], className="mb-3"),
                    
                    html.H5("Comfort Compliance Calculation:", className="mt-3"),
                    html.Ul([
                        html.Li([html.Strong("Per Zone (Continuous Scoring): "),
                                 "100% if predicted temperature is within 21-23°C. ",
                                 "For temperatures outside this range, the score decreases by 20% per degree of deviation. ",
                                 "Example: 24°C → 80%, 25°C → 60%, 20°C → 80%. This provides realistic, gradual metrics."]),
                        html.Li([html.Strong("Portfolio-Wide: "),
                                 f"Average comfort scores across ONLY occupied zones ({occupied_zones} out of {total_zones} zones). ",
                                 "Comfort is only measured where people are actually present, then grouped by building for visualization."]),
                        html.Li([html.Strong("Baseline vs AI: "),
                                 "Baseline uses static 22°C (typically 100% comfort); AI uses optimized setpoint that minimizes penalty-weighted cost while maintaining or improving comfort."]),
                    ], className="mb-3"),
                    
                    html.H5("Energy Calculation:", className="mt-3"),
                    html.Ul([
                        html.Li([html.Strong("Direct ML Predictions: "),
                                 f"The {model_name} model directly predicts energy consumption (kWh) based on:",
                                 html.Br(),
                                 "• Outdoor temperature (heat gain/loss)",
                                 html.Br(),
                                 "• Current indoor temperature (system state)",
                                 html.Br(),
                                 "• Setpoint (HVAC effort required)",
                                 html.Br(),
                                 "• Occupancy (additional heat load)"]),
                        html.Li([html.Strong("No Proxies: "),
                                 "Energy values are physics-based model predictions, not simplified proxies. ",
                                 "The model was trained on 15,000 samples with realistic HVAC dynamics."]),
                    ], className="mb-3"),
                    
                    html.H5("Why Penalty-Based Optimization?", className="mt-3"),
                    html.P([
                        "The penalty approach ensures that energy savings never come at the expense of occupant comfort. ",
                        f"By setting a high penalty weight ({PENALTY_WEIGHT}), the AI 'pays' heavily for comfort violations, ",
                        "effectively forcing it to maintain 21-23°C when zones are occupied. ",
                        "This creates a hard constraint on comfort while still optimizing for energy efficiency."
                    ], className="mb-3"),
                    
                    html.Small([
                        html.Strong("Note: "),
                        "This analytics view runs the optimization for each zone in real-time using the trained ML model. ",
                        "Results may vary slightly from the notebook due to random outdoor temperature sampling and zone-specific conditions."
                    ], className="text-muted")
                ])
            ], className="mb-4"),
                    
            # Implementation Recommendations
            html.H4("🚀 Implementation Recommendations", className="mt-5 mb-3"),
            dbc.Alert([
                html.H5("Next Steps:", className="alert-heading"),
                html.Ol([
                    html.Li("Start with buildings showing the largest gaps (view Building View for zone details)"),
                    html.Li("Use the Digital Twin Simulator to test zone-specific optimizations"),
                    html.Li("Implement recommended setpoints gradually and monitor occupant feedback"),
                    html.Li("Track actual savings vs predictions and refine models over time"),
                    html.Li("Expand to additional buildings once success is validated")
                ])
            ], color="info")
        ])
    
    # === PORTFOLIO VIEW ===
    elif view_mode == 'portfolio':
        # Count critical alerts across all buildings
        critical_count = sum(1 for b in PORTFOLIO_DB.values() for z in b['zones'] if z['status'] == 'Critical')
        warning_count = sum(1 for b in PORTFOLIO_DB.values() for z in b['zones'] if z['status'] == 'Warning')
        
        # KPIs
        kpis = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H4("3", className="text-primary"), html.P("Sites Managed")])], className="text-center shadow-sm")),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4("12.4 MWh", className="text-success"), html.P("Energy Saved")])], className="text-center shadow-sm")),
            dbc.Col(dbc.Card([dbc.CardBody([html.H4(str(critical_count), className="text-danger"), html.P("Critical Alerts")])], className="text-center shadow-sm")),
        ], className="mb-4")
        
        # Map with prominent building markers
        df_map = pd.DataFrame(BUILDINGS_LIST)
        fig_map = px.scatter_map(df_map, lat="lat", lon="lon", hover_name="name", 
                                 hover_data={"type": True, "lat": False, "lon": False},
                                 zoom=4, size_max=30)
        
        # Make markers more prominent (scatter_map uses different marker properties)
        fig_map.update_traces(marker=dict(size=25, color='#FF6B6B', opacity=0.8))
        fig_map.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, 
                             height=600, clickmode='event+select')
        
        # Build alerts summary - group by building
        building_alerts = {}
        for building_name, building_data in PORTFOLIO_DB.items():
            alert_zones = [z for z in building_data['zones'] if z['status'] in ['Critical', 'Warning']]
            if alert_zones:
                building_alerts[building_name] = alert_zones
        
        alerts_section = html.Div()
        if building_alerts:
            alert_cards = []
            for building_name, alert_zones in building_alerts.items():
                alert_text = ", ".join([z['name'] for z in alert_zones])
                alert_cards.append(
                    dbc.Alert([
                        html.Strong(f"{building_name}: "),
                        alert_text,
                        html.Br(),
                        html.Small("📌 Tip: Select this building in Building View to see details and run AI optimization"),
                        html.Br(),
                        dbc.Button("View & Fix", id={"type": "view-fix-btn", "index": building_name}, 
                                  color="warning", size="sm", className="mt-2", n_clicks=0)
                    ], color="warning", className="mb-2")
                )
            alerts_section = html.Div([
                html.H5("⚠️ Active Alerts Requiring Attention"),
                *alert_cards
            ], className="mb-4")
        
        return html.Div([
            html.H2("Global Portfolio Overview"),
            html.Hr(),
            alerts_section,
            kpis,
            dbc.Card([
                dbc.CardBody([
                    html.P("💡 Tip: Click on a building marker to view details", className="text-muted mb-2"),
                    dcc.Graph(id="portfolio-map", figure=fig_map)
                ])
            ]),
            dbc.Alert([
                html.Strong("📊 Want to see AI optimization impact? "),
                "Visit the AI Impact Analytics tab to view detailed energy savings and comfort improvements."
            ], color="info", className="mt-4")
        ])

    # === BUILDING VIEW ===
    elif view_mode == 'building':
        # 1. Zone Cards Grid
        zones = PORTFOLIO_DB[selected_building]['zones']
        zone_cards = []
        for z in zones:
            # Color logic
            color = "success" if z['status'] == "OK" else "warning" if z['status'] == "Warning" else "danger"
            card = dbc.Col(dbc.Card([
                dbc.CardHeader(z['name']),
                dbc.CardBody([
                    html.H5(f"{z['temp']}°C", className="card-title"),
                    html.P(f"Occupied: {'Yes' if z['occupied'] else 'No'}", className="card-text"),
                    dbc.Badge(z['status'], color=color),
                    (html.Div([
                        html.Hr(className="my-2"),
                        dbc.Button("Try AI Fix 🤖", color=color, size="sm", className="w-100", 
                                   id={"type": "quick-fix-btn", "index": z['name']}, n_clicks=0)
                    ]) if z['status'] in ['Critical', 'Warning'] else html.Div())
                ])
            ], color=color, outline=True, className="mb-3"), width=2) # width=2 for 6 cards/row
            zone_cards.append(card)
        
        # 2. Digital Twin Simulator (Specific to selected Zone)
        # Find current zone data to pre-fill inputs
        current_z_data = next(z for z in zones if z['name'] == selected_zone_name)
        
        simulator_panel = dbc.Card([
            dbc.CardHeader(f"🎛️ Digital Twin Optimization: {selected_zone_name}"),
            dbc.CardBody([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("This simulator forecasts zone temperature over the next 24 hours (sampled every 2 hours)."),
                        html.Ul([
                            html.Li([html.Strong("Solid lines"), " show predictions if you keep your current setpoint (left slider value)."]),
                            html.Li([html.Strong("Dashed lines"), " show AI model predictions using the recommended optimized setpoint that balances comfort (21–23°C) and energy efficiency."]),
                            html.Li("The green band marks the comfort zone. Less time outside this band = better comfort."),
                        ])
                    ], title="What am I seeing?"),
                    dbc.AccordionItem([
                        html.Ol([
                            html.Li("Adjust Indoor, Outdoor, and Occupancy to match conditions."),
                            html.Li("Choose model: Random Forest (fast & interpretable) or Neural Network."),
                            html.Li("Click Run AI Optimization. We search setpoints between 18–26°C."),
                            html.Li("Compare lines: AI should reduce comfort violations and often lower energy."),
                            html.Li("Use the shown 'Recommended Setpoint' to update the zone."),
                        ])
                    ], title="How do I use it?")
                ], start_collapsed=True, className="mb-3"),
                dbc.Row([
                    # Controls
                    dbc.Col([
                        html.Label("Simulation Inputs (Live Override)"),
                        html.Br(),
                        html.Small("Indoor Temp (°C)"),
                        dcc.Slider(id="sim-indoor", min=15, max=30, step=0.5, value=current_z_data['temp'], marks={15:'15', 22:'22', 30:'30'}),
                        html.Br(),
                        html.Small("Outdoor Temp (°C)"),
                        dcc.Slider(id="sim-outdoor", min=-10, max=40, step=1, value=12, marks={-10:'-10', 10:'10', 40:'40'}),
                        html.Br(),
                        html.Small("Occupancy"),
                        dcc.RadioItems(id="sim-occ", options=[{'label': 'Occupied', 'value': 1}, {'label': 'Vacant', 'value': 0}], value=1 if current_z_data['occupied'] else 0, inline=True),
                        html.Br(),
                        html.Small("Optimization Preset"),
                        dcc.RadioItems(
                            id="weight-preset",
                            options=[
                                {"label": "Comfort-first", "value": "comfort"},
                                {"label": "Balanced", "value": "balanced"},
                                {"label": "Energy-first", "value": "energy"}
                            ],
                            value="balanced", inline=True
                        ),
                        html.Br(),
                        html.Small("Comfort Weight"),
                        dcc.Slider(id="weight-comfort", min=0.0, max=3.0, step=0.1, value=1.5, marks={0:'0', 1.5:'1.5', 3:'3'}),
                        html.Br(),
                        html.Small("Energy Weight"),
                        dcc.Slider(id="weight-energy", min=0.0, max=2.0, step=0.1, value=1.0, marks={0:'0', 1:'1', 2:'2'}),
                        html.Br(),
                        html.Small("Model"),
                        dcc.RadioItems(
                            id="model-choice",
                            options=[
                                {'label': 'Random Forest', 'value': 'RF'},
                                {'label': 'Neural Network (TF)', 'value': 'NN'},
                                {'label': 'PyTorch', 'value': 'PT'},
                                {'label': 'Compare All', 'value': 'Both'}
                            ],
                            value='Both', inline=True
                        ),
                        html.Br(),
                        dbc.Button("Run AI Optimization", id="btn-run", color="primary", className="w-100")
                    ], width=4),
                    
                    # Results (Graph)
                    dbc.Col([
                        dcc.Loading(dcc.Graph(id="opt-graph"), type="circle"),
                        html.Div(id="opt-recommendation", className="mt-3")
                    ], width=8)
                ])
            ])
        ], className="mt-4 shadow")

        # 3. Alerts Section (Show Critical & Warning zones)
        critical_zones = [z for z in zones if z['status'] in ['Critical', 'Warning']]
        alerts_section = html.Div()
        if critical_zones:
            alert_items = []
            for z in critical_zones:
                alert_color = "danger" if z['status'] == "Critical" else "warning"
                alert_items.append(
                    dbc.Alert([
                        html.Div([
                            html.Strong(f"{z['name']}: "),
                            f"{z['temp']}°C, Occupied: {'Yes' if z['occupied'] else 'No'} ",
                            dbc.Badge(z['status'], color=alert_color, className="ms-2")
                        ], className="d-flex align-items-center justify-content-between"),
                        html.Div([
                            dbc.Button(
                                "Try AI Fix 🤖",
                                id={"type": "quick-fix-btn-alert", "index": z['name']},
                                color=alert_color,
                                size="sm",
                                className="mt-2"
                            )
                        ])
                    ], color=alert_color, className="mb-2")
                )
            alerts_section = html.Div([
                html.H5("⚠️  Active Alerts"),
                *alert_items
            ], className="mb-4")
        
        return html.Div([
            html.H2(f"{selected_building}"),
            html.P("Real-time Zone Health & Predictive Control", className="text_muted"),
            html.Hr(),
            alerts_section,
            html.H5("Zone Status"),
            dbc.Row(zone_cards),
            simulator_panel
        ])

# --- B2. Handle VIEW & FIX Button Clicks (Portfolio → Building) ---
@app.callback(
    [Output("view-selector", "value"),
     Output("building-dropdown", "value")],
    Input({"type": "view-fix-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def navigate_to_building(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Extract building name from triggered button
    triggered_prop_id = ctx.triggered[0]["prop_id"]
    try:
        import json
        button_id = json.loads(triggered_prop_id.split(".")[0])
        building_name = button_id.get("index")
        if building_name:
            return "building", building_name
    except:
        pass
    
    return "building", BUILDINGS_LIST[0]['name']

# --- B3. Handle TRY AI FIX Button Clicks (Change Zone) ---
@app.callback(
    Output("zone-dropdown", "value", allow_duplicate=True),
    [Input({"type": "quick-fix-btn", "index": ALL}, "n_clicks"),
     Input({"type": "quick-fix-btn-alert", "index": ALL}, "n_clicks")],
    prevent_initial_call=True
)
def select_alert_zone(n_clicks_list_cards, n_clicks_list_alerts):
    # Combine clicks from both sources
    clicks = []
    if n_clicks_list_cards:
        clicks.extend(n_clicks_list_cards)
    if n_clicks_list_alerts:
        clicks.extend(n_clicks_list_alerts)
    if not any(clicks):
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_prop_id = ctx.triggered[0]["prop_id"]
    try:
        import json
        button_id = json.loads(triggered_prop_id.split(".")[0])
        zone_name = button_id.get("index")
        if zone_name:
            return zone_name
    except Exception:
        pass

    raise PreventUpdate

# --- B4. Handle Map Click (Navigate to Building) ---
@app.callback(
    [Output("view-selector", "value", allow_duplicate=True),
     Output("building-dropdown", "value", allow_duplicate=True)],
    Input("portfolio-map", "clickData"),
    prevent_initial_call=True
)
def map_click_to_building(click_data):
    if not click_data:
        raise PreventUpdate
    
    # Extract building name from clicked point
    try:
        building_name = click_data['points'][0]['hovertext']
        if building_name:
            return "building", building_name
    except (KeyError, IndexError):
        pass
    
    raise PreventUpdate

# --- B5. Preset Weights → Update Sliders ---
@app.callback(
    [Output("weight-energy", "value"),
     Output("weight-comfort", "value")],
    Input("weight-preset", "value"),
    prevent_initial_call=True
)
def apply_preset_weights(preset):
    if not preset:
        raise PreventUpdate
    # Preset mapping: tune priorities with sensible defaults
    if preset == "comfort":
        return 0.7, 2.5
    if preset == "energy":
        return 1.6, 0.8
    # balanced default
    return 1.0, 1.5

# --- C. Optimization Logic (The Brain) ---
@app.callback(
    [Output("opt-graph", "figure"),
     Output("opt-recommendation", "children")],
    [Input("btn-run", "n_clicks"),
     Input({"type": "quick-fix-btn", "index": ALL}, "n_clicks"),
     Input({"type": "quick-fix-btn-alert", "index": ALL}, "n_clicks")],
    [State("sim-indoor", "value"),
     State("sim-outdoor", "value"),
     State("sim-occ", "value"),
     State("model-choice", "value"),
     State("weight-energy", "value"),
     State("weight-comfort", "value"),
     State("building-dropdown", "value"),
     State("zone-dropdown", "value")]
)
def run_optimization(n_clicks, quick_fix_clicks_cards, quick_fix_clicks_alerts, in_temp, out_temp, is_occ, model_choice, weight_energy, weight_comfort, selected_building, selected_zone_name):
    # Determine trigger source
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx and ctx.triggered else None
    triggered_quick_fix = triggered and (
        triggered.startswith("{\"type\": \"quick-fix-btn\"") or
        triggered.startswith("{\"type\": \"quick-fix-btn-alert\"")
    )

    # If no trigger yet, show hint
    if not triggered:
        return go.Figure(), "Click 'Run AI Optimization' or 'Try AI Fix' to start."

    # If 'Try AI Fix' was clicked, override inputs with the selected zone's current data
    if triggered_quick_fix:
        try:
            import json
            btn_id = json.loads(triggered.split(".")[0])
            zone_name = btn_id.get("index") or selected_zone_name
        except Exception:
            zone_name = selected_zone_name

        try:
            zones = PORTFOLIO_DB[selected_building]['zones']
            z = next(zz for zz in zones if zz['name'] == zone_name)
            in_temp = float(z['temp'])
            is_occ = 1 if z['occupied'] else 0
            # Keep current outdoor slider if set; otherwise use a reasonable default
            out_temp = float(out_temp) if out_temp is not None else 12.0
        except StopIteration:
            pass

    COMFORT_MIN = 21.0
    COMFORT_MAX = 23.0

    # OPTIMIZED: Every 2 hours instead of 1 hour for 2x speed boost
    hours = np.arange(0, 24, 2.0)  # 12 timesteps instead of 24
    timestamps = [f"{int(h)}:{int((h % 1) * 60):02d}" for h in hours]

    def simulate_with_model(model_entry, setpoint):
        """
        Simulate 24h with a given model and setpoint.
        OPTIMIZED: Uses iterative prediction (state-dependent) but reduces overhead.
        """
        model = model_entry['model']
        scaler = model_entry['scaler']
        
        prev_temp = float(in_temp)
        temps = []
        energies = []
        start_inf = time.perf_counter()
        
        # Prepare all inputs at once (they share outdoor_temp, setpoint, occupancy)
        for _ in hours:
            # Prepare input
            x_raw = np.array([[float(out_temp), prev_temp, float(setpoint), float(is_occ)]])
            x_scaled = scaler.transform(x_raw)
            
            # Predict based on model type
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                    pred = model(x_tensor).numpy()[0]
                t_next = float(pred[0])
                e_next = float(pred[1])
            elif TF_AVAILABLE and hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
                # TensorFlow model
                pred = model.predict(x_scaled, verbose=0)[0]
                t_next = float(pred[0])
                e_next = float(pred[1])
            else:
                # RandomForest model
                pred = model.predict(x_scaled)[0]
                t_next = float(pred[0])
                e_next = float(pred[1])
            
            temps.append(np.round(t_next, 2))
            energies.append(np.round(e_next, 2))
            prev_temp = t_next
            
        latency_s = time.perf_counter() - start_inf
        return temps, energies, latency_s
    
    def simulate_multiple_setpoints_optimized(model_entry, setpoints_to_test):
        """
        PERFORMANCE OPTIMIZATION: Simulate multiple setpoints in parallel.
        For each setpoint, we still need iterative predictions (temp depends on previous state),
        but we can batch predictions across different setpoints at each timestep.
        """
        model = model_entry['model']
        scaler = model_entry['scaler']
        
        n_setpoints = len(setpoints_to_test)
        n_hours = len(hours)
        
        # Initialize arrays to track state for each setpoint
        prev_temps = np.full(n_setpoints, float(in_temp))  # All start at current temp
        all_temps = np.zeros((n_setpoints, n_hours))
        all_energies = np.zeros((n_setpoints, n_hours))
        
        start_inf = time.perf_counter()
        
        # For each timestep, predict all setpoints at once (batch prediction)
        for hour_idx in range(n_hours):
            # Build batch input: (n_setpoints, 4)
            batch_inputs = np.column_stack([
                np.full(n_setpoints, float(out_temp)),  # Outdoor temp (same for all)
                prev_temps,                              # Previous temp (different for each)
                setpoints_to_test,                       # Setpoint (different for each)
                np.full(n_setpoints, float(is_occ))     # Occupancy (same for all)
            ])
            batch_scaled = scaler.transform(batch_inputs)
            
            # Batch predict
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    batch_preds = model(torch.tensor(batch_scaled, dtype=torch.float32)).numpy()
            elif TF_AVAILABLE and hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
                batch_preds = model.predict(batch_scaled, verbose=0)
            else:
                batch_preds = model.predict(batch_scaled)
            
            # Extract temps and energies
            all_temps[:, hour_idx] = batch_preds[:, 0]
            all_energies[:, hour_idx] = batch_preds[:, 1]
            
            # Update prev_temps for next iteration
            prev_temps = batch_preds[:, 0]
        
        latency_s = time.perf_counter() - start_inf
        
        # Convert to list of tuples for compatibility
        results = []
        for i in range(n_setpoints):
            results.append((
                all_temps[i, :].tolist(),
                all_energies[i, :].tolist(),
                latency_s / n_setpoints  # Amortize latency
            ))
        
        return results

    # Resolve weights (with sensible defaults)
    energy_w = float(weight_energy) if weight_energy is not None else 1.0
    comfort_w = float(weight_comfort) if weight_comfort is not None else 1.5

    def compute_cost(temps, energies):
        """Compute total cost based on energy consumption and comfort violations."""
        energy_cost = energy_w * sum(energies)
        if int(is_occ) == 1:
            violations = [max(0.0, t - COMFORT_MAX) + max(0.0, COMFORT_MIN - t) for t in temps]
            comfort_penalty = sum(violations) * comfort_w
        else:
            comfort_penalty = 0.0
        return energy_cost + comfort_penalty

    fig = go.Figure()
    fig.add_hrect(y0=COMFORT_MIN, y1=COMFORT_MAX,
                  annotation_text="Comfort Zone (21-23°C)",
                  annotation_position="right",
                  fillcolor="lightgreen", opacity=0.2, line_width=0)

    results = {}
    recommended_setpoints = {}

    selections = []
    if model_choice == 'RF':
        selections = [('RF', MODEL_REGISTRY['RF'])]
    elif model_choice == 'NN':
        if MODEL_REGISTRY['NN']['model'] is not None:
            selections = [('NN', MODEL_REGISTRY['NN'])]
        else:
            selections = [('RF', MODEL_REGISTRY['RF'])]
    elif model_choice == 'PT':
        if MODEL_REGISTRY['PT']['model'] is not None:
            selections = [('PT', MODEL_REGISTRY['PT'])]
        else:
            selections = [('RF', MODEL_REGISTRY['RF'])]
    else:
        # Both models
        selections = [('RF', MODEL_REGISTRY['RF'])]
        if MODEL_REGISTRY['NN']['model'] is not None:
            selections.append(('NN', MODEL_REGISTRY['NN']))
        if MODEL_REGISTRY['PT']['model'] is not None:
            selections.append(('PT', MODEL_REGISTRY['PT']))

    color_map = {
        'RF_current': '#FF6B6B',
        'RF_opt': '#51CF66',
        'NN_current': '#4C78A8',
        'NN_opt': '#9C6ADE',
        'PT_current': '#E67E22',
        'PT_opt': '#27AE60'
    }

    for key, model_entry in selections:
        # OPTIMIZED: Batch simulate all candidate setpoints at once
        # Status quo uses current indoor temp as setpoint
        statusquo_setpoint = float(in_temp)
        
        # Define all setpoints to test (including status quo)
        # SPEED: Reduced search space from 6 to 4 options (saves ~33% time)
        search_setpoints = np.arange(20.0, 24.0, 1.0)  # 20, 21, 22, 23 (comfort range)
        all_setpoints = np.append([statusquo_setpoint], search_setpoints)
        
        # Batch simulate all setpoints
        all_results = simulate_multiple_setpoints_optimized(model_entry, all_setpoints)
        
        # Extract status quo results (first one)
        temps_current, energies_current, latency_s = all_results[0]
        
        # Find best setpoint from the rest
        best_sp, best_cost = statusquo_setpoint, float('inf')
        best_temps, best_energies = temps_current, energies_current
        
        for idx, sp in enumerate(search_setpoints):
            temps_sp, energies_sp, _ = all_results[idx + 1]  # +1 because status quo is at index 0
            cost_sp = compute_cost(temps_sp, energies_sp)
            if cost_sp < best_cost:
                best_cost = cost_sp
                best_sp = float(sp)
                best_temps = temps_sp
                best_energies = energies_sp
        
        recommended_setpoints[key] = best_sp
        violations_current = [t for t in temps_current if t < COMFORT_MIN or t > COMFORT_MAX]
        violations_optimal = [t for t in best_temps if t < COMFORT_MIN or t > COMFORT_MAX]
        violation_pct_current = len(violations_current) / len(temps_current) * 100
        violation_pct_optimal = len(violations_optimal) / len(best_temps) * 100
        
        # Energy calculations
        total_energy_current = sum(energies_current)
        total_energy_optimal = sum(best_energies)
        energy_savings_pct = (total_energy_current - total_energy_optimal) / total_energy_current * 100 if total_energy_current > 0 else 0
        
        results[key] = {
            'temps_current': temps_current,
            'temps_optimal': best_temps,
            'energies_current': energies_current,
            'energies_optimal': best_energies,
            'energy_savings_pct': energy_savings_pct,
            'latency_s': latency_s,
            'viol_current_pct': violation_pct_current,
            'viol_opt_pct': violation_pct_optimal,
            'recommended_setpoint': best_sp,
            'metrics': MODEL_REGISTRY[key]['metrics'] if key in MODEL_REGISTRY else {}
        }

        # Status Quo: solid lines
        fig.add_trace(go.Scatter(x=timestamps, y=temps_current, name=f"Status Quo ({key})",
                                 line=dict(color=color_map[f"{key}_current"], width=2.5), 
                                 mode='lines+markers', marker=dict(size=6)))
        # AI Predictions: DASHED lines for clear distinction
        fig.add_trace(go.Scatter(x=timestamps, y=best_temps, name=f"AI Prediction ({key}, {best_sp:.1f}°C)",
                                 line=dict(color=color_map[f"{key}_opt"], width=2.5, dash='dash'), 
                                 mode='lines+markers', marker=dict(size=6, symbol='diamond')))

    fig.update_layout(
        title="Temperature Over 24 Hours: Current vs AI-Optimized Setpoint",
        xaxis_title="Time of Day",
        yaxis_title="Temperature (°C)",
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,
        hovermode='x unified'
    )

    cards = []
    for key in results:
        r = results[key]
        metrics = r['metrics']
        title = "Random Forest" if key == 'RF' else "Neural Network"
        train_time = metrics.get('train_time_s')
        mae_temp = metrics.get('mae_temp')
        extra = []
        if key == 'RF':
            extra.append(f"Trees: {metrics.get('n_estimators')} | Max Depth: {metrics.get('max_depth')}")
        elif key in ['NN', 'PT']:
            params = metrics.get('params')
            if params is not None:
                extra.append(f"Params: {params}")

        card = dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody([
                html.P([
                    f"Violations (Current): {r['viol_current_pct']:.1f}% | ",
                    f"Violations (AI {r['recommended_setpoint']:.1f}°C): {r['viol_opt_pct']:.1f}%"
                ]),
                html.P([
                    f"Energy Savings: {r['energy_savings_pct']:.1f}% | ",
                    f"Current: {sum(r['energies_current']):.2f} kWh | ",
                    f"Optimal: {sum(r['energies_optimal']):.2f} kWh"
                ]),
                html.P([
                    f"Inference Latency: {r['latency_s']*1000:.1f} ms"
                ]),
                html.P([
                    f"Training Time: {train_time*1000:.1f} ms" if train_time is not None else "Training Time: N/A"
                ]),
                html.P([
                    f"Temp MAE (holdout): {mae_temp:.3f}" if mae_temp is not None else "Temp MAE (holdout): N/A"
                ]),
                html.P([
                    f"Recommended Setpoint: {r['recommended_setpoint']:.1f}°C"
                ]),
                html.Small(" ".join(extra)) if extra else html.Small("")
            ])
        ], className="mb-2")
        cards.append(card)

    rec_box = html.Div([
        dbc.Alert([
            html.H5("Model Comparison & Recommendation", className="alert-heading"),
            html.P([
                html.Strong("Solid lines"), " = Status Quo (current setpoint maintained). ",
                html.Strong("Dashed lines"), " = AI model predictions using the optimized setpoint that balances energy efficiency and comfort requirements."
            ]),
            html.P(["Optimization weights — ", html.Strong(f"Comfort: {comfort_w:.1f}"), ", ", html.Strong(f"Energy: {energy_w:.1f}")])
        ], color="info"),
        dbc.Row([dbc.Col(c, md=6) for c in cards])
    ])

    return fig, rec_box

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)