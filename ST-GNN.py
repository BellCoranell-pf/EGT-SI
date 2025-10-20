# -*- coding: utf-8 -*-
"""
ST-GNN vs Kriging Interpolation Performance Comparison Based on Real Meteorological Data
Input: yearavetemp.csv (year, station, lon, lat, elevation, temp)
Output: Training loss, validation/test RMSE comparison, visualization plots
✅ Functionality identical to original, only replaced with real data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Load real meteorological data
print("Loading real data...")
try:
    df_raw = pd.read_csv('yearavetemp.csv', encoding='utf-8')
except FileNotFoundError:
    raise FileNotFoundError("❌ Please ensure 'yearavetemp.csv' is in the current directory!")
except UnicodeDecodeError:
    df_raw = pd.read_csv('yearavetemp.csv', encoding='gbk')  # Try GBK encoding (common in Chinese systems)

# Rename columns (adapt common Chinese field names)
df_raw.rename(columns={
    '年份': 'year',
    '站点编号': 'station',
    '经度': 'lon',
    '纬度': 'lat',
    '高程': 'elevation',
    '年均气温': 'temp'
}, inplace=True)

# Check required columns
required_cols = ['year', 'station', 'lon', 'lat', 'elevation', 'temp']
for col in required_cols:
    if col not in df_raw.columns:
        raise ValueError(f"❌ Missing required column: {col}")

# Clean data
df_full = df_raw[required_cols].dropna().copy()
df_full = df_full[(df_full['temp'] > -60) & (df_full['temp'] < 50)]  # Filter abnormal temperatures
df_full['station'] = df_full['station'].astype(str)  # Ensure station ID is string
print(f"✅ Data loaded: {len(df_full)} records, covering {df_full['station'].nunique()} stations")

# 2. Extract station information
station_list = sorted(df_full['station'].unique())
years = sorted(df_full['year'].unique())

# Build station attribute dictionary (use first occurrence, assume constant)
station_info = df_full.groupby('station').first()[['lat', 'lon', 'elevation']].to_dict('index')
COORDS = {st: (info['lon'], info['lat']) for st, info in station_info.items()}
ELEVATIONS_DICT = {st: info['elevation'] for st, info in station_info.items()}
print(f"Year range: {min(years)} - {max(years)}")
print(f"Total stations: {len(station_list)}")

# 3. Select subset of stations for experiment (e.g., up to 321)
np.random.seed(42)
selected_stations = np.random.choice(station_list, size=min(321, len(station_list)), replace=False)
selected_stations = sorted(selected_stations)

# Filter data
df_small = df_full[df_full['station'].isin(selected_stations)].copy()

# Split into train/val/test stations
train_stations, temp_stations = train_test_split(selected_stations, test_size=0.3, random_state=42)
val_stations, test_stations = train_test_split(temp_stations, test_size=0.5, random_state=42)

# Mark dataset splits
df_small['split'] = 'train'
df_small.loc[df_small['station'].isin(val_stations), 'split'] = 'val'
df_small.loc[df_small['station'].isin(test_stations), 'split'] = 'test'

print(f"✅ Experiment setup:")
print(f"Total experimental stations: {len(selected_stations)}")
print(f"Train: {len(train_stations)} | Val: {len(val_stations)} | Test: {len(test_stations)}")

# 4. Build graph structure (based on selected stations)
selected_coords = {st: COORDS[st] for st in selected_stations}
selected_elevations = {st: ELEVATIONS_DICT[st] for st in selected_stations}
positions = np.array([selected_coords[st] for st in selected_stations])
dist_matrix = squareform(pdist(positions))

edge_index_list = []
edge_attr_list = []
idx_map = {st: i for i, st in enumerate(selected_stations)}  # Map station to index

for i in range(len(selected_stations)):
    for j in range(len(selected_stations)):
        if i != j:
            edge_index_list.append([i, j])
            edge_attr_list.append([1.0 / (dist_matrix[i, j] + 1e-8)])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_attr_list, dtype=torch.float32).squeeze(-1)

print(f"✅ Graph structure built:")
print(f"Number of nodes: {len(selected_stations)}")
print(f"Number of edges: {edge_index.shape[1]}")
print(f"Max node index: {edge_index.max().item()}")

# 5. Model Definition
class STGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x  # [N, out_channels]

class TempPredictor(nn.Module):
    def __init__(self, node_dim, extra_dim, hidden_dim=32):
        super(TempPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_node, extra_features):
        x = torch.cat([h_node, extra_features], dim=1)
        return self.mlp(x).squeeze(-1)

# 6. Kriging interpolation (baseline)
def kriging_predict(st, yr, df_data, coords):
    other_data = df_data[(df_data['year'] == yr) & (df_data['station'] != st)]
    if len(other_data) == 0:
        return df_data['temp'].mean()

    lons = other_data['lon'].values.tolist()
    lats = other_data['lat'].values.tolist()
    temps = other_data['temp'].values
    target_lon, target_lat = coords[st]

    try:
        ok = OrdinaryKriging(
            lons, lats, temps,
            variogram_model='linear',
            verbose=False,
            enable_plotting=False
        )
        pred, _ = ok.execute('points', [target_lon], [target_lat])
        return pred.item()
    except Exception as e:
        return np.mean(temps)

# 7. Build node features for a given year
def build_node_features_for_year(df_data, year, stations, elevations, coords):
    """Return feature matrix for all stations in the given year [num_nodes, 6]"""
    features = []
    for st in stations:
        data_st = df_data[(df_data['station'] == st) & (df_data['year'] == year)]
        avg_temp = data_st['temp'].mean() if len(data_st) > 0 else df_data['temp'].mean()
        elev = elevations[st]
        lon, lat = coords[st]
        # Normalize (using global statistics)
        lat_norm = (lat - 34.0) / 10.0  # Approximate range: 18-53
        lon_norm = (lon - 105.0) / 20.0  # Approximate range: 73-135
        elev_norm = (elev - 1000) / 1500
        features.append([avg_temp, elev_norm, lon_norm, lat_norm, elev, lat])
    return torch.tensor(features, dtype=torch.float32)

# 8. Training Preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Initialize models
encoder = STGCNEncoder(in_channels=6, hidden_channels=32, out_channels=16).to(device)
predictor = TempPredictor(node_dim=16, extra_dim=3).to(device)

# Optimizer and loss
optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-5)
criterion = nn.MSELoss()

# Training samples
train_samples = [(row['station'], row['year'], row['temp'])
                 for _, row in df_small[df_small['split'] == 'train'].iterrows()]
print(f"✅ Number of training samples: {len(train_samples)}")

# 9. Early stopping setup
patience = 5
best_val_rmse = float('inf')
counter = 0
best_model_saved = False

# Logging results
results = {'epoch': [], 'train_loss': [], 'val_rmse_gnn': [], 'val_rmse_kriging': []}

# 10. Main Training Loop (with early stopping)
num_epochs = 50
for epoch in range(num_epochs):
    # --- Training ---
    encoder.train()
    predictor.train()
    train_loss = 0.0
    train_count = 0

    for st, yr, temp_true in train_samples:
        x = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations, selected_coords)
        x = x.to(device)
        edge_weight_tensor = edge_weight.to(device)

        h_all = encoder(x, edge_index, edge_weight_tensor)
        st_idx = idx_map[st]
        h_st = h_all[st_idx:st_idx+1]

        elev_st = selected_elevations[st]
        lon_st, lat_st = selected_coords[st]
        extra_feat = torch.tensor([[elev_st, lon_st, lat_st]], dtype=torch.float32).to(device)

        pred = predictor(h_st, extra_feat)
        temp_true_tensor = torch.tensor([temp_true], dtype=torch.float32).to(device)

        loss = criterion(pred, temp_true_tensor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_count += 1

    train_loss /= train_count if train_count > 0 else 1

    # --- Validation ---
    encoder.eval()
    val_rmse_gnn = 0.0
    val_rmse_kriging = 0.0
    val_count = 0

    with torch.no_grad():
        for _, row in df_small[df_small['split'] == 'val'].iterrows():
            st, yr, temp_true = row['station'], row['year'], row['temp']

            # GNN prediction
            x = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations, selected_coords)
            x = x.to(device)
            h_all = encoder(x, edge_index, edge_weight.to(device))
            st_idx = idx_map[st]
            h_st = h_all[st_idx:st_idx+1]

            elev_st = selected_elevations[st]
            lon_st, lat_st = selected_coords[st]
            extra_feat = torch.tensor([[elev_st, lon_st, lat_st]], dtype=torch.float32).to(device)

            pred_gnn = predictor(h_st, extra_feat).item()
            val_rmse_gnn += (pred_gnn - temp_true) ** 2

            # Kriging prediction
            pred_kriging = kriging_predict(st, yr, df_small, selected_coords)
            val_rmse_kriging += (pred_kriging - temp_true) ** 2
            val_count += 1

    val_rmse_gnn = (val_rmse_gnn / val_count) ** 0.5 if val_count > 0 else 0
    val_rmse_kriging = (val_rmse_kriging / val_count) ** 0.5 if val_count > 0 else 0

    results['epoch'].append(epoch)
    results['train_loss'].append(train_loss)
    results['val_rmse_gnn'].append(val_rmse_gnn)
    results['val_rmse_kriging'].append(val_rmse_kriging)

    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Val RMSE (GNN): {val_rmse_gnn:.4f} | (Kriging): {val_rmse_kriging:.4f}")

    # Save log
    with open('log.csv', 'w') as f:
        f.write("Epoch,Loss,Val_RMSE_GNN,Val_RMSE_Kriging\n")
        f.write(f"{epoch+1},{train_loss:.4f},{val_rmse_gnn:.4f},{val_rmse_kriging:.4f}")

    # 11. Early stopping and save best model
    if val_rmse_gnn < best_val_rmse:
        best_val_rmse = val_rmse_gnn
        counter = 0
        torch.save(encoder.state_dict(), 'encoder_best.pth')
        torch.save(predictor.state_dict(), 'predictor_best.pth')
        best_model_saved = True
        print(f"New best model saved at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# 12. Load best model and test
if best_model_saved:
    print(f"\nLoading best model for final test (Val RMSE: {best_val_rmse:.4f})...")
    encoder.load_state_dict(torch.load('encoder_best.pth', map_location=device))
    predictor.load_state_dict(torch.load('predictor_best.pth', map_location=device))
    encoder.eval()

    test_rmse_gnn = 0.0
    test_rmse_kriging = 0.0
    test_count = 0

    with torch.no_grad():
        for _, row in df_small[df_small['split'] == 'test'].iterrows():
            st, yr, temp_true = row['station'], row['year'], row['temp']

            x = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations, selected_coords)
            x = x.to(device)
            h_all = encoder(x, edge_index, edge_weight.to(device))
            st_idx = idx_map[st]
            h_st = h_all[st_idx:st_idx+1]

            elev_st = selected_elevations[st]
            lon_st, lat_st = selected_coords[st]
            extra_feat = torch.tensor([[elev_st, lon_st, lat_st]], dtype=torch.float32).to(device)

            pred_gnn = predictor(h_st, extra_feat).item()
            test_rmse_gnn += (pred_gnn - temp_true) ** 2

            pred_kriging = kriging_predict(st, yr, df_small, selected_coords)
            test_rmse_kriging += (pred_kriging - temp_true) ** 2
            test_count += 1

    test_rmse_gnn = (test_rmse_gnn / test_count) ** 0.5 if test_count > 0 else 0
    test_rmse_kriging = (test_rmse_kriging / test_count) ** 0.5 if test_count > 0 else 0

    print(f"\n✅ Final Test RMSE:")
    print(f"ST-GNN: {test_rmse_gnn:.4f}")
    print(f"Kriging: {test_rmse_kriging:.4f}")

# 13. Result Visualization
plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['train_loss'], label='Train Loss', marker='x')
plt.plot(results['epoch'], results['val_rmse_gnn'], label='Val RMSE (GNN)', marker='o')
plt.plot(results['epoch'], results['val_rmse_kriging'], label='Val RMSE (Kriging)', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss / RMSE')
plt.title('ST-GNN vs Kriging - Real Meteorological Data Experiment')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

print("Experiment completed!")