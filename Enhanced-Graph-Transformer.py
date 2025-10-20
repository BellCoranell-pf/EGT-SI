# -*- coding: utf-8 -*-
"""
Enhanced ST-GNN: Spatial Interpolation of Meteorological Data using Graph Transformer (Full Runnable Version)
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# 1. Load real meteorological data
print("Loading real data...")
try:
    df_raw = pd.read_csv('yearavetemp.csv', encoding='utf-8')
except FileNotFoundError:
    raise FileNotFoundError("❌ Please ensure 'yearavetemp.csv' is in the current directory!")
except UnicodeDecodeError:
    df_raw = pd.read_csv('yearavetemp.csv', encoding='gbk')

df_raw.rename(columns={
    '年份': 'year',
    '站点编号': 'station',
    '经度': 'lon',
    '纬度': 'lat',
    '高程': 'elevation',
    '年均气温': 'temp'
}, inplace=True)

required_cols = ['year', 'station', 'lon', 'lat', 'elevation', 'temp']
for col in required_cols:
    if col not in df_raw.columns:
        raise ValueError(f"❌ Missing required column: {col}")

df_full = df_raw[required_cols].dropna().copy()
df_full = df_full[(df_full['temp'] > -60) & (df_full['temp'] < 50)]  # Filter abnormal values
df_full['station'] = df_full['station'].astype(str)
print(f"✅ Data loaded: {len(df_full)} records, covering {df_full['station'].nunique()} stations")

# 2. Extract station info
station_list = sorted(df_full['station'].unique())
years = sorted(df_full['year'].unique())
station_info = df_full.groupby('station').first()[['lat', 'lon', 'elevation']].to_dict('index')
COORDS = {st: (info['lon'], info['lat']) for st, info in station_info.items()}
ELEVATIONS_DICT = {st: info['elevation'] for st, info in station_info.items()}
print(f"Year range: {min(years)} - {max(years)}")
print(f"Total stations: {len(station_list)}")

# 3. Select subset of stations
np.random.seed(42)
selected_stations = np.random.choice(station_list, size=min(321, len(station_list)), replace=False)
selected_stations = sorted(selected_stations)
df_small = df_full[df_full['station'].isin(selected_stations)].copy()

train_stations, temp_stations = train_test_split(selected_stations, test_size=0.3, random_state=42)
val_stations, test_stations = train_test_split(temp_stations, test_size=0.5, random_state=42)

df_small['split'] = 'train'
df_small.loc[df_small['station'].isin(val_stations), 'split'] = 'val'
df_small.loc[df_small['station'].isin(test_stations), 'split'] = 'test'

print(f"✅ Experiment setup:")
print(f"Total experimental stations: {len(selected_stations)}")
print(f"Train: {len(train_stations)} | Val: {len(val_stations)} | Test: {len(test_stations)}")

# 4. Build fixed global graph
selected_coords = {st: COORDS[st] for st in selected_stations}
selected_elevations = {st: ELEVATIONS_DICT[st] for st in selected_stations}
positions = np.array([selected_coords[st] for st in selected_stations])
dist_matrix = squareform(pdist(positions))

edge_index_list = []
idx_map = {st: i for i, st in enumerate(selected_stations)}
for i in range(len(selected_stations)):
    for j in range(len(selected_stations)):
        if i != j:
            edge_index_list.append([i, j])
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
print(f"✅ Graph built: {len(selected_stations)} nodes, {edge_index.shape[1]} edges")

# 5. Year-to-index mapping
all_years = sorted(df_small['year'].unique())
year_to_idx = {year: idx for idx, year in enumerate(all_years)}
num_year_classes = len(all_years)
print(
    f"Year mapping: {min(all_years)} → 0, ..., {max(all_years)} → {num_year_classes - 1} ({num_year_classes} classes)")


# 6. Graph Transformer Encoder
class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_channels=16, num_layers=2, nhead=4,
                 dropout=0.1, num_years=50):
        super(GraphTransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.year_embedding = nn.Embedding(num_years, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim)

        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, year=None, coords=None):
        N = x.size(0)
        h = self.input_proj(x)  # [N, hidden_dim]

        # Year embedding
        if year is not None:
            if year in year_to_idx:
                year_idx = torch.tensor([year_to_idx[year]], device=x.device, dtype=torch.long)
            else:
                year_idx = torch.tensor([0], device=x.device, dtype=torch.long)  # default
            year_emb = self.year_embedding(year_idx)  # [1, hidden_dim]
            h += year_emb.expand(N, -1)  # [N, hidden_dim]

        # Coordinate positional encoding
        if coords is not None:
            coord_emb = self.coord_proj(coords)  # [N, hidden_dim]
            h += coord_emb

        h = self.transformer(h.unsqueeze(0)).squeeze(0)  # [N, hidden_dim]
        h_out = self.output_proj(h)  # [N, out_channels]
        return h_out


# 7. Temperature Predictor (MLP head)
class TempPredictor(nn.Module):
    def __init__(self, node_dim, extra_dim, hidden_dim=32, dropout=0.1):
        super(TempPredictor, self).__init__()
        self.fc1 = nn.Linear(node_dim + extra_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h_node, extra_features):
        x = torch.cat([h_node, extra_features], dim=1)
        x = self.fc1(x)
        if x.size(0) == 1 and self.training:
            self.bn.eval()
            x = self.bn(x)
            self.bn.train()
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# 8. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

encoder = GraphTransformerEncoder(
    in_channels=6,
    hidden_dim=64,
    out_channels=16,
    num_layers=2,
    nhead=4,
    num_years=num_year_classes
).to(device)

predictor = TempPredictor(node_dim=16, extra_dim=3).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
criterion = nn.MSELoss()

train_samples = [(row['station'], row['year'], row['temp'])
                 for _, row in df_small[df_small['split'] == 'train'].iterrows()]
print(f"✅ Number of training samples: {len(train_samples)}")

# 9. Early stopping
patience = 5
best_val_rmse = float('inf')
counter = 0
best_model_saved = False
results = {'epoch': [], 'train_loss': [], 'val_rmse_gnn': [], 'val_rmse_kriging': []}
num_epochs = 200


# 10. Kriging baseline
def kriging_predict(st, yr, df_data, coords):
    other_data = df_data[(df_data['year'] == yr) & (df_data['station'] != st)]
    if len(other_data) == 0:
        return df_data['temp'].mean()
    lons = other_data['lon'].values.tolist()
    lats = other_data['lat'].values.tolist()
    temps = other_data['temp'].values
    target_lon, target_lat = coords[st]
    try:
        ok = OrdinaryKriging(lons, lats, temps, variogram_model='linear', verbose=False, enable_plotting=False)
        pred, _ = ok.execute('points', [target_lon], [target_lat])
        return pred.item()
    except:
        return np.mean(temps)


# 11. Build node features (return x and coords tensor)
def build_node_features_for_year(df_data, year, stations, elevations, coords):
    features = []
    coord_list = []
    for st in stations:
        data_st = df_data[(df_data['station'] == st) & (df_data['year'] == year)]
        avg_temp = data_st['temp'].mean() if len(data_st) > 0 else df_data['temp'].mean()
        elev = elevations[st]
        lon, lat = coords[st]
        lat_norm = (lat - 34.0) / 10.0
        lon_norm = (lon - 105.0) / 20.0
        elev_norm = (elev - 1000) / 1500
        features.append([avg_temp, elev_norm, lon_norm, lat_norm, elev, lat])
        coord_list.append([lon, lat])
    x = torch.tensor(features, dtype=torch.float32)
    coords_tensor = torch.tensor(coord_list, dtype=torch.float32)
    return x, coords_tensor


# 12. Training loop
for epoch in range(num_epochs):
    # Training
    encoder.train()
    predictor.train()
    train_loss = 0.0
    train_count = 0
    for st, yr, temp_true in train_samples:
        x, coords_tensor = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations,
                                                        selected_coords)
        x, coords_tensor = x.to(device), coords_tensor.to(device)
        temp_true_tensor = torch.tensor([temp_true], dtype=torch.float32).to(device)

        h_all = encoder(x=x, year=yr, coords=coords_tensor)
        st_idx = idx_map[st]
        h_st = h_all[st_idx:st_idx + 1]
        elev_st = selected_elevations[st]
        lon_st, lat_st = selected_coords[st]
        extra_feat = torch.tensor([[elev_st, lon_st, lat_st]], dtype=torch.float32).to(device)

        pred = predictor(h_st, extra_feat)
        loss = criterion(pred, temp_true_tensor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_count += 1
    train_loss /= train_count if train_count > 0 else 1

    # Validation
    encoder.eval()
    val_rmse_gnn = 0.0
    val_rmse_kriging = 0.0
    val_count = 0
    with torch.no_grad():
        for _, row in df_small[df_small['split'] == 'val'].iterrows():
            st, yr, temp_true = row['station'], row['year'], row['temp']
            x, coords_tensor = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations,
                                                            selected_coords)
            x, coords_tensor = x.to(device), coords_tensor.to(device)
            h_all = encoder(x=x, year=yr, coords=coords_tensor)
            st_idx = idx_map[st]
            h_st = h_all[st_idx:st_idx + 1]
            elev_st = selected_elevations[st]
            lon_st, lat_st = selected_coords[st]
            extra_feat = torch.tensor([[elev_st, lon_st, lat_st]], dtype=torch.float32).to(device)
            pred_gnn = predictor(h_st, extra_feat).item()
            val_rmse_gnn += (pred_gnn - temp_true) ** 2
            pred_kriging = kriging_predict(st, yr, df_small, selected_coords)
            val_rmse_kriging += (pred_kriging - temp_true) ** 2
            val_count += 1
    val_rmse_gnn = (val_rmse_gnn / val_count) ** 0.5 if val_count > 0 else 0
    val_rmse_kriging = (val_rmse_kriging / val_count) ** 0.5 if val_count > 0 else 0

    results['epoch'].append(epoch)
    results['train_loss'].append(train_loss)
    results['val_rmse_gnn'].append(val_rmse_gnn)
    results['val_rmse_kriging'].append(val_rmse_kriging)

    print(
        f"Epoch {epoch + 1:2d} | Loss: {train_loss:.4f} | Val RMSE (Graph Transformer): {val_rmse_gnn:.4f} | (Kriging): {val_rmse_kriging:.4f}")

    # Early stopping
    if val_rmse_gnn < best_val_rmse:
        best_val_rmse = val_rmse_gnn
        counter = 0
        torch.save(encoder.state_dict(), 'encoder_best.pth')
        torch.save(predictor.state_dict(), 'predictor_best.pth')
        best_model_saved = True
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

# 13. Final Test
if best_model_saved and os.path.exists('encoder_best.pth') and os.path.exists('predictor_best.pth'):
    print(f"\nLoading best model for final test...")
    encoder.load_state_dict(torch.load('encoder_best.pth', map_location=device))
    predictor.load_state_dict(torch.load('predictor_best.pth', map_location=device))
    encoder.eval()

    test_rmse_gnn = 0.0
    test_rmse_kriging = 0.0
    test_count = 0
    with torch.no_grad():
        for _, row in df_small[df_small['split'] == 'test'].iterrows():
            st, yr, temp_true = row['station'], row['year'], row['temp']
            x, coords_tensor = build_node_features_for_year(df_small, yr, selected_stations, selected_elevations,
                                                            selected_coords)
            x, coords_tensor = x.to(device), coords_tensor.to(device)
            h_all = encoder(x=x, year=yr, coords=coords_tensor)
            st_idx = idx_map[st]
            h_st = h_all[st_idx:st_idx + 1]
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
    print(f"Graph Transformer: {test_rmse_gnn:.4f}")
    print(f"Kriging: {test_rmse_kriging:.4f}")

# 14. Visualization
plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['train_loss'], label='Train Loss', marker='x')
plt.plot(results['epoch'], results['val_rmse_gnn'], label='Val RMSE (Graph Transformer)', marker='o')
plt.plot(results['epoch'], results['val_rmse_kriging'], label='Val RMSE (Kriging)', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss / RMSE')
plt.title('Graph Transformer vs Kriging - Real Meteorological Data')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print("Experiment completed!")