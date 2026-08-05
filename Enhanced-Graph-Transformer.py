# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "yearavetemp.csv"
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = list(range(42, 52))
VAL_SIZE = 0.2

EGT_HIDDEN = 32
EGT_OUT = 8
EGT_LAYERS = 2
EGT_HEADS = 4
EGT_DROPOUT = 0.1063
EGT_LR = 9.855e-5
EGT_WEIGHT_DECAY = 1.100e-6
EGT_PATIENCE = 28
EGT_EPOCHS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    except:
        df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    df_raw.rename(columns={
        "年份": "year", "站点编号": "station", "经度": "lon",
        "纬度": "lat", "高程": "elevation", "年均气温": "temp"
    }, inplace=True)
    df = df_raw[["year", "station", "lon", "lat", "elevation", "temp"]].dropna().copy()
    df["year"] = df["year"].astype(int)
    df["station"] = df["station"].astype(str).str.strip()
    df = df[(df["temp"] > -60) & (df["temp"] < 80)]
    return df

df_full = load_data()
years_all = sorted(df_full["year"].unique())
station_year_counts = df_full.groupby("station")["year"].nunique()
common_stations = station_year_counts[station_year_counts == len(years_all)].index.tolist()
common_stations = sorted(common_stations)
df = df_full[df_full["station"].isin(common_stations)].copy()
station_info = df.groupby("station").first()[["lon", "lat", "elevation"]].to_dict("index")
station_list_all = common_stations
coords_all = {st: (station_info[st]["lon"], station_info[st]["lat"]) for st in station_list_all}
elev_all = {st: station_info[st]["elevation"] for st in station_list_all}

def split_stations_by_density(stations, density, seed):
    np.random.seed(seed)
    shuffled = np.random.permutation(stations)
    n = len(shuffled)
    n_val = int(n * density)
    n_test = int(n * density)
    n_train = n - n_val - n_test
    train_st = shuffled[:n_train]
    val_st = shuffled[n_train:n_train+n_val]
    test_st = shuffled[n_train+n_val:]
    return list(train_st), list(val_st), list(test_st)

def compute_climatology(df, target_year, station_list):
    clim = {}
    for st in station_list:
        sub = df[(df["station"] == st) & (df["year"] != target_year)]
        if len(sub) > 0:
            clim[st] = sub["temp"].mean()
        else:
            val = df[(df["station"] == st) & (df["year"] == target_year)]["temp"]
            clim[st] = val.values[0] if len(val) > 0 else np.nan
    global_clim = np.nanmean(list(clim.values()))
    for st in station_list:
        if np.isnan(clim[st]):
            clim[st] = global_clim
    return clim

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1_rad = np.radians(lon1); lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2); lat2_rad = np.radians(lat2)
    dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def idw_prior(target_station, obs_set, df_test_year, coords, power=2):
    target_lon, target_lat = coords[target_station]
    weights = []
    temps = []
    for st in obs_set:
        lon, lat = coords[st]
        row = df_test_year[df_test_year["station"] == st]
        if row.empty:
            continue
        temp = row["temp"].values[0]
        d = haversine(target_lon, target_lat, lon, lat)
        if d < 1e-6:
            return temp
        weights.append(1.0 / (d ** power))
        temps.append(temp)
    if not weights:
        return np.mean(temps) if temps else 0.0
    weights = np.array(weights)
    temps = np.array(temps)
    return np.sum(weights * temps) / np.sum(weights)

all_clim_vals = np.array([compute_climatology(df, years_all[0], station_list_all)[st] for st in station_list_all])
clim_mean_global = np.mean(all_clim_vals)
clim_std_global = np.std(all_clim_vals) + 1e-8

class EncoderWithAttn(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, out_channels=8,
                 num_layers=2, nhead=4, dropout=0.1,
                 use_year=True, use_coord=True, year_to_idx=None,
                 use_attn=True):
        super().__init__()
        self.use_year = use_year
        self.use_coord = use_coord
        self.year_to_idx = year_to_idx
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        if use_year:
            self.year_embedding = nn.Embedding(len(year_to_idx), hidden_dim)
        if use_coord:
            self.coord_proj = nn.Linear(2, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.layers.append(layer)

        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, year, coords):
        h = self.input_proj(x)
        if self.use_year:
            yidx = self.year_to_idx.get(int(year), 0)
            yemb = self.year_embedding(torch.tensor([yidx], device=x.device))
            h = h + yemb.expand(h.size(0), -1)
        if self.use_coord:
            h = h + self.coord_proj(coords)
        h = h.unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        h = h.squeeze(0)
        return self.output_proj(h)

class TempPredictor(nn.Module):
    def __init__(self, node_dim=8, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, h_node):
        if h_node.dim() == 1:
            h_node = h_node.unsqueeze(0)
        z = self.fc1(h_node); z = self.relu(z); z = self.drop(z); z = self.fc2(z)
        return z.squeeze(-1)

def build_features_idw(year, target_set, station_list, COORDS, ELEV,
                       year_station_temp, GLOBAL_TRAIN_MEAN, get_station_prior,
                       clim_dict, df_test_year, obs_set,
                       include_lon=True, include_lat=True, include_elev=True):
    lons = np.array([COORDS[st][0] for st in station_list])
    lats = np.array([COORDS[st][1] for st in station_list])
    elevs = np.array([ELEV[st] for st in station_list])
    lon_mean, lon_std = lons.mean(), lons.std() + 1e-8
    lat_mean, lat_std = lats.mean(), lats.std() + 1e-8
    elev_mean, elev_std = elevs.mean(), elevs.std() + 1e-8

    feats = []
    coords = []
    target_indices = []
    for idx, st in enumerate(station_list):
        lon, lat = COORDS[st]
        elev = ELEV[st]
        is_target = 1.0 if st in target_set else 0.0

        lon_norm = (lon - lon_mean) / lon_std
        lat_norm = (lat - lat_mean) / lat_std
        elev_norm = (elev - elev_mean) / elev_std
        clim_norm = (clim_dict.get(st, clim_mean_global) - clim_mean_global) / clim_std_global

        if st in target_set:
            idw_temp = idw_prior(st, obs_set, df_test_year, COORDS)
            temp_val = idw_temp
            has_obs = 0.0
        else:
            key = (year, st)
            temp_val = year_station_temp.get(key, clim_dict.get(st, GLOBAL_TRAIN_MEAN))
            has_obs = 1.0

        temp_norm = (temp_val - GLOBAL_TRAIN_MEAN) / 20.0
        feat = [temp_norm]
        if include_lon:
            feat.append(lon_norm)
        if include_lat:
            feat.append(lat_norm)
        if include_elev:
            feat.append(elev_norm)
        feat += [clim_norm, has_obs, is_target]
        feats.append(feat)
        coords.append([lon, lat])
        if is_target == 1.0:
            target_indices.append(idx)

    x = torch.tensor(feats, dtype=torch.float32)
    c = torch.tensor(coords, dtype=torch.float32)
    return x, c, target_indices

def run_single_experiment(seed, density):
    np.random.seed(seed)
    torch.manual_seed(seed)

    yr_test = int(np.random.choice(years_all))
    df_test_year = df[df["year"] == yr_test].copy()
    df_other = df[df["year"] != yr_test].copy()
    train_df, val_df = train_test_split(df_other, test_size=VAL_SIZE, random_state=seed)

    train_st, val_st, test_st = split_stations_by_density(station_list_all, density, seed)
    year_stations = df_test_year["station"].tolist()
    obs_set = [st for st in train_st if st in year_stations]
    target_set = [st for st in test_st if st in year_stations]
    if len(obs_set) < 3:
        print(f"Seed {seed} Density {density}: Too few observations ({len(obs_set)}), skipping.")
        return None

    station_list = sorted(df["station"].unique())
    station_info_full = df.groupby("station").first()[["lon", "lat", "elevation"]].to_dict("index")
    COORDS_full = {st: (station_info_full[st]["lon"], station_info_full[st]["lat"]) for st in station_list}
    ELEV_full = {st: station_info_full[st]["elevation"] for st in station_list}

    GLOBAL_TRAIN_MEAN = float(train_df["temp"].mean())
    STATION_CLIMATE_TRAIN = train_df.groupby("station")["temp"].mean().to_dict()
    def get_station_prior(st):
        return STATION_CLIMATE_TRAIN.get(st, GLOBAL_TRAIN_MEAN)

    clim_dict = compute_climatology(df, yr_test, station_list)
    year_station_temp = {}
    for _, row in pd.concat([train_df, val_df]).iterrows():
        year_station_temp[(int(row["year"]), str(row["station"]))] = float(row["temp"])

    years_unique = sorted(df["year"].unique())
    year_to_idx = {y: i for i, y in enumerate(years_unique)}
    true_temps = {str(row["station"]): float(row["temp"]) for _, row in df_test_year.iterrows()}

    include_lon = True
    include_lat = True
    include_elev = True
    use_year = True
    use_coord = True

    dummy_x, _, _ = build_features_idw(
        yr_test, target_set, station_list, COORDS_full, ELEV_full,
        year_station_temp, GLOBAL_TRAIN_MEAN, get_station_prior,
        clim_dict, df_test_year, obs_set,
        include_lon=include_lon, include_lat=include_lat, include_elev=include_elev
    )
    in_ch = dummy_x.shape[1]

    encoder = EncoderWithAttn(
        in_channels=in_ch,
        hidden_dim=EGT_HIDDEN,
        out_channels=EGT_OUT,
        num_layers=EGT_LAYERS,
        nhead=EGT_HEADS,
        dropout=EGT_DROPOUT,
        use_year=use_year,
        use_coord=use_coord,
        year_to_idx=year_to_idx
    ).to(DEVICE)

    predictor = TempPredictor(node_dim=EGT_OUT, hidden_dim=16, dropout=EGT_DROPOUT).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()),
                           lr=EGT_LR, weight_decay=EGT_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_samples = [(row["station"], row["year"], row["temp"]) for _, row in train_df.iterrows()]
    val_samples = [(row["station"], row["year"], row["temp"]) for _, row in val_df.iterrows()]

    best_val_rmse = float("inf")
    wait = 0
    best_encoder_sd = None
    best_predictor_sd = None

    for epoch in range(1, EGT_EPOCHS+1):
        encoder.train()
        predictor.train()
        total_loss = 0.0
        n = 0
        for st, yr, y_true in train_samples:
            target_set_single = {st}
            x, c, target_indices = build_features_idw(
                yr, target_set_single, station_list, COORDS_full, ELEV_full,
                year_station_temp, GLOBAL_TRAIN_MEAN, get_station_prior,
                clim_dict, df_test_year, obs_set,
                include_lon=include_lon, include_lat=include_lat, include_elev=include_elev
            )
            x, c = x.to(DEVICE), c.to(DEVICE)
            h_all = encoder(x, yr, c)
            h_target = h_all[target_indices[0]:target_indices[0]+1]
            pred = predictor(h_target)
            loss = criterion(pred, torch.tensor([y_true], device=DEVICE))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        encoder.eval()
        predictor.eval()
        preds_val, trues_val = [], []
        with torch.no_grad():
            for st, yr, y_true in val_samples:
                target_set_single = {st}
                x, c, target_indices = build_features_idw(
                    yr, target_set_single, station_list, COORDS_full, ELEV_full,
                    year_station_temp, GLOBAL_TRAIN_MEAN, get_station_prior,
                    clim_dict, df_test_year, obs_set,
                    include_lon=include_lon, include_lat=include_lat, include_elev=include_elev
                )
                x, c = x.to(DEVICE), c.to(DEVICE)
                h_all = encoder(x, yr, c)
                h_target = h_all[target_indices[0]:target_indices[0]+1]
                pred = predictor(h_target)
                preds_val.append(pred.item())
                trues_val.append(y_true)
        rmse_val = np.sqrt(mean_squared_error(trues_val, preds_val))

        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            wait = 0
            best_encoder_sd = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_predictor_sd = {k: v.cpu().clone() for k, v in predictor.state_dict().items()}
        else:
            wait += 1
            if wait >= EGT_PATIENCE:
                break

    encoder.load_state_dict(best_encoder_sd)
    predictor.load_state_dict(best_predictor_sd)
    encoder.eval()
    predictor.eval()
    preds_test, trues_test = [], []
    with torch.no_grad():
        for st in target_set:
            yr = yr_test
            target_set_single = {st}
            x, c, target_indices = build_features_idw(
                yr, target_set_single, station_list, COORDS_full, ELEV_full,
                year_station_temp, GLOBAL_TRAIN_MEAN, get_station_prior,
                clim_dict, df_test_year, obs_set,
                include_lon=include_lon, include_lat=include_lat, include_elev=include_elev
            )
            x, c = x.to(DEVICE), c.to(DEVICE)
            h_all = encoder(x, yr, c)
            h_target = h_all[target_indices[0]:target_indices[0]+1]
            pred = predictor(h_target)
            preds_test.append(pred.item())
            trues_test.append(true_temps[st])

    preds_c = (np.array(preds_test) - 32.0) * 5.0 / 9.0
    trues_c = (np.array(trues_test) - 32.0) * 5.0 / 9.0
    rmse = np.sqrt(mean_squared_error(trues_c, preds_c))
    mae = mean_absolute_error(trues_c, preds_c)
    r2 = r2_score(trues_c, preds_c)
    return rmse, mae, r2, preds_c, trues_c

results = []
all_preds = {}
all_trues = {}

for density in DENSITIES:
    print(f"\n===== Density {density*100:.0f}% =====")
    rmse_list, mae_list, r2_list = [], [], []
    preds_all, trues_all = [], []
    for seed in SEEDS:
        res = run_single_experiment(seed, density)
        if res is None:
            continue
        rmse, mae, r2, preds, trues = res
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        preds_all.extend(preds.tolist())
        trues_all.extend(trues.tolist())
        print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    if len(rmse_list) > 0:
        overall_rmse = np.sqrt(mean_squared_error(trues_all, preds_all))
        overall_mae = mean_absolute_error(trues_all, preds_all)
        overall_r2 = r2_score(trues_all, preds_all)
        results.append({
            'Density': f"{density*100:.0f}%",
            'RMSE_mean': np.mean(rmse_list),
            'RMSE_std': np.std(rmse_list),
            'MAE_mean': np.mean(mae_list),
            'MAE_std': np.std(mae_list),
            'R2_mean': np.mean(r2_list),
            'R2_std': np.std(r2_list),
            'Overall_RMSE': overall_rmse,
            'Overall_MAE': overall_mae,
            'Overall_R2': overall_r2,
            'preds': preds_all,
            'trues': trues_all
        })
    else:
        results.append({
            'Density': f"{density*100:.0f}%",
            'RMSE_mean': np.nan, 'RMSE_std': np.nan,
            'MAE_mean': np.nan, 'MAE_std': np.nan,
            'R2_mean': np.nan, 'R2_std': np.nan,
            'Overall_RMSE': np.nan, 'Overall_MAE': np.nan, 'Overall_R2': np.nan,
            'preds': [], 'trues': []
        })

all_preds_total = []
all_trues_total = []
for res in results:
    all_preds_total.extend(res['preds'])
    all_trues_total.extend(res['trues'])

total_rmse = np.sqrt(mean_squared_error(all_trues_total, all_preds_total))
total_mae = mean_absolute_error(all_trues_total, all_preds_total)
total_r2 = r2_score(all_trues_total, all_preds_total)

summary_df = pd.DataFrame(results)
summary_df = summary_df[['Density', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']]
summary_df.columns = ['Density', 'RMSE', 'RMSE_std', 'MAE', 'MAE_std', 'R²', 'R²_std']
summary_df = summary_df.round(4)

total_row = pd.DataFrame({
    'Density': ['Total'],
    'RMSE': [total_rmse],
    'RMSE_std': [np.nan],
    'MAE': [total_mae],
    'MAE_std': [np.nan],
    'R²': [total_r2],
    'R²_std': [np.nan]
})
summary_df = pd.concat([summary_df, total_row], ignore_index=True)

print("\n" + "="*80)
print("Full Model Performance Across Densities")
print("="*80)
print(summary_df.to_string(index=False))
print("\nOverall (all densities combined): RMSE = {:.4f}°C, MAE = {:.4f}°C, R² = {:.4f}".format(total_rmse, total_mae, total_r2))

summary_df.to_csv("full_model_multi_density_results.csv", index=False)
print("\nResults saved to full_model_multi_density_results.csv")
