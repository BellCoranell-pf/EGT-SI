import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay, KDTree
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    raise ImportError("Please install pykrige: pip install pykrige")
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")
try:
    from torch_geometric.nn import GCNConv, BatchNorm
except ImportError:
    raise ImportError("Please install torch-geometric: pip install torch-geometric")

DATA_PATH      = "TAVE.csv"
DEM_PATH       = "chinadem.tif"
OUTPUT_PNG     = "7methods_full_station_interp.png"
OUTPUT_RASTER  = "7methods_full_station_interp.tif"
OUTPUT_RES     = 0.5
SEED = 45

EGT_HIDDEN    = 32
EGT_OUT       = 8
EGT_LAYERS    = 2
EGT_HEADS     = 4
EGT_DROPOUT   = 0.1063
EGT_LR        = 9.855e-5
EGT_WD        = 1.100e-6
EGT_EPOCHS    = 200
EGT_SAMPLES   = 3000

STGCN_HIDDEN  = 64
STGCN_OUT     = 8
STGCN_LR      = 1e-5
STGCN_EPOCHS  = 200
STGCN_PATIENCE = 10
ST_DROPOUT    = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    except Exception:
        df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    df_raw.rename(columns={
        "station_id": "station", "longitude": "lon", "latitude": "lat", "Tave": "temp"
    }, inplace=True)
    df = df_raw[["year", "station", "lon", "lat", "elevation", "temp"]].dropna().copy()
    df["year"]    = df["year"].astype(int)
    df["station"] = df["station"].astype(str).str.strip()
    df["temp"]    = df["temp"].astype(np.float32)
    df = df[(df["temp"] > -60) & (df["temp"] < 80)]
    df["temp"] = (df["temp"] - 32.0) * 5.0 / 9.0
    return df

df_full = load_data()
years_all = sorted(df_full["year"].unique())
sy_counts = df_full.groupby("station")["year"].nunique()
common_stations = sorted(sy_counts[sy_counts == len(years_all)].index.tolist())
df = df_full[df_full["station"].isin(common_stations)].copy()
n_stations = len(common_stations)
station_idx = {st: i for i, st in enumerate(common_stations)}

station_info = df.groupby("station").first()[["lon", "lat", "elevation"]]
lons_raw  = station_info.loc[common_stations, "lon"].values.astype(np.float32)
lats_raw  = station_info.loc[common_stations, "lat"].values.astype(np.float32)
elevs_raw = station_info.loc[common_stations, "elevation"].values.astype(np.float32)

rng_exp = np.random.RandomState(SEED)
yr_test = int(rng_exp.choice(years_all))
print(f"\n==== Target prediction year yr_test = {yr_test} ====")

train_lon = lons_raw.copy()
train_lat = lats_raw.copy()
train_elev = elevs_raw.copy()

year_temp_array = {}
for yr in years_all:
    yr = int(yr)
    arr = np.full(n_stations, np.nan, dtype=np.float32)
    sub = df[df["year"] == yr]
    for st, t in zip(sub["station"].values, sub["temp"].values):
        arr[station_idx[st]] = np.float32(t)
    year_temp_array[yr] = arr
year_to_idx = {int(y): i for i, y in enumerate(years_all)}
train_clim = np.nanmean(list(year_temp_array.values()), axis=0)
clim_mean = float(np.nanmean(train_clim))
clim_std = float(np.nanstd(train_clim) + 1e-8)
global_train_mean = float(df["temp"].mean())

print("\nLoading DEM and constructing grid...")
with rasterio.open(DEM_PATH) as src:
    dem_data = src.read(1)
    dem_transform = src.transform
    dem_crs = src.crs
    dem_nodata = src.nodata

valid_mask_dem = (dem_data != dem_nodata) & (~np.isnan(dem_data))
rows_all, cols_all = np.where(valid_mask_dem)
lon_min_dem = dem_transform[2] + cols_all.min() * dem_transform[0]
lon_max_dem = dem_transform[2] + (cols_all.max() + 1) * dem_transform[0]
lat_max_dem = dem_transform[5] + rows_all.min() * dem_transform[4]
lat_min_dem = dem_transform[5] + (rows_all.max() + 1) * dem_transform[4]
if lat_min_dem > lat_max_dem: lat_min_dem, lat_max_dem = lat_max_dem, lat_min_dem
if lon_min_dem > lon_max_dem: lon_min_dem, lon_max_dem = lon_max_dem, lon_min_dem

def align_extent(vmin, vmax, res):
    return np.floor(vmin / res) * res, np.ceil(vmax / res) * res

lon_min_grid, lon_max_grid = align_extent(lon_min_dem, lon_max_dem, OUTPUT_RES)
lat_min_grid, lat_max_grid = align_extent(lat_min_dem, lat_max_dem, OUTPUT_RES)
xs = np.arange(lon_min_grid + OUTPUT_RES/2, lon_max_grid, OUTPUT_RES)
ys = np.arange(lat_max_grid - OUTPUT_RES/2, lat_min_grid, -OUTPUT_RES)
print(f"Grid: {len(ys)} rows x {len(xs)} cols")

lon_grid, lat_grid = np.meshgrid(xs, ys, indexing="xy")
lons_flat = lon_grid.flatten().astype(np.float32)
lats_flat = lat_grid.flatten().astype(np.float32)
n_grid_all = lons_flat.size

print("Sampling DEM elevation...")
with rasterio.open(DEM_PATH) as src:
    elev_flat = np.array([next(src.sample([(lon, lat)]))[0]
                          for lon, lat in zip(lons_flat, lats_flat)], dtype=np.float32)
    valid_flat = (elev_flat != src.nodata) & (~np.isnan(elev_flat))

valid_idx_flat = np.where(valid_flat)[0]
rows_valid, cols_valid = np.unravel_index(valid_idx_flat, (len(ys), len(xs)))
lons_valid  = lons_flat[valid_flat]
lats_valid  = lats_flat[valid_flat]
elevs_valid = elev_flat[valid_flat]
n_grid = len(lons_valid)
print(f"Valid grid points: {n_grid}")

def reconstruct_grid(preds):
    grid_out = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    grid_out[rows_valid, cols_valid] = preds
    return grid_out

def haversine_mat_vec(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1_r = np.radians(lon1)[:, None]; lat1_r = np.radians(lat1)[:, None]
    lon2_r = np.radians(lon2)[None, :]; lat2_r = np.radians(lat2)[None, :]
    dlon = lon2_r - lon1_r; dlat = lat2_r - lat1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).astype(np.float32)

D_GRID_TRAIN = haversine_mat_vec(lons_valid, lats_valid, train_lon, train_lat)
yr_train_temp = year_temp_array[yr_test]

print("\n[1/7] IDW")
w_idw = 1.0 / (D_GRID_TRAIN ** 2 + 1e-12)
pred_idw = (w_idw @ yr_train_temp) / w_idw.sum(axis=1)

print("\n[2/7] OK Ordinary Kriging")
try:
    ok_model = OrdinaryKriging(train_lon, train_lat, yr_train_temp, variogram_model="spherical", verbose=False, enable_plotting=False)
except:
    ok_model = OrdinaryKriging(train_lon, train_lat, yr_train_temp, variogram_model="linear", verbose=False, enable_plotting=False)
z_ok, _ = ok_model.execute("points", lons_valid, lats_valid)
pred_ok = np.clip(np.asarray(z_ok, dtype=np.float32), np.nanmin(yr_train_temp)-5, np.nanmax(yr_train_temp)+5)

print("\n[3/7] RK Regression-Kriging")
lr = LinearRegression().fit(train_elev.reshape(-1,1), yr_train_temp)
residuals = yr_train_temp - lr.predict(train_elev.reshape(-1,1))
try:
    rk_model = OrdinaryKriging(train_lon, train_lat, residuals, variogram_model="spherical", verbose=False, enable_plotting=False)
except:
    rk_model = OrdinaryKriging(train_lon, train_lat, residuals, variogram_model="linear", verbose=False, enable_plotting=False)
z_res, _ = rk_model.execute("points", lons_valid, lats_valid)
pred_rk = lr.predict(elevs_valid.reshape(-1,1)) + np.asarray(z_res, dtype=np.float32)

print("\n[4/7] XGBoost")
df_train_block = df.copy()
X_train = df_train_block[["lon","lat","elevation","year"]].values.astype(np.float32)
y_train = df_train_block["temp"].values.astype(np.float32)
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")
xgb_model.fit(X_train, y_train)
X_grid = np.column_stack([lons_valid, lats_valid, elevs_valid, np.full(n_grid, yr_test, dtype=np.float32)]).astype(np.float32)
pred_xgb = xgb_model.predict(X_grid).astype(np.float32)

print("\n[5/7] Random Forest")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_grid).astype(np.float32)

print("\n[6/7] EGT-SI distance-penalty attention")
class SpatialAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_ff)
        self.linear2   = nn.Linear(dim_ff, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.distance_penalty = nn.Parameter(torch.tensor(5.0))
    def forward(self, src, dist_matrix):
        attn_mask = -self.distance_penalty * dist_matrix
        src2, _ = self.self_attn(src, src, src, attn_mask=attn_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class EGTEncoder(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, n_layers, nhead, dropout, year_to_idx):
        super().__init__()
        self.year_to_idx = year_to_idx
        self.input_proj  = nn.Linear(in_ch, hidden)
        self.year_emb    = nn.Embedding(len(year_to_idx), hidden)
        self.coord_proj  = nn.Linear(2, hidden)
        self.layers = nn.ModuleList([SpatialAttentionLayer(hidden, nhead, hidden*4, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, out_ch)
    def forward(self, x, year, coords, dist_matrix):
        h = self.input_proj(x)
        yidx = self.year_to_idx.get(int(year),0)
        yemb = self.year_emb(torch.tensor([yidx], dtype=torch.long, device=x.device))
        h = h + yemb.expand(h.size(0),-1) + self.coord_proj(coords)
        h = h.unsqueeze(0)
        for layer in self.layers:
            h = layer(h, dist_matrix)
        return self.output_proj(h.squeeze(0))

class TempPredictor(nn.Module):
    def __init__(self, node_dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden,1)
    def forward(self, h):
        if h.dim()==1: h=h.unsqueeze(0)
        return self.fc2(self.drop(torch.relu(self.fc1(h)))).squeeze(-1)

n_train_st = len(train_lon)
DIST_TRAIN_KM = haversine_mat_vec(train_lon, train_lat, train_lon, train_lat)
DIST_TRAIN_NORM = DIST_TRAIN_KM/(DIST_TRAIN_KM.max()+1e-8)
COORDS_TRAIN = torch.tensor(np.stack([train_lon, train_lat],axis=1), dtype=torch.float32, device=DEVICE)
DIST_TRAIN_T = torch.tensor(DIST_TRAIN_NORM, dtype=torch.float32, device=DEVICE)

def train_block_loo_idw(target_i, obs_idx, obs_arr, dist_mat):
    context = obs_idx[obs_idx != target_i]
    d = dist_mat[target_i, context]
    w = 1.0/(d**2+1e-12)
    v = obs_arr[context]
    mask = ~np.isnan(v)
    return float((w[mask]@v[mask])/(w[mask].sum()+1e-12))

all_egt_samples = []
for yr in years_all:
    arr = year_temp_array[int(yr)]
    for i in range(n_train_st):
        if not np.isnan(arr[i]):
            all_egt_samples.append((int(yr), i, float(arr[i])))

encoder = EGTEncoder(7, EGT_HIDDEN, EGT_OUT, EGT_LAYERS, EGT_HEADS, EGT_DROPOUT, year_to_idx).to(DEVICE)
predictor = TempPredictor(EGT_OUT, 16, EGT_DROPOUT).to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters())+list(predictor.parameters()), lr=EGT_LR, weight_decay=EGT_WD)
criterion = nn.MSELoss()

X_base_train = {}
for yr in years_all:
    yr=int(yr)
    t_arr = year_temp_array[yr].copy()
    X = np.zeros((n_train_st,7), dtype=np.float32)
    X[:,0] = (t_arr - global_train_mean)/20.0
    X[:,1] = (train_lon-lons_raw.mean())/(lons_raw.std()+1e-8)
    X[:,2] = (train_lat-lats_raw.mean())/(lats_raw.std()+1e-8)
    X[:,3] = (train_elev-elevs_raw.mean())/(elevs_raw.std()+1e-8)
    X[:,4] = (train_clim - clim_mean)/clim_std
    X[:,5]=1.0; X[:,6]=0.0
    X_base_train[yr] = X

for epoch in range(EGT_EPOCHS):
    encoder.train(); predictor.train()
    np.random.shuffle(all_egt_samples)
    subset = all_egt_samples[:EGT_SAMPLES]
    for yr, sidx, y_true in subset:
        X = X_base_train[yr].copy()
        loo_t = train_block_loo_idw(sidx, np.arange(n_train_st), year_temp_array[yr], DIST_TRAIN_KM)
        X[sidx,0] = (loo_t - global_train_mean)/20.0
        X[sidx,4] = (train_block_loo_idw(sidx, np.arange(n_train_st), train_clim, DIST_TRAIN_KM)-clim_mean)/clim_std
        X[sidx,5]=0.0; X[sidx,6]=1.0
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        h_all = encoder(X_t, yr, COORDS_TRAIN, DIST_TRAIN_T)
        pred = predictor(h_all[sidx:sidx+1])
        loss = criterion(pred, torch.tensor([y_true], dtype=torch.float32, device=DEVICE))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.0)
        optimizer.step()
    if (epoch+1)%50 ==0:
        print(f"  Epoch {epoch+1}/{EGT_EPOCHS} done.")

encoder.eval(); predictor.eval()

BATCH=100
pred_egt = np.zeros(n_grid, dtype=np.float32)
with torch.no_grad():
    for start in range(0, n_grid, BATCH):
        end = min(start+BATCH, n_grid)
        nb = end-start
        b_lons = lons_valid[start:end]; b_lats = lats_valid[start:end]; b_elevs = elevs_valid[start:end]
        all_lon = np.concatenate([train_lon, b_lons])
        all_lat = np.concatenate([train_lat, b_lats])
        n_total = n_train_st + nb
        d_full = haversine_mat_vec(all_lon, all_lat, all_lon, all_lat)
        d_full_norm = np.clip(d_full/(DIST_TRAIN_KM.max()+1e-8),0,1)
        d_full_norm[n_train_st:, n_train_st:] = 1.0
        X = np.zeros((n_total,7), dtype=np.float32)
        t_now = year_temp_array[yr_test].copy()
        X[:n_train_st,0] = (t_now - global_train_mean)/20.0
        X[:n_train_st,1] = (train_lon-lons_raw.mean())/(lons_raw.std()+1e-8)
        X[:n_train_st,2] = (train_lat-lats_raw.mean())/(lats_raw.std()+1e-8)
        X[:n_train_st,3] = (train_elev-elevs_raw.mean())/(elevs_raw.std()+1e-8)
        X[:n_train_st,4] = (train_clim-clim_mean)/clim_std
        X[:n_train_st,5]=1.0; X[:n_train_st,6]=0.0
        for k in range(nb):
            row_d = d_full[n_train_st+k, :n_train_st]
            w = 1.0/(row_d**2+1e-12)
            mask_t = ~np.isnan(t_now)
            v_t = (w[mask_t] @ t_now[mask_t])/(w[mask_t].sum()+1e-12)
            v_c = (w @ train_clim)/(w.sum()+1e-12)
            X[n_train_st+k,0] = (v_t - global_train_mean)/20.0
            X[n_train_st+k,1] = (b_lons[k]-lons_raw.mean())/(lons_raw.std()+1e-8)
            X[n_train_st+k,2] = (b_lats[k]-lats_raw.mean())/(lats_raw.std()+1e-8)
            X[n_train_st+k,3] = (b_elevs[k]-elevs_raw.mean())/(elevs_raw.std()+1e-8)
            X[n_train_st+k,4] = (v_c - clim_mean)/clim_std
            X[n_train_st+k,5]=0.0; X[n_train_st+k,6]=1.0
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        c_t = torch.tensor(np.stack([all_lon, all_lat],axis=1), dtype=torch.float32, device=DEVICE)
        d_t = torch.tensor(d_full_norm, dtype=torch.float32, device=DEVICE)
        h_all = encoder(X_t, yr_test, c_t, d_t)
        pred_egt[start:end] = predictor(h_all[n_train_st:]).cpu().numpy()

print("\n[7/7] ST-GCN distance-penalty attention (Dynamic Delaunay)")
class STGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = BatchNorm(hidden)
        self.conv3 = GCNConv(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x, ei, ew):
        x = self.drop(self.relu(self.bn1(self.conv1(x, ei, ew))))
        x = self.drop(self.relu(self.bn2(self.conv2(x, ei, ew))))
        return self.conv3(x, ei, ew)

class TempPredictorST(nn.Module):
    def __init__(self, node_dim, extra_dim, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim+extra_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden,1)
        )
    def forward(self,h,extra):
        return self.mlp(torch.cat([h,extra],dim=1)).squeeze(-1)

coords_train_st = np.column_stack([train_lon, train_lat])
tri_train = Delaunay(coords_train_st)
edges_train = set()
for simplex in tri_train.simplices:
    for k in range(3):
        u, v = simplex[k], simplex[(k+1)%3]
        edges_train.add((u, v)); edges_train.add((v, u))
edge_list_train = list(edges_train)
ei_st = torch.tensor(edge_list_train, dtype=torch.long).t().contiguous().to(DEVICE)
ew_st = torch.tensor([1.0/(np.linalg.norm(coords_train_st[u] - coords_train_st[v])+1e-6) for u,v in edge_list_train], dtype=torch.float32).to(DEVICE)

feat_mean_train = np.array([np.mean(train_lon), np.mean(train_lat), np.mean(train_elev), np.mean(years_all)], dtype=np.float32)
feat_std_train = np.array([np.std(train_lon)+1e-8, np.std(train_lat)+1e-8, np.std(train_elev)+1e-8, np.std(years_all)+1e-8])
temp_mean_tr = global_train_mean
temp_std_tr = float(np.nanstd(df["temp"])+1e-8)

node_feats_cache = {}
for yr in years_all:
    feats = [(np.array([train_lon[i], train_lat[i], train_elev[i], yr], dtype=np.float32)-feat_mean_train)/feat_std_train for i in range(n_train_st)]
    node_feats_cache[int(yr)] = torch.tensor(np.array(feats), dtype=torch.float32, device=DEVICE)

all_st_samples = []
for yr in years_all:
    arr = year_temp_array[int(yr)]
    for i in range(n_train_st):
        if not np.isnan(arr[i]):
            all_st_samples.append((i, int(yr), (float(arr[i])-temp_mean_tr)/temp_std_tr))

st_encoder = STGCNEncoder(in_dim=4, hidden=STGCN_HIDDEN, out_dim=STGCN_OUT, dropout=ST_DROPOUT).to(DEVICE)
st_predictor = TempPredictorST(STGCN_OUT, 3, 16).to(DEVICE)
st_opt = optim.Adam(list(st_encoder.parameters())+list(st_predictor.parameters()), lr=STGCN_LR)
st_crit = nn.MSELoss()

np.random.shuffle(all_st_samples)
split = int(0.8*len(all_st_samples))
train_samp, val_samp = all_st_samples[:split], all_st_samples[split:]

best_val, wait = float("inf"), 0
best_e_sd, best_p_sd = None, None

for epoch in range(STGCN_EPOCHS):
    st_encoder.train(); st_predictor.train()
    for sidx, yr, t_norm in train_samp:
        x = node_feats_cache[yr]
        h_all = st_encoder(x, ei_st, ew_st)
        extra = torch.tensor([(np.array([train_lon[sidx], train_lat[sidx], train_elev[sidx]])-feat_mean_train[:3])/feat_std_train[:3]], dtype=torch.float32, device=DEVICE)
        pred = st_predictor(h_all[sidx:sidx+1], extra)
        loss = st_crit(pred, torch.tensor([t_norm], dtype=torch.float32, device=DEVICE))
        st_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(st_encoder.parameters(),1.0)
        st_opt.step()

    st_encoder.eval(); st_predictor.eval()
    pv, tv = [], []
    with torch.no_grad():
        for sidx, yr, t_norm in val_samp:
            x = node_feats_cache[yr]
            h_all = st_encoder(x, ei_st, ew_st)
            extra = torch.tensor([(np.array([train_lon[sidx], train_lat[sidx], train_elev[sidx]])-feat_mean_train[:3])/feat_std_train[:3]], dtype=torch.float32, device=DEVICE)
            p = st_predictor(h_all[sidx:sidx+1], extra).item()
            pv.append(p*temp_std_tr+temp_mean_tr)
            tv.append(t_norm*temp_std_tr+temp_mean_tr)

    rmse_v = float(np.sqrt(np.mean((np.array(pv)-np.array(tv))**2)))
    if rmse_v < best_val:
        best_val, wait = rmse_v, 0
        best_e_sd = {k:v.cpu().clone() for k,v in st_encoder.state_dict().items()}
        best_p_sd = {k:v.cpu().clone() for k,v in st_predictor.state_dict().items()}
    else:
        wait += 1
        if wait >= STGCN_PATIENCE:
            break
    if (epoch+1)%50==0:
        print(f"  Epoch {epoch+1}/{STGCN_EPOCHS} val RMSE={rmse_v:.4f}")

st_encoder.load_state_dict(best_e_sd)
st_predictor.load_state_dict(best_p_sd)
st_encoder.eval(); st_predictor.eval()

print("\nPerforming point-by-point dynamic Delaunay grid inference...")
pred_stgnn = np.zeros(n_grid, dtype=np.float32)
feats_train_np = node_feats_cache[yr_test].cpu().numpy()

t0 = time.time()
with torch.no_grad():
    for idx in range(n_grid):
        tgt_lon = lons_valid[idx]
        tgt_lat = lats_valid[idx]
        tgt_elev = elevs_valid[idx]

        curr_coords = np.vstack([coords_train_st, [[tgt_lon, tgt_lat]]])
        tri = Delaunay(curr_coords)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                u, v = simplex[k], simplex[(k+1)%3]
                edges.add((u, v)); edges.add((v, u))
        edge_list = list(edges)
        ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(DEVICE)
        weights = [1.0 / (np.linalg.norm(curr_coords[u] - curr_coords[v]) + 1e-6) for u,v in edge_list]
        ew = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        tgt_feat_raw = np.array([tgt_lon, tgt_lat, tgt_elev, yr_test], dtype=np.float32)
        tgt_feat_norm = (tgt_feat_raw - feat_mean_train) / feat_std_train
        curr_feats = np.vstack([feats_train_np, tgt_feat_norm])
        x_tensor = torch.tensor(curr_feats, dtype=torch.float32).to(DEVICE)

        h_all = st_encoder(x_tensor, ei, ew)
        h_target = h_all[-1:]

        extra_raw = np.array([tgt_lon, tgt_lat, tgt_elev])
        extra_norm = (extra_raw - feat_mean_train[:3]) / feat_std_train[:3]
        extra_tensor = torch.tensor([extra_norm], dtype=torch.float32).to(DEVICE)

        p_norm = st_predictor(h_target, extra_tensor).item()
        pred_stgnn[idx] = p_norm * temp_std_tr + temp_mean_tr

print(f"ST-GCN full map inference completed in {time.time()-t0:.1f} seconds")

predictions = {
    "IDW": pred_idw,
    "Ordinary Kriging (OK)": pred_ok,
    "Regression Kriging (RK)": pred_rk,
    "XGBoost": pred_xgb,
    "Random Forest": pred_rf,
    "EGT-SI": pred_egt,
    "ST-GCN": pred_stgnn,
}
all_vals = np.concatenate([v for v in predictions.values()])
vmin = max(float(np.nanmin(all_vals)), np.nanmin(yr_train_temp)-3.0)
vmax = min(float(np.nanmax(all_vals)), np.nanmax(yr_train_temp)+3.0)

def save_single_plot(name, pred_arr, title_suffix=""):
    fig_single, ax_single = plt.subplots(figsize=(8, 6), dpi=300)
    grid_img = reconstruct_grid(pred_arr)
    im = ax_single.imshow(grid_img, extent=[lon_min_grid, lon_max_grid, lat_min_grid, lat_max_grid],
                          origin="upper", cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="auto")

    ax_single.set_xlim(73, 136)
    ax_single.set_ylim(15, 55)
    ax_single.set_title(f"{name} {title_suffix}", fontsize=12, fontweight="bold")
    ax_single.set_xlabel("Longitude °E"); ax_single.set_ylabel("Latitude °N")

    ax_single.grid(True, linestyle='--', color='gray', alpha=0.5)
    cbar = fig_single.colorbar(im, ax=ax_single, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.tight_layout()
    fn = f"single_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig_single)
    print(f"  Saved single map: {fn}")

print("\nGenerating 8-in-1 main plot and single plots...")
fig, axes = plt.subplots(2, 4, figsize=(24, 11), dpi=150)
axes_flat = axes.flatten()

ax0 = axes_flat[0]
dem_plot_data = dem_data.copy().astype(float)
dem_plot_data[~valid_mask_dem] = np.nan

dem_img = ax0.imshow(dem_plot_data, extent=[lon_min_dem, lon_max_dem, lat_min_dem, lat_max_dem],
                     origin="upper", cmap="terrain", aspect="auto")

ax0.set_xlim(73, 136)
ax0.set_ylim(15, 55)
ax0.set_title("DEM Elevation", fontsize=13, fontweight='bold')
ax0.set_xlabel("Longitude °E"); ax0.set_ylabel("Latitude °N")

ax0.set_xticks(np.arange(75, 135, 5))
ax0.set_yticks(np.arange(15, 55, 5))
ax0.grid(True, linestyle=':', color='gray', alpha=0.6)

fig_dem_single, ax_dem_single = plt.subplots(figsize=(8, 6), dpi=300)
dem_img_single = ax_dem_single.imshow(dem_plot_data, extent=[lon_min_dem, lon_max_dem, lat_min_dem, lat_max_dem],
                                      origin="upper", cmap="terrain", aspect="auto")
ax_dem_single.set_xlim(73, 136)
ax_dem_single.set_ylim(15, 55)
ax_dem_single.set_title("DEM Elevation", fontsize=12, fontweight="bold")
ax_dem_single.set_xlabel("Longitude °E"); ax_dem_single.set_ylabel("Latitude °N")
ax_dem_single.grid(True, linestyle='--', color='gray', alpha=0.5)
cbar_dem = fig_dem_single.colorbar(dem_img_single, ax=ax_dem_single, fraction=0.046, pad=0.04)
cbar_dem.set_label("Elevation (m)")
plt.tight_layout()
plt.savefig("single_DEM_Elevation.png", dpi=300, bbox_inches="tight")
plt.close(fig_dem_single)
print("  Saved single map: single_DEM_Elevation.png")

for ax, (name, pred) in zip(axes_flat[1:], predictions.items()):
    grid_img = reconstruct_grid(pred)
    im = ax.imshow(grid_img, extent=[lon_min_grid, lon_max_grid, lat_min_grid, lat_max_grid],
                   origin="upper", cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="auto")
    ax.set_xlim(73, 136)
    ax.set_ylim(15, 55)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel("Longitude °E"); ax.set_ylabel("Latitude °N")

    ax.set_xticks(np.arange(75, 135, 5))
    ax.set_yticks(np.arange(15, 55, 5))
    ax.grid(True, linestyle=':', color='gray', alpha=0.4)

    save_single_plot(name, pred, f"| target-year {yr_test}")

cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.55])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Temperature (°C)", fontsize=12)

cbar_ax_dem = fig.add_axes([0.92, 0.75, 0.012, 0.15])
cbar_dem_fig = fig.colorbar(dem_img, cax=cbar_ax_dem)
cbar_dem_fig.set_label("Elevation (m)", fontsize=12)

plt.suptitle(f"Seven-Method Spatial Interpolation Mapping | target-year {yr_test}", fontsize=16, y=0.96, fontweight='bold')
plt.tight_layout(rect=[0,0,0.91,0.95])
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n8-in-1 main plot saved: {OUTPUT_PNG}")

transform = from_origin(lon_min_grid, lat_max_grid, OUTPUT_RES, -OUTPUT_RES)
band_names = list(predictions.keys())
with rasterio.open(OUTPUT_RASTER, "w", driver="GTiff",
                   height=len(ys), width=len(xs), count=7, dtype="float32",
                   crs=dem_crs, transform=transform, nodata=np.nan) as dst:
    for i, name in enumerate(band_names):
        dst.write(reconstruct_grid(predictions[name]), i+1)
        dst.set_band_description(i+1, name)
print(f"Multi-band GeoTIFF saved: {OUTPUT_RASTER}")
