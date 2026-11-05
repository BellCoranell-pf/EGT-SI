import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from scipy.spatial import Delaunay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "TAVE.csv"
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = list(range(42, 52))

GCN_HIDDEN = 64
GCN_OUT = 8
PRED_HIDDEN = 32
GCN_LR = 1e-5
GCN_PATIENCE = 10
GCN_EPOCHS = 200
CLIP = 1.0
BATCH_K = 32

TARGET_COLUMNS = {'Tave': 'Tave'}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}, BATCH_K={BATCH_K}")


def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception:
        df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    df_raw.rename(columns={"station_id": "station", "longitude": "lon",
                           "latitude": "lat"}, inplace=True)
    df = df_raw[["year", "station", "lon", "lat", "elevation", "Tave"]].dropna().copy()
    df["year"] = df["year"].astype(int)
    df["station"] = df["station"].astype(str).str.strip()
    return df


def haversine_matrix(lon, lat):
    R = 6371.0
    lon_r = np.radians(lon); lat_r = np.radians(lat)
    dlon = lon_r[None, :] - lon_r[:, None]
    dlat = lat_r[None, :] - lat_r[:, None]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon/2)**2
    return (R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)


def build_delaunay_edges(lon, lat):
    pts = np.column_stack([lon, lat]).astype(np.float64)
    try:
        tri = Delaunay(pts)
    except Exception:
        rng = np.random.RandomState(0)
        tri = Delaunay(pts + rng.normal(0.0, 1e-6, pts.shape))
    edges = set()
    for simplex in tri.simplices:
        for k in range(3):
            i = int(simplex[k]); j = int(simplex[(k + 1) % 3])
            if i > j: i, j = j, i
            edges.add((i, j))
    return edges


def split_stations_by_density(stations, density, seed):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(stations)
    n = len(shuffled)
    n_val = int(n * density); n_test = int(n * density)
    n_train = n - n_val - n_test
    return (list(shuffled[:n_train]),
            list(shuffled[n_train:n_train + n_val]),
            list(shuffled[n_train + n_val:]))


def compute_clim_idw(target_idx, context_idx, context_clim, DIST_MATRIX):
    if len(context_idx) == 0:
        return float(np.nanmean(context_clim)) if len(context_clim) else 0.0
    d = DIST_MATRIX[target_idx, context_idx]
    w = 1.0 / (d ** 2 + 1e-12)
    mask = ~np.isnan(context_clim)
    if not mask.any(): return 0.0
    return float((w[mask] * context_clim[mask]).sum() / w[mask].sum())


class GCNEncoder(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class TempPredictor(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_node):
        if h_node.dim() == 1: h_node = h_node.unsqueeze(0)
        return self.mlp(h_node).squeeze(-1)


def prepare_seed_fast(seed, density, target_col, df, years_all,
                      station_list_all, station_idx, n_stations, DIST_MATRIX,
                      lon_norm, lat_norm, elev_norm):
    np.random.seed(seed)
    yr_test = int(np.random.choice(years_all))
    train_st, val_st, test_st = split_stations_by_density(station_list_all, density, seed)
    train_idx = np.array([station_idx[st] for st in train_st], dtype=np.int64)
    val_idx = np.array([station_idx[st] for st in val_st], dtype=np.int64)
    test_idx = np.array([station_idx[st] for st in test_st], dtype=np.int64)

    df_other = df[df["year"] != yr_test]
    train_df = df_other[df_other["station"].isin(train_st)].copy()

    train_clim = np.full(n_stations, np.nan, dtype=np.float32)
    for st, v in train_df.groupby("station")[target_col].mean().items():
        train_clim[station_idx[st]] = np.float32(v)
    valid_clim = train_clim[train_idx]; valid_clim = valid_clim[~np.isnan(valid_clim)]
    clim_mean = float(valid_clim.mean()) if len(valid_clim) else 0.0
    clim_std = float(valid_clim.std() + 1e-8) if len(valid_clim) else 1.0
    global_mean = float(train_df[target_col].mean())

    clim_arr = train_clim.copy()
    non_train_idx = np.setdiff1d(np.arange(n_stations), train_idx)
    if len(non_train_idx) > 0 and len(train_idx) > 0:
        d_nt = DIST_MATRIX[non_train_idx][:, train_idx]
        w_nt = 1.0 / (d_nt ** 2 + 1e-12)
        mask = ~np.isnan(train_clim[train_idx])
        if mask.any():
            vals = train_clim[train_idx][mask]
            ww = w_nt[:, mask]
            clim_arr[non_train_idx] = (ww @ vals) / ww.sum(axis=1)
        else:
            clim_arr[non_train_idx] = global_mean
    clim_arr = np.where(np.isnan(clim_arr), global_mean, clim_arr).astype(np.float32)

    clim_except = np.zeros(n_stations, dtype=np.float32)
    for i in train_idx:
        ctx = train_idx[train_idx != i]
        clim_except[int(i)] = compute_clim_idw(i, ctx, train_clim[ctx], DIST_MATRIX)

    n_train = len(train_idx)
    D_tt = DIST_MATRIX[np.ix_(train_idx, train_idx)]
    W_tt = 1.0 / (D_tt ** 2 + 1e-12); np.fill_diagonal(W_tt, 0.0)
    D_vt = DIST_MATRIX[np.ix_(val_idx, train_idx)]
    W_vt = 1.0 / (D_vt ** 2 + 1e-12)
    D_te_t = DIST_MATRIX[np.ix_(test_idx, train_idx)]
    W_te_t = 1.0 / (D_te_t ** 2 + 1e-12)

    def _idw_train(yr):
        tt = np.full(n_train, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        yvals = dict(zip(sub["station"].values, sub[target_col].values))
        for k, si in enumerate(train_idx):
            st = station_list_all[si]
            if st in yvals: tt[k] = yvals[st]
        m = ~np.isnan(tt)
        if not m.any(): return np.zeros(n_train, dtype=np.float32)
        W = W_tt * m[None, :]; dn = W.sum(axis=1)
        num = W @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt))
        return np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    def _idw_query(q_idx, W_q_base, yr):
        tt = np.full(n_train, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        yvals = dict(zip(sub["station"].values, sub[target_col].values))
        for k, si in enumerate(train_idx):
            st = station_list_all[si]
            if st in yvals: tt[k] = yvals[st]
        m = ~np.isnan(tt)
        if not m.any(): return np.zeros(len(q_idx), dtype=np.float32)
        W = W_q_base * m[None, :]; dn = W.sum(axis=1)
        num = W @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt))
        return np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    hist_years = [int(y) for y in years_all if int(y) != yr_test]
    idw_train_by_yr = {yr: _idw_train(yr) for yr in hist_years}
    idw_val_by_yr = {yr: _idw_query(val_idx, W_vt, yr) for yr in hist_years}
    idw_test_by_yr = {yr: _idw_query(test_idx, W_te_t, yr) for yr in hist_years}
    idw_test_final = _idw_query(test_idx, W_te_t, yr_test)

    X_source_by_yr = {}
    for yr in hist_years:
        temp_arr = np.full(n_stations, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        for st, t in zip(sub["station"].values, sub[target_col].values):
            temp_arr[station_idx[st]] = np.float32(t)
        temp_arr = np.where(np.isnan(temp_arr), clim_arr, temp_arr).astype(np.float32)

        X = np.zeros((n_stations, 7), dtype=np.float32)
        X[:, 0] = (temp_arr - global_mean) / 20.0
        X[val_idx, 0] = (idw_val_by_yr[yr] - global_mean) / 20.0
        X[test_idx, 0] = (idw_test_by_yr[yr] - global_mean) / 20.0
        X[:, 1] = lon_norm; X[:, 2] = lat_norm; X[:, 3] = elev_norm
        X[:, 4] = (clim_arr - clim_mean) / clim_std
        X[:, 5] = 0.0; X[:, 6] = 0.0
        X[train_idx, 5] = 1.0
        X_source_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    X_val_by_yr = {}
    for yr in hist_years:
        temp_arr = np.full(n_stations, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        for st, t in zip(sub["station"].values, sub[target_col].values):
            temp_arr[station_idx[st]] = np.float32(t)
        temp_arr = np.where(np.isnan(temp_arr), clim_arr, temp_arr).astype(np.float32)

        X = np.zeros((n_stations, 7), dtype=np.float32)
        X[:, 0] = (temp_arr - global_mean) / 20.0
        X[val_idx, 0] = (idw_val_by_yr[yr] - global_mean) / 20.0
        X[test_idx, 0] = (idw_test_by_yr[yr] - global_mean) / 20.0
        X[:, 1] = lon_norm; X[:, 2] = lat_norm; X[:, 3] = elev_norm
        X[:, 4] = (clim_arr - clim_mean) / clim_std
        X[:, 5] = 0.0; X[:, 6] = 0.0
        X[train_idx, 5] = 1.0
        X[val_idx, 5] = 0.0; X[val_idx, 6] = 1.0
        X_val_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    temp_arr_test = np.full(n_stations, np.nan, dtype=np.float32)
    sub = df[df["year"] == yr_test]
    for st, t in zip(sub["station"].values, sub[target_col].values):
        temp_arr_test[station_idx[st]] = np.float32(t)
    temp_arr_test = np.where(np.isnan(temp_arr_test), clim_arr, temp_arr_test).astype(np.float32)

    X_test = np.zeros((n_stations, 7), dtype=np.float32)
    X_test[:, 0] = (temp_arr_test - global_mean) / 20.0
    X_test[test_idx, 0] = (idw_test_final - global_mean) / 20.0
    X_test[:, 1] = lon_norm; X_test[:, 2] = lat_norm; X_test[:, 3] = elev_norm
    X_test[:, 4] = (clim_arr - clim_mean) / clim_std
    X_test[:, 5] = 0.0; X_test[:, 6] = 0.0
    X_test[train_idx, 5] = 1.0
    X_test[test_idx, 5] = 0.0; X_test[test_idx, 6] = 1.0
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    def _get_y_for_stations(yr, st_idx_list):
        sub = df[df["year"] == yr]
        ydict = dict(zip(sub["station"].values, sub[target_col].values))
        return np.array([float(ydict.get(station_list_all[si], np.nan))
                         for si in st_idx_list], dtype=np.float32)

    y_train_by_yr = {yr: torch.tensor(_get_y_for_stations(yr, train_idx),
                                       dtype=torch.float32, device=DEVICE) for yr in hist_years}
    y_val_by_yr = {yr: _get_y_for_stations(yr, val_idx) for yr in hist_years}
    y_test = _get_y_for_stations(yr_test, test_idx)

    return dict(
        seed=seed, yr_test=yr_test, hist_years=hist_years,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        X_source_by_yr=X_source_by_yr, X_val_by_yr=X_val_by_yr, X_test=X_test_t,
        y_train_by_yr=y_train_by_yr, y_val_by_yr=y_val_by_yr, y_test=y_test,
        idw_train_by_yr=idw_train_by_yr, clim_except=clim_except,
        clim_mean=clim_mean, clim_std=clim_std, global_mean=global_mean,
    )


def train_and_eval_fast(prep, params, edge_index_dev, edge_weight_dev):
    seed = prep['seed']; np.random.seed(seed); torch.manual_seed(seed)

    train_idx = prep['train_idx']; val_idx = prep['val_idx']; test_idx = prep['test_idx']
    hist_years = prep['hist_years']; yr_test = prep['yr_test']
    X_source_by_yr = prep['X_source_by_yr']
    X_val_by_yr = prep['X_val_by_yr']
    X_test_t = prep['X_test']
    y_train_by_yr = prep['y_train_by_yr']
    y_val_by_yr = prep['y_val_by_yr']
    y_test = prep['y_test']
    idw_train_by_yr = prep['idw_train_by_yr']
    clim_except = prep['clim_except']
    clim_mean = prep['clim_mean']; clim_std = prep['clim_std']
    global_mean = prep['global_mean']

    n_train = len(train_idx)
    groups = [list(range(s, min(s + BATCH_K, n_train))) for s in range(0, n_train, BATCH_K)]

    encoder = GCNEncoder(in_channels=7, hidden_channels=GCN_HIDDEN, out_channels=GCN_OUT).to(DEVICE)
    predictor = TempPredictor(node_dim=GCN_OUT, hidden_dim=PRED_HIDDEN).to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=GCN_LR)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf"); wait = 0
    best_enc_sd = best_pred_sd = None

    for epoch in range(1, GCN_EPOCHS + 1):
        encoder.train(); predictor.train()

        for yr in hist_years:
            X_src = X_source_by_yr[yr]
            idw_yr = idw_train_by_yr[yr]
            y_yr = y_train_by_yr[yr]

            for group in groups:
                X = X_src.clone()
                for k in group:
                    si = int(train_idx[k])
                    X[si, 0] = float((float(idw_yr[k]) - global_mean) / 20.0)
                    X[si, 4] = float((float(clim_except[si]) - clim_mean) / clim_std)
                    X[si, 5] = 0.0
                    X[si, 6] = 1.0

                h_all = encoder(X, edge_index_dev, edge_weight_dev)
                target_stations = train_idx[group]
                preds = predictor(h_all[target_stations])
                y_true = y_yr[group]
                mask = ~torch.isnan(y_true)
                if mask.sum() == 0:
                    continue
                loss = criterion(preds[mask], y_true[mask])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), CLIP)
                optimizer.step()

        encoder.eval(); predictor.eval()
        pv, tv = [], []
        with torch.no_grad():
            for yr in hist_years:
                X = X_val_by_yr[yr]
                h_all = encoder(X, edge_index_dev, edge_weight_dev)
                preds = predictor(h_all[val_idx]).cpu().numpy()
                y_true = y_val_by_yr[yr]
                m = ~np.isnan(y_true)
                if m.any():
                    pv.extend(preds[m].tolist()); tv.extend(y_true[m].tolist())
        if not pv:
            continue
        rv = float(np.sqrt(mean_squared_error(tv, pv)))

        if rv < best_val_rmse:
            best_val_rmse = rv; wait = 0
            best_enc_sd = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_pred_sd = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
        else:
            wait += 1
            if wait >= GCN_PATIENCE:
                break

    if best_enc_sd is None:
        best_enc_sd = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        best_pred_sd = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}

    encoder.load_state_dict(best_enc_sd); predictor.load_state_dict(best_pred_sd)
    encoder.eval(); predictor.eval()

    with torch.no_grad():
        h_all = encoder(X_test_t, edge_index_dev, edge_weight_dev)
        preds = predictor(h_all[test_idx]).cpu().numpy()
    m = ~np.isnan(y_test)
    preds, y_true = preds[m], y_test[m]

    test_rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    test_mae = float(mean_absolute_error(y_true, preds))
    test_r2 = float(r2_score(y_true, preds)) if len(y_true) > 1 else float('nan')
    return test_rmse, test_mae, test_r2


if __name__ == "__main__":
    t0_global = time.time()

    df_full = load_data()
    years_all = sorted(df_full["year"].unique())
    station_year_counts = df_full.groupby("station")["year"].nunique()
    common_stations = sorted(station_year_counts[station_year_counts == len(years_all)].index.tolist())
    df = df_full[df_full["station"].isin(common_stations)].copy()

    station_list_all = common_stations
    n_stations = len(station_list_all)
    station_idx = {st: i for i, st in enumerate(station_list_all)}

    station_info = df.groupby("station").first()[["lon", "lat", "elevation"]]
    lons_raw = station_info.loc[station_list_all, "lon"].values.astype(np.float32)
    lats_raw = station_info.loc[station_list_all, "lat"].values.astype(np.float32)
    elevs_raw = station_info.loc[station_list_all, "elevation"].values.astype(np.float32)

    lon_mean, lon_std = lons_raw.mean(), lons_raw.std() + 1e-8
    lat_mean, lat_std = lats_raw.mean(), lats_raw.std() + 1e-8
    elev_mean, elev_std = elevs_raw.mean(), elevs_raw.std() + 1e-8
    lon_norm = ((lons_raw - lon_mean) / lon_std).astype(np.float32)
    lat_norm = ((lats_raw - lat_mean) / lat_std).astype(np.float32)
    elev_norm = ((elevs_raw - elev_mean) / elev_std).astype(np.float32)

    DIST_MATRIX = haversine_matrix(lons_raw, lats_raw)
    delaunay_edges = build_delaunay_edges(lons_raw, lats_raw)
    src, dst, w = [], [], []
    for (i, j) in delaunay_edges:
        dij = float(DIST_MATRIX[i, j])
        wij = 1.0 / (dij ** 2 + 1e-8)
        src += [i, j]; dst += [j, i]; w += [wij, wij]
    edge_index_dev = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    edge_weight_dev = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    print(f"Delaunay graph: {n_stations} nodes, {len(delaunay_edges)} edges")

    for target_name, target_col in TARGET_COLUMNS.items():
        print(f"\n{'=' * 80}")
        print(f"ST-GCN (fast batch) | {target_name} ({target_col})")
        print(f"{'=' * 80}")

        df_target = df.copy()
        df_target = df_target[(df_target[target_col] > -40) & (df_target[target_col] < 50)]

        results = []
        per_seed_rows = []

        for density in DENSITIES:
            print(f"\n===== Density {density * 100:.0f}% =====")
            rmse_list, mae_list, r2_list = [], [], []
            t0 = time.time()

            for seed in SEEDS:
                prep = prepare_seed_fast(seed, density, target_col, df_target, years_all,
                                          station_list_all, station_idx, n_stations, DIST_MATRIX,
                                          lon_norm, lat_norm, elev_norm)
                test_rmse, test_mae, test_r2 = train_and_eval_fast(
                    prep, {}, edge_index_dev, edge_weight_dev)
                rmse_list.append(test_rmse); mae_list.append(test_mae); r2_list.append(test_r2)
                per_seed_rows.append({"Target": target_name, "density": density, "seed": seed,
                                       "rmse": test_rmse, "mae": test_mae, "r2": test_r2})
                print(f"  Seed {seed}: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}")

            print(f"  [Density {density*100:.0f}% done in {time.time()-t0:.1f}s]")
            if len(rmse_list) > 0:
                results.append({
                    'Density': f"{density*100:.0f}%",
                    'RMSE_mean': np.mean(rmse_list), 'RMSE_std': np.std(rmse_list),
                    'MAE_mean': np.mean(mae_list), 'MAE_std': np.std(mae_list),
                    'R2_mean': np.mean(r2_list), 'R2_std': np.std(r2_list),
                })

        summary_df = pd.DataFrame(results).round(4)
        print(f"\n{'=' * 80}")
        print(f"ST-GCN (fast) Performance for {target_name}")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))

        summary_df.to_csv(f"stgcn_{target_name}_fast_results.csv", index=False)
        pd.DataFrame(per_seed_rows).to_csv(f"stgcn_{target_name}_fast_per_seed.csv", index=False)

    print(f"\nTotal time: {(time.time()-t0_global)/60:.1f} min")
