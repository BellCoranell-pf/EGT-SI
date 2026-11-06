import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "TAVE.csv"
TARGET_COLUMNS = {'Tave': 'Tave'}

DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = list(range(42, 52))
HISTORY_YEARS = 10
HISTORY_MODE = "after"

CONFIGS = [
    {'name': 'cfg1',
     'hidden': 32, 'out': 16, 'layers': 3, 'dropout': 0.0, 'weight_decay': 1e-5},
    {'name': 'cfg2',
     'hidden': 32, 'out': 16, 'layers': 2, 'dropout': 0.0, 'weight_decay': 1e-5},
    {'name': 'cfg3',
     'hidden': 16, 'out': 8, 'layers': 3, 'dropout': 0.0, 'weight_decay': 1e-5},
]

FIXED_PARAMS = {
    'heads': 4,
    'lr': 1e-3,
    'patience': 20,
    'epochs': 200,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


df_full = load_data()
years_all = sorted(df_full["year"].unique())
cnts = df_full.groupby("station")["year"].nunique()
common = sorted(cnts[cnts == len(years_all)].index.tolist())
df_raw = df_full[df_full["station"].isin(common)].copy()

station_list_all = common
n_stations = len(station_list_all)
station_idx = {st: i for i, st in enumerate(station_list_all)}

info = df_raw.groupby("station").first()[["lon", "lat", "elevation"]]
lons_raw = info.loc[station_list_all, "lon"].values.astype(np.float32)
lats_raw = info.loc[station_list_all, "lat"].values.astype(np.float32)
elevs_raw = info.loc[station_list_all, "elevation"].values.astype(np.float32)

lon_mu, lon_sd = lons_raw.mean(), lons_raw.std() + 1e-8
lat_mu, lat_sd = lats_raw.mean(), lats_raw.std() + 1e-8
elev_mu, elev_sd = elevs_raw.mean(), elevs_raw.std() + 1e-8
lon_norm = ((lons_raw - lon_mu) / lon_sd).astype(np.float32)
lat_norm = ((lats_raw - lat_mu) / lat_sd).astype(np.float32)
elev_norm = ((elevs_raw - elev_mu) / elev_sd).astype(np.float32)
COORDS_arr = np.stack([lons_raw, lats_raw], axis=1).astype(np.float32)


def haversine_matrix(lon, lat):
    R = 6371.0
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    dlon = lon_r[None, :] - lon_r[:, None]
    dlat = lat_r[None, :] - lat_r[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return (R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)


DIST_MATRIX = haversine_matrix(lons_raw, lats_raw)
year_to_idx = {int(y): i for i, y in enumerate(years_all)}


def split_stations_by_density(stations, density, seed):
    rng = np.random.RandomState(seed)
    sh = rng.permutation(stations)
    n = len(sh)
    n_val = int(n * density)
    n_test = int(n * density)
    n_train = n - n_val - n_test
    return (list(sh[:n_train]), list(sh[n_train:n_train + n_val]), list(sh[n_train + n_val:]))


def get_history_years(yr_test, all_years, n_years, mode="after"):
    all_years = sorted(int(y) for y in all_years)
    if mode == "after":
        cand = [y for y in all_years if y > yr_test]
        hist = cand[:n_years]
        if len(hist) < n_years:
            past = [y for y in all_years if y < yr_test][::-1]
            hist = hist + past[:n_years - len(hist)]
    else:
        cand = [y for y in all_years if y < yr_test][::-1]
        hist = cand[:n_years]
        if len(hist) < n_years:
            fut = [y for y in all_years if y > yr_test]
            hist = hist + fut[:n_years - len(hist)]
    return sorted(hist)


def compute_clim_idw(target_idx, context_idx, context_clim):
    if len(context_idx) == 0:
        return float(np.nanmean(context_clim)) if len(context_clim) else 0.0
    d = DIST_MATRIX[target_idx, context_idx]
    w = 1.0 / (d ** 2 + 1e-12)
    mask = ~np.isnan(context_clim)
    return float((w[mask] * context_clim[mask]).sum() / w[mask].sum()) if mask.any() else 0.0


class EncoderWithAttn(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, nhead,
                 dropout, year_to_idx):
        super().__init__()
        self.year_to_idx = year_to_idx
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.year_embedding = nn.Embedding(len(year_to_idx), hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                    dim_feedforward=hidden_dim * 4,
                                    dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, year, coords):
        h = self.input_proj(x)
        yidx = self.year_to_idx.get(int(year), 0)
        yemb = self.year_embedding(torch.tensor([yidx], dtype=torch.long, device=x.device))
        h = h + yemb.expand(h.size(0), -1) + self.coord_proj(coords)
        h = h.unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h.squeeze(0))


class TempPredictor(nn.Module):
    def __init__(self, node_dim, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.fc2(self.drop(self.relu(self.fc1(h)))).squeeze(-1)


def prepare_seed(seed, density, df, year_temp_array, history_years):
    np.random.seed(seed)
    torch.manual_seed(seed)
    yr_test = int(np.random.choice(years_all))
    hist_years = get_history_years(yr_test, years_all, history_years, HISTORY_MODE)

    train_st, val_st, test_st = split_stations_by_density(station_list_all, density, seed)
    train_idx = np.array([station_idx[s] for s in train_st], dtype=np.int64)
    val_idx = np.array([station_idx[s] for s in val_st], dtype=np.int64)
    test_idx = np.array([station_idx[s] for s in test_st], dtype=np.int64)

    df_h = df[df["year"].isin(hist_years)]
    train_df = df_h[df_h["station"].isin(train_st)].copy()

    train_clim = np.full(n_stations, np.nan, dtype=np.float32)
    for st, v in train_df.groupby("station")["temp"].mean().items():
        train_clim[station_idx[st]] = np.float32(v)
    valid = train_clim[train_idx]
    valid = valid[~np.isnan(valid)]
    clim_mean = float(valid.mean()) if len(valid) else 0.0
    clim_std = float(valid.std() + 1e-8) if len(valid) else 1.0
    global_mean = float(train_df["temp"].mean())

    clim_arr = train_clim.copy()
    non_train_idx = np.setdiff1d(np.arange(n_stations), train_idx)
    if len(non_train_idx) > 0 and len(train_idx) > 0:
        d_nt = DIST_MATRIX[non_train_idx][:, train_idx]
        w_nt = 1.0 / (d_nt ** 2 + 1e-12)
        mask = ~np.isnan(train_clim[train_idx])
        if mask.any():
            vals = train_clim[train_idx][mask]
            clim_arr[non_train_idx] = (w_nt[:, mask] @ vals) / w_nt[:, mask].sum(axis=1)
        else:
            clim_arr[non_train_idx] = global_mean
    clim_arr = np.where(np.isnan(clim_arr), global_mean, clim_arr).astype(np.float32)

    clim_except = np.zeros(n_stations, dtype=np.float32)
    for i in train_idx:
        ctx = train_idx[train_idx != i]
        clim_except[int(i)] = compute_clim_idw(i, ctx, train_clim[ctx])

    X_tpl = np.zeros((n_stations, 7), dtype=np.float32)
    X_tpl[:, 1] = lon_norm
    X_tpl[:, 2] = lat_norm
    X_tpl[:, 3] = elev_norm
    X_tpl[:, 4] = (clim_arr - clim_mean) / clim_std
    X_tpl[:, 5] = 0.0
    X_tpl[:, 6] = 0.0
    X_tpl[train_idx, 5] = 1.0

    D_tt = DIST_MATRIX[np.ix_(train_idx, train_idx)]
    W_tt = 1.0 / (D_tt ** 2 + 1e-12)
    np.fill_diagonal(W_tt, 0.0)
    idw_train = {}
    for yr in hist_years:
        tt = year_temp_array[yr][train_idx].astype(np.float32)
        m = ~np.isnan(tt)
        W_m = W_tt * m[None, :]
        dn = W_m.sum(axis=1)
        num = W_m @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt)) if m.any() else 0.0
        idw_train[yr] = np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    def _idw(q_idx, yr):
        tt = year_temp_array[yr][train_idx].astype(np.float32)
        m = ~np.isnan(tt)
        if not m.any():
            return np.full(len(q_idx), np.nanmean(tt) if len(tt) else 0.0, dtype=np.float32)
        D_q = DIST_MATRIX[np.ix_(q_idx, train_idx)]
        W_q = 1.0 / (D_q ** 2 + 1e-12) * m[None, :]
        dn = W_q.sum(axis=1)
        num = W_q @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt))
        return np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    idw_val = {yr: _idw(val_idx, yr) for yr in hist_years}
    idw_test = _idw(test_idx, yr_test)
    y_val = {yr: year_temp_array[yr][val_idx].astype(np.float32) for yr in hist_years}
    y_test = year_temp_array[yr_test][test_idx].astype(np.float32)

    return dict(seed=seed, yr_test=yr_test, hist_years=hist_years,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                X_tpl=X_tpl, clim_except=clim_except[train_idx],
                clim_arr_val=clim_arr[val_idx], clim_arr_test=clim_arr[test_idx],
                clim_mean=clim_mean, clim_std=clim_std, global_mean=global_mean,
                idw_train=idw_train, idw_val=idw_val, idw_test=idw_test,
                y_val=y_val, y_test=y_test, year_temp_array=year_temp_array)


def train_and_eval(prep, params):
    seed = prep['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va, te = prep['train_idx'], prep['val_idx'], prep['test_idx']
    X_tpl = prep['X_tpl']

    enc = EncoderWithAttn(7, params['hidden'], params['out'],
                          params['layers'], params['heads'], params['dropout'],
                          year_to_idx=year_to_idx).to(DEVICE)
    prd = TempPredictor(params['out'], 16, params['dropout']).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(prd.parameters()),
                     lr=params['lr'], weight_decay=params['weight_decay'])
    crit = nn.MSELoss()
    c_t = torch.tensor(COORDS_arr, dtype=torch.float32, device=DEVICE)

    X_tr_by_yr, y_tr_by_yr = {}, {}
    for yr in prep['hist_years']:
        X = X_tpl.copy()
        X[tr, 0] = (prep['idw_train'][yr] - prep['global_mean']) / 20.0
        X[tr, 4] = (prep['clim_except'] - prep['clim_mean']) / prep['clim_std']
        X[tr, 5] = 0.0
        X[tr, 6] = 1.0
        X_tr_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_tr_by_yr[yr] = torch.tensor(prep['year_temp_array'][yr][tr], dtype=torch.float32, device=DEVICE)

    X_va_by_yr = {}
    for yr in prep['hist_years']:
        X = X_tpl.copy()
        X[va, 0] = (prep['idw_val'][yr] - prep['global_mean']) / 20.0
        X[va, 4] = (prep['clim_arr_val'] - prep['clim_mean']) / prep['clim_std']
        X[va, 5] = 0.0
        X[va, 6] = 1.0
        X_va_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    X_te = X_tpl.copy()
    X_te[te, 0] = (prep['idw_test'] - prep['global_mean']) / 20.0
    X_te[te, 4] = (prep['clim_arr_test'] - prep['clim_mean']) / prep['clim_std']
    X_te[te, 5] = 0.0
    X_te[te, 6] = 1.0
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    best_v = float("inf")
    wait = 0
    best_ep = 0
    best_enc = best_prd = None

    for epoch in range(1, params['epochs'] + 1):
        enc.train()
        prd.train()
        for yr in prep['hist_years']:
            h = enc(X_tr_by_yr[yr], yr, c_t)
            p = prd(h[tr])
            yt = y_tr_by_yr[yr]
            m = ~torch.isnan(yt)
            if m.sum() == 0: continue
            loss = crit(p[m], yt[m])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prd.parameters(), 1.0)
            opt.step()

        if epoch % 2 != 0 and epoch != params['epochs']: continue

        enc.eval()
        prd.eval()
        pv, tv = [], []
        with torch.no_grad():
            for yr in prep['hist_years']:
                h = enc(X_va_by_yr[yr], yr, c_t)
                pp = prd(h[va]).cpu().numpy()
                yt = prep['y_val'][yr]
                m = ~np.isnan(yt)
                if m.any():
                    pv.extend(pp[m].tolist())
                    tv.extend(yt[m].tolist())
        if not pv: continue
        rv = float(np.sqrt(mean_squared_error(tv, pv)))
        if rv < best_v:
            best_v = rv
            wait = 0
            best_ep = epoch
            best_enc = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_prd = {k: v.detach().cpu().clone() for k, v in prd.state_dict().items()}
        else:
            wait += 1
            if wait >= params['patience']: break

    if best_enc is None:
        best_enc = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        best_prd = {k: v.detach().cpu().clone() for k, v in prd.state_dict().items()}

    enc.load_state_dict(best_enc)
    prd.load_state_dict(best_prd)
    enc.eval()
    prd.eval()

    with torch.no_grad():
        h = enc(X_te_t, prep['yr_test'], c_t)
        preds_all = prd(h[te]).cpu().numpy()
    yt_all = prep['y_test']
    m = ~np.isnan(yt_all)
    preds = preds_all[m]
    yt = yt_all[m]

    test_rmse = float(np.sqrt(mean_squared_error(yt, preds)))
    test_mae = float(mean_absolute_error(yt, preds))
    test_r2 = float(r2_score(yt, preds)) if len(yt) > 1 else float('nan')

    return best_v, test_rmse, test_mae, test_r2, best_ep


if __name__ == "__main__":
    t_start = time.time()

    target_name, target_col = list(TARGET_COLUMNS.items())[0]

    df = df_raw.copy()
    df["temp"] = df[target_col].astype(np.float32)
    df = df[(df["temp"] > -40) & (df["temp"] < 50)]

    yta = {}
    for yr in years_all:
        yr = int(yr)
        arr = np.full(n_stations, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        for st, t in zip(sub["station"].values, sub["temp"].values):
            arr[station_idx[st]] = np.float32(t)
        yta[yr] = arr

    print("Precomputing data splits for 10 seeds...")
    prep_cache = {}
    for seed in SEEDS:
        for density in DENSITIES:
            prep_cache[(seed, density)] = prepare_seed(seed, density, df, yta, HISTORY_YEARS)
    print(f"Precomputation done, time: {time.time() - t_start:.1f}s")

    comparison_rows = []

    for cfg in CONFIGS:
        cfg_name = cfg['name']
        print(f"\n{'=' * 70}")
        print(f"Running {target_name} - {cfg_name}: hidden={cfg['hidden']}, "
              f"out={cfg['out']}, layers={cfg['layers']}")
        print(f"{'=' * 70}")

        full_params = {**FIXED_PARAMS, **cfg}
        per_seed_rows = []
        t_cfg = time.time()

        for density in DENSITIES:
            print(f"  Density {density * 100:.0f}%")
            for seed in SEEDS:
                prep = prep_cache[(seed, density)]
                val_rmse, test_rmse, test_mae, test_r2, best_ep = train_and_eval(prep, full_params)

                per_seed_rows.append({
                    'Target': target_name, 'Config': cfg_name,
                    'Density': density, 'Seed': seed,
                    'val_rmse': val_rmse, 'test_rmse': test_rmse,
                    'test_mae': test_mae, 'test_r2': test_r2,
                    'best_epoch': best_ep
                })

        df_ps = pd.DataFrame(per_seed_rows)
        df_ps.to_csv(f"egtsi_Tave_{cfg_name}_per_seed.csv", index=False, encoding="utf-8-sig")

        summary_rows = []
        for density in DENSITIES:
            sub = df_ps[df_ps['Density'] == density]
            summary_rows.append({
                'Density': f"{density * 100:.0f}%",
                'RMSE_mean': sub['test_rmse'].mean(),
                'RMSE_std': sub['test_rmse'].std(),
                'MAE_mean': sub['test_mae'].mean(),
                'MAE_std': sub['test_mae'].std(),
                'R2_mean': sub['test_r2'].mean(),
                'R2_std': sub['test_r2'].std(),
            })

        total_rmse = df_ps['test_rmse'].mean()
        total_mae = df_ps['test_mae'].mean()
        total_r2 = df_ps['test_r2'].mean()

        summary_rows.append({
            'Density': 'Total',
            'RMSE_mean': total_rmse,
            'RMSE_std': np.nan,
            'MAE_mean': total_mae,
            'MAE_std': np.nan,
            'R2_mean': total_r2,
            'R2_std': np.nan,
        })

        df_sum = pd.DataFrame(summary_rows).round(4)
        df_sum.to_csv(f"egtsi_Tave_{cfg_name}_results.csv", index=False, encoding="utf-8-sig")

        print(f"\n  {cfg_name} Summary:")
        print(df_sum.to_string(index=False))
        print(f"  Time: {(time.time() - t_cfg) / 60:.1f} min")

        comparison_rows.append({
            'Config': cfg_name,
            'Hidden': cfg['hidden'],
            'Out': cfg['out'],
            'Layers': cfg['layers'],
            'Dropout': cfg['dropout'],
            'WD': cfg['weight_decay'],
            'Total_RMSE': total_rmse,
            'Total_MAE': total_mae,
            'Total_R2': total_r2,
        })

    df_cmp = pd.DataFrame(comparison_rows).sort_values('Total_RMSE').round(4)
    df_cmp.to_csv("tave_configs_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"\n\n{'=' * 80}")
    print("Final comparison of 3 TAVE configurations (sorted by Total RMSE ascending)")
    print(f"{'=' * 80}")
    print(df_cmp.to_string(index=False))
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} minutes")
    print("Saved: tave_configs_comparison.csv")
