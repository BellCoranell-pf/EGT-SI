import warnings

warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "TAVE.csv"
TARGET_COL = 'Tave'
DENSITY = 0.20
SEEDS = list(range(42, 52))
HISTORY_YEARS = 10
HISTORY_MODE = "after"

BASE_CFG = {
    'hidden': 16, 'out': 8, 'layers': 3, 'heads': 4,
    'dropout': 0.0, 'lr': 1e-3, 'weight_decay': 1e-5,
    'patience': 20, 'epochs': 200,
}

ABLATIONS = [
    ('full', 'Full model', None),
    ('no_year', 'w/o Year Embedding', 'no_year'),
    ('no_attn_identity', 'w/o Attention (Identity)', 'no_attn_identity'),
    ('no_attn_mlp', 'w/o Attention (MLP)', 'no_attn_mlp'),
    ('no_idw', 'w/o IDW Prior', 'no_idw'),
    ('no_clim', 'w/o Climatology', 'no_clim'),
    ('no_idw_clim', 'w/o IDW & Climatology', 'no_idw_clim'),
    ('no_elev', 'w/o Elevation', 'no_elev'),
    ('no_lon', 'w/o Longitude (both paths)', 'no_lon'),
    ('no_lat', 'w/o Latitude (both paths)', 'no_lat'),
    ('no_coord_feature_only', 'w/o Coord Feature (keep injection)', 'no_coord_feature_only'),
    ('no_coord_injection_only', 'w/o Coord Injection (keep feature)', 'no_coord_injection_only'),
    ('no_coord_all', 'w/o All Coordinates', 'no_coord_all'),
]

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
df = df_full[df_full["station"].isin(common)].copy()

station_list_all = common
n_stations = len(station_list_all)
station_idx = {st: i for i, st in enumerate(station_list_all)}

info = df.groupby("station").first()[["lon", "lat", "elevation"]]
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


class AblationEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, nhead,
                 dropout, year_to_idx, ablation_type):
        super().__init__()
        self.ablation_type = ablation_type
        self.year_to_idx = year_to_idx

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.use_year_emb = (ablation_type != 'no_year')
        if self.use_year_emb:
            self.year_embedding = nn.Embedding(len(year_to_idx), hidden_dim)
        else:
            self.year_embedding = None

        no_injection_types = {'no_coord_injection_only', 'no_coord_all'}
        if ablation_type in no_injection_types:
            self.coord_proj = None
        else:
            self.coord_proj = nn.Linear(2, hidden_dim)

        self.mask_lon = (ablation_type == 'no_lon')
        self.mask_lat = (ablation_type == 'no_lat')

        if ablation_type == 'no_attn_identity':
            self.layers = None
            self.encoder_mlp = None
        elif ablation_type == 'no_attn_mlp':
            self.layers = None
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                        dim_feedforward=hidden_dim * 4,
                                        dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.encoder_mlp = None

        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, year, coords):
        h = self.input_proj(x)

        if self.year_embedding is not None:
            yidx = self.year_to_idx.get(int(year), 0)
            yemb = self.year_embedding(torch.tensor([yidx], dtype=torch.long, device=x.device))
            h = h + yemb.expand(h.size(0), -1)

        if self.coord_proj is not None:
            coords_masked = coords.clone()
            if self.mask_lon: coords_masked[:, 0] = 0.0
            if self.mask_lat: coords_masked[:, 1] = 0.0
            h = h + self.coord_proj(coords_masked)

        if self.ablation_type == 'no_attn_identity':
            return self.output_proj(h)
        if self.ablation_type == 'no_attn_mlp':
            h = self.encoder_mlp(h)
            return self.output_proj(h)

        h = h.unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h.squeeze(0))


class TempPredictor(nn.Module):
    def __init__(self, node_dim, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        if h.dim() == 1: h = h.unsqueeze(0)
        return self.fc2(self.drop(self.relu(self.fc1(h)))).squeeze(-1)


def prepare_seed(seed, density, df, yta, history_years):
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
        tt = yta[yr][train_idx].astype(np.float32)
        m = ~np.isnan(tt)
        W_m = W_tt * m[None, :]
        dn = W_m.sum(axis=1)
        num = W_m @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt)) if m.any() else 0.0
        idw_train[yr] = np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    def _idw(q_idx, yr):
        tt = yta[yr][train_idx].astype(np.float32)
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
    y_val = {yr: yta[yr][val_idx].astype(np.float32) for yr in hist_years}
    y_test = yta[yr_test][test_idx].astype(np.float32)

    return dict(seed=seed, yr_test=yr_test, hist_years=hist_years,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                X_tpl=X_tpl, clim_except=clim_except[train_idx],
                clim_arr_val=clim_arr[val_idx], clim_arr_test=clim_arr[test_idx],
                clim_mean=clim_mean, clim_std=clim_std, global_mean=global_mean,
                idw_train=idw_train, idw_val=idw_val, idw_test=idw_test,
                y_val=y_val, y_test=y_test, yta=yta)


def train_and_eval(prep, params, ablation_type):
    seed = prep['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va, te = prep['train_idx'], prep['val_idx'], prep['test_idx']
    X_tpl = prep['X_tpl'].copy()

    if ablation_type in ('no_idw', 'no_idw_clim'): X_tpl[:, 0] = 0.0
    if ablation_type in ('no_lon', 'no_coord_feature_only', 'no_coord_all'): X_tpl[:, 1] = 0.0
    if ablation_type in ('no_lat', 'no_coord_feature_only', 'no_coord_all'): X_tpl[:, 2] = 0.0
    if ablation_type == 'no_elev': X_tpl[:, 3] = 0.0
    if ablation_type in ('no_clim', 'no_idw_clim'): X_tpl[:, 4] = 0.0

    enc = AblationEncoder(
        in_channels=7, hidden_dim=params['hidden'], out_channels=params['out'],
        num_layers=params['layers'], nhead=params['heads'], dropout=params['dropout'],
        year_to_idx=year_to_idx, ablation_type=ablation_type,
    ).to(DEVICE)
    prd = TempPredictor(params['out'], 16, params['dropout']).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(prd.parameters()),
                     lr=params['lr'], weight_decay=params['weight_decay'])
    crit = nn.MSELoss()
    c_t = torch.tensor(COORDS_arr, dtype=torch.float32, device=DEVICE)

    X_tr_by_yr, y_tr_by_yr = {}, {}
    for yr in prep['hist_years']:
        X = X_tpl.copy()
        if ablation_type not in ('no_idw', 'no_idw_clim'):
            X[tr, 0] = (prep['idw_train'][yr] - prep['global_mean']) / 20.0
        if ablation_type not in ('no_clim', 'no_idw_clim'):
            X[tr, 4] = (prep['clim_except'] - prep['clim_mean']) / prep['clim_std']
        X[tr, 5] = 0.0
        X[tr, 6] = 1.0
        X_tr_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_tr_by_yr[yr] = torch.tensor(prep['yta'][yr][tr], dtype=torch.float32, device=DEVICE)

    X_va_by_yr = {}
    for yr in prep['hist_years']:
        X = X_tpl.copy()
        if ablation_type not in ('no_idw', 'no_idw_clim'):
            X[va, 0] = (prep['idw_val'][yr] - prep['global_mean']) / 20.0
        if ablation_type not in ('no_clim', 'no_idw_clim'):
            X[va, 4] = (prep['clim_arr_val'] - prep['clim_mean']) / prep['clim_std']
        X[va, 5] = 0.0
        X[va, 6] = 1.0
        X_va_by_yr[yr] = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    X_te = X_tpl.copy()
    if ablation_type not in ('no_idw', 'no_idw_clim'):
        X_te[te, 0] = (prep['idw_test'] - prep['global_mean']) / 20.0
    if ablation_type not in ('no_clim', 'no_idw_clim'):
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

    return test_rmse, test_mae, test_r2, best_ep


if __name__ == "__main__":
    t_start = time.time()
    print(f"{'=' * 70}\nEGT-SI Ablation Study (TAVE, cfg3 Small Capacity)\n{'=' * 70}")
    print(f"Target: {TARGET_COL}, Density: {DENSITY}, Seeds: {len(SEEDS)}")

    df["temp"] = df[TARGET_COL].astype(np.float32)
    df = df[(df["temp"] > -40) & (df["temp"] < 50)]

    yta = {}
    for yr in years_all:
        yr = int(yr)
        arr = np.full(n_stations, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        for st, t in zip(sub["station"].values, sub["temp"].values):
            arr[station_idx[st]] = np.float32(t)
        yta[yr] = arr

    print("Precomputing data splits...")
    preps = {seed: prepare_seed(seed, DENSITY, df, yta, HISTORY_YEARS) for seed in SEEDS}
    print(f"Done, time: {time.time() - t_start:.1f}s")

    all_rows = []

    for abl_key, abl_label, abl_type in ABLATIONS:
        abl_type_actual = 'full' if abl_key == 'full' else abl_type
        print(f"\n[{abl_label}]")
        t0 = time.time()

        for seed in SEEDS:
            prep = preps[seed]
            test_rmse, test_mae, test_r2, best_ep = train_and_eval(prep, BASE_CFG, abl_type_actual)
            all_rows.append({
                'ablation': abl_key, 'label': abl_label,
                'seed': seed, 'yr_test': prep['yr_test'],
                'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2,
                'best_epoch': best_ep
            })
            print(f"  Seed={seed}: RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  R2={test_r2:.4f}  (ep={best_ep})")

        print(f"  Time: {time.time() - t0:.1f}s")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv('ablation_Tave_per_seed.csv', index=False, encoding='utf-8-sig')

    summary_rows = []
    for abl_key, abl_label, _ in ABLATIONS:
        sub = df_all[df_all['ablation'] == abl_key]
        summary_rows.append({
            'ablation': abl_key, 'label': abl_label,
            'rmse_mean': sub['rmse'].mean(), 'rmse_std': sub['rmse'].std(),
            'mae_mean': sub['mae'].mean(), 'r2_mean': sub['r2'].mean()
        })
    df_summary = pd.DataFrame(summary_rows)

    full_rmse = df_summary.loc[df_summary['ablation'] == 'full', 'rmse_mean'].values[0]
    df_summary['delta_rmse'] = df_summary['rmse_mean'] - full_rmse
    df_summary = df_summary.round(4)

    print(f"\n\n{'=' * 80}")
    print("ABLATION TABLE (TAVE)")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<35} {'RMSE':>15} {'ΔRMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"{'-' * 80}")
    for _, row in df_summary.iterrows():
        marker = ' <- Full' if row['ablation'] == 'full' else ''
        print(f"{row['label']:<35} "
              f"{row['rmse_mean']:.3f}±{row['rmse_std']:.3f} "
              f"{row['delta_rmse']:>+9.3f} "
              f"{row['mae_mean']:>9.3f} "
              f"{row['r2_mean']:>9.4f}{marker}")

    df_summary.to_csv('ablation_Tave_results.csv', index=False, encoding='utf-8-sig')

    full_ps = df_all[df_all['ablation'] == 'full'].set_index('seed')['rmse']
    pair_rows = []
    print(f"\n{'=' * 80}\nPAIRWISE T-TEST (vs Full)\n{'=' * 80}")
    print(f"{'Configuration':<35} {'ΔRMSE':>10} {'Wins':>8} {'p-value':>10} {'Sig?':>8}")
    print(f"{'-' * 80}")

    for abl_key, abl_label, _ in ABLATIONS:
        if abl_key == 'full': continue
        sub_ps = df_all[df_all['ablation'] == abl_key].set_index('seed')['rmse']
        d = sub_ps - full_ps
        t_stat, p_val = stats.ttest_rel(sub_ps.values, full_ps.values)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))

        print(f"{abl_label:<35} {d.mean():>+9.3f} {int((d > 0).sum()):>3}/{len(d):>2} "
              f"{p_val:>9.4f} {sig:>8}")

        pair_rows.append({
            'ablation': abl_key, 'label': abl_label,
            'delta_rmse': float(d.mean()), 'delta_std': float(d.std()),
            'wins_full': int((d > 0).sum()), 'total': len(d),
            't_stat': float(t_stat), 'p_value': float(p_val), 'significant': p_val < 0.05
        })

    df_pair = pd.DataFrame(pair_rows).round(4)
    df_pair.to_csv('ablation_Tave_pairwise.csv', index=False, encoding='utf-8-sig')

    print(f"\n\nTotal time: {(time.time() - t_start) / 60:.1f} min")
