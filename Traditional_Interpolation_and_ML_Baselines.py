import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    raise ImportError("Please install pykrige: pip install pykrige")

DATA_PATH = "TAVE.csv"
TARGET_COL = "Tave"
TARGET_NAME = "Tave"

DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = list(range(42, 52))
HISTORY_YEARS = 10
HISTORY_MODE = "after"

MODELS = ["idw", "ok", "rk", "rf", "xgb"]

RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9,
                  reg_alpha=0.0, reg_lambda=1.0, n_jobs=-1, verbosity=0)

VARIOGRAM_MODEL = "exponential"

print(f"Baseline 5 models | {TARGET_NAME} | densities={DENSITIES} | "
      f"seeds={SEEDS} | history={HISTORY_YEARS}y")


def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception:
        df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    df_raw.rename(columns={"station_id": "station", "longitude": "lon",
                           "latitude": "lat"}, inplace=True)
    df = df_raw[["year", "station", "lon", "lat", "elevation",
                 TARGET_COL]].dropna().copy()
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
idx_to_station = {i: st for st, i in station_idx.items()}

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


def haversine_matrix(lon, lat):
    R = 6371.0
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    dlon = lon_r[None, :] - lon_r[:, None]
    dlat = lat_r[None, :] - lat_r[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return (R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)


DIST_MATRIX = haversine_matrix(lons_raw, lats_raw)


def split_stations_by_density(stations, density, seed):
    rng = np.random.RandomState(seed)
    sh = rng.permutation(stations)
    n = len(sh)
    n_val = int(n * density)
    n_test = int(n * density)
    n_train = n - n_val - n_test
    return (list(sh[:n_train]),
            list(sh[n_train:n_train + n_val]),
            list(sh[n_train + n_val:]))


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


def build_prep(seed, density, df, year_temp_array, history_years):
    np.random.seed(seed)
    yr_test = int(np.random.choice(years_all))
    hist_years = get_history_years(yr_test, years_all, history_years, HISTORY_MODE)

    train_st, val_st, test_st = split_stations_by_density(station_list_all, density, seed)
    train_idx = np.array([station_idx[s] for s in train_st], dtype=np.int64)
    val_idx = np.array([station_idx[s] for s in val_st], dtype=np.int64)
    test_idx = np.array([station_idx[s] for s in test_st], dtype=np.int64)

    df_h = df[df["year"].isin(hist_years)]
    train_df = df_h[df_h["station"].isin(train_st)].copy()

    train_clim = np.full(n_stations, np.nan, dtype=np.float32)
    for st, v in train_df.groupby("station")[TARGET_COL].mean().items():
        train_clim[station_idx[st]] = np.float32(v)
    valid = train_clim[train_idx]
    valid = valid[~np.isnan(valid)]
    clim_mean = float(valid.mean()) if len(valid) else 0.0
    clim_std = float(valid.std() + 1e-8) if len(valid) else 1.0
    global_mean = float(train_df[TARGET_COL].mean())

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
        d = DIST_MATRIX[i, ctx]
        w = 1.0 / (d ** 2 + 1e-12)
        vals = train_clim[ctx]
        m = ~np.isnan(vals)
        clim_except[int(i)] = float((w[m] * vals[m]).sum() / w[m].sum()) if m.any() else global_mean

    D_tt = DIST_MATRIX[np.ix_(train_idx, train_idx)]
    W_tt = 1.0 / (D_tt ** 2 + 1e-12)
    np.fill_diagonal(W_tt, 0.0)

    def _idw_loo(yr):
        tt = year_temp_array[yr][train_idx].astype(np.float32)
        m = ~np.isnan(tt)
        W_m = W_tt * m[None, :]
        dn = W_m.sum(axis=1)
        num = W_m @ np.where(m, tt, 0.0).astype(np.float32)
        fb = float(np.nanmean(tt)) if m.any() else 0.0
        return np.where(dn > 1e-12, num / (dn + 1e-12), fb).astype(np.float32)

    def _idw_all(q_idx, yr):
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

    idw_train = {yr: _idw_loo(yr) for yr in hist_years}
    idw_val = {yr: _idw_all(val_idx, yr) for yr in hist_years}
    idw_test = _idw_all(test_idx, yr_test)
    y_val = {yr: year_temp_array[yr][val_idx].astype(np.float32) for yr in hist_years}
    y_test = year_temp_array[yr_test][test_idx].astype(np.float32)

    def _make_train():
        rows, ys = [], []
        for yr in hist_years:
            for k, i in enumerate(train_idx):
                rows.append([(idw_train[yr][k] - global_mean) / 20.0,
                             lon_norm[i], lat_norm[i], elev_norm[i],
                             (clim_except[int(i)] - clim_mean) / clim_std, 0.0, 1.0])
                ys.append(year_temp_array[yr][i])
        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        m = ~np.isnan(y)
        return X[m], y[m]

    def _make_val():
        rows, ys = [], []
        for yr in hist_years:
            for k, i in enumerate(val_idx):
                rows.append([(idw_val[yr][k] - global_mean) / 20.0,
                             lon_norm[i], lat_norm[i], elev_norm[i],
                             (clim_arr[i] - clim_mean) / clim_std, 0.0, 1.0])
                ys.append(year_temp_array[yr][i])
        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        m = ~np.isnan(y)
        return X[m], y[m]

    def _make_test():
        rows, ys = [], []
        for k, i in enumerate(test_idx):
            rows.append([(idw_test[k] - global_mean) / 20.0,
                         lon_norm[i], lat_norm[i], elev_norm[i],
                         (clim_arr[i] - clim_mean) / clim_std, 0.0, 1.0])
            ys.append(year_temp_array[yr_test][i])
        return np.asarray(rows, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    X_train, y_train = _make_train()
    X_val, y_val_arr = _make_val()
    X_test, y_test_arr = _make_test()

    return dict(
        seed=seed, yr_test=yr_test, hist_years=hist_years,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        idw_val=idw_val, idw_test=idw_test,
        y_val=y_val, y_test=y_test,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val_arr=y_val_arr,
        X_test=X_test, y_test_arr=y_test_arr,
        year_temp_array=year_temp_array,
    )


def run_idw(prep):
    pv, tv = [], []
    for yr in prep['hist_years']:
        p = prep['idw_val'][yr]
        y = prep['y_val'][yr]
        m = ~np.isnan(y)
        if m.any():
            pv.extend(p[m].tolist())
            tv.extend(y[m].tolist())
    val_rmse = float(np.sqrt(mean_squared_error(tv, pv))) if pv else float('nan')

    preds_all = prep['idw_test']
    y_all = prep['y_test']
    m = ~np.isnan(y_all)
    return val_rmse, preds_all[m], y_all[m], prep['test_idx'][m]


def run_ok(prep):
    tr = prep['train_idx']
    te = prep['test_idx']
    hist = prep['hist_years']
    yta = prep['year_temp_array']
    x_tr = lons_raw[tr]
    y_tr = lats_raw[tr]
    x_te = lons_raw[te]
    y_te = lats_raw[te]

    pv, tv = [], []
    for yr in hist:
        z_tr = yta[yr][tr].astype(np.float64)
        m = ~np.isnan(z_tr)
        if m.sum() < 4:
            pv.extend([np.nanmean(z_tr)] * len(prep['val_idx']))
            tv.extend(yta[yr][prep['val_idx']].tolist())
            continue
        try:
            ok = OrdinaryKriging(x_tr[m], y_tr[m], z_tr[m],
                                 variogram_model=VARIOGRAM_MODEL,
                                 verbose=False, enable_plotting=False)
            z_pred, _ = ok.execute('points',
                                   lons_raw[prep['val_idx']].astype(np.float64),
                                   lats_raw[prep['val_idx']].astype(np.float64))
            pv.extend(np.asarray(z_pred).tolist())
        except Exception:
            pv.extend([np.nanmean(z_tr)] * len(prep['val_idx']))
        tv.extend(yta[yr][prep['val_idx']].tolist())
    tv_arr = np.asarray(tv, dtype=np.float32)
    pv_arr = np.asarray(pv, dtype=np.float32)
    m = ~np.isnan(tv_arr)
    val_rmse = float(np.sqrt(mean_squared_error(tv_arr[m], pv_arr[m]))) if m.any() else float('nan')

    z_te = yta[prep['yr_test']][tr].astype(np.float64)
    m = ~np.isnan(z_te)
    if m.sum() < 4:
        preds_all = np.full(len(te), np.nanmean(z_te) if len(z_te) else 0.0, dtype=np.float32)
    else:
        try:
            ok = OrdinaryKriging(x_tr[m], y_tr[m], z_te[m],
                                 variogram_model=VARIOGRAM_MODEL,
                                 verbose=False, enable_plotting=False)
            z_pred, _ = ok.execute('points', x_te.astype(np.float64), y_te.astype(np.float64))
            preds_all = np.asarray(z_pred, dtype=np.float32)
        except Exception:
            preds_all = np.full(len(te), np.nanmean(z_te), dtype=np.float32)

    y_all = prep['y_test']
    m_test = ~np.isnan(y_all)
    return val_rmse, preds_all[m_test], y_all[m_test], te[m_test]


def run_rk(prep):
    tr = prep['train_idx']
    te = prep['test_idx']
    hist = prep['hist_years']
    yta = prep['year_temp_array']

    x_tr = lons_raw[tr].astype(np.float64)
    y_tr = lats_raw[tr].astype(np.float64)
    elev_tr = elevs_raw[tr].astype(np.float64)

    x_te = lons_raw[te].astype(np.float64)
    y_te = lats_raw[te].astype(np.float64)
    elev_te = elevs_raw[te].astype(np.float64)

    x_va = lons_raw[prep['val_idx']].astype(np.float64)
    y_va = lats_raw[prep['val_idx']].astype(np.float64)
    elev_va = elevs_raw[prep['val_idx']].astype(np.float64)

    pv, tv = [], []
    for yr in hist:
        z_tr = yta[yr][tr].astype(np.float64)
        m = ~np.isnan(z_tr)
        if m.sum() < 5:
            pv.extend([np.nanmean(z_tr)] * len(prep['val_idx']))
            tv.extend(yta[yr][prep['val_idx']].tolist())
            continue

        X_train = np.column_stack([np.ones(m.sum()), elev_tr[m]])
        y_train = z_tr[m]
        coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        trend_tr = coef[0] + coef[1] * elev_tr[m]
        trend_va = coef[0] + coef[1] * elev_va
        resid_tr = y_train - trend_tr

        try:
            ok = OrdinaryKriging(x_tr[m], y_tr[m], resid_tr,
                                 variogram_model=VARIOGRAM_MODEL,
                                 verbose=False, enable_plotting=False)
            z_resid, _ = ok.execute('points', x_va, y_va)
            resid_pred = np.array(z_resid.data if hasattr(z_resid, 'data') else z_resid)
        except Exception:
            resid_pred = np.zeros(len(prep['val_idx']))

        pred = trend_va + resid_pred
        pv.extend(pred.tolist())
        tv.extend(yta[yr][prep['val_idx']].tolist())

    tv_arr = np.asarray(tv, dtype=np.float32)
    pv_arr = np.asarray(pv, dtype=np.float32)
    m = ~np.isnan(tv_arr)
    val_rmse = float(np.sqrt(mean_squared_error(tv_arr[m], pv_arr[m]))) if m.any() else float('nan')

    z_te = yta[prep['yr_test']][tr].astype(np.float64)
    m = ~np.isnan(z_te)
    if m.sum() < 5:
        preds_all = np.full(len(te), np.nanmean(z_te) if len(z_te) else 0.0, dtype=np.float32)
    else:
        X_train = np.column_stack([np.ones(m.sum()), elev_tr[m]])
        y_train = z_te[m]
        coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        trend_tr = coef[0] + coef[1] * elev_tr[m]
        trend_te = coef[0] + coef[1] * elev_te
        resid_tr = y_train - trend_tr

        try:
            ok = OrdinaryKriging(x_tr[m], y_tr[m], resid_tr,
                                 variogram_model=VARIOGRAM_MODEL,
                                 verbose=False, enable_plotting=False)
            z_resid, _ = ok.execute('points', x_te, y_te)
            resid_pred = np.array(z_resid.data if hasattr(z_resid, 'data') else z_resid)
        except Exception:
            resid_pred = np.zeros(len(te))

        preds_all = (trend_te + resid_pred).astype(np.float32)

    y_all = prep['y_test']
    m_test = ~np.isnan(y_all)
    return val_rmse, preds_all[m_test], y_all[m_test], te[m_test]


def run_rf(prep, seed):
    model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    model.fit(prep['X_train'], prep['y_train'])
    p_val = model.predict(prep['X_val'])
    val_rmse = (float(np.sqrt(mean_squared_error(prep['y_val_arr'], p_val)))
                if len(prep['y_val_arr']) else float('nan'))
    p_test = model.predict(prep['X_test'])
    y_all = prep['y_test_arr']
    m = ~np.isnan(y_all)
    return val_rmse, p_test[m], y_all[m], prep['test_idx'][m]


def run_xgb(prep, seed):
    model = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
    model.fit(prep['X_train'], prep['y_train'], verbose=False)
    p_val = model.predict(prep['X_val'])
    val_rmse = (float(np.sqrt(mean_squared_error(prep['y_val_arr'], p_val)))
                if len(prep['y_val_arr']) else float('nan'))
    p_test = model.predict(prep['X_test'])
    y_all = prep['y_test_arr']
    m = ~np.isnan(y_all)
    return val_rmse, p_test[m], y_all[m], prep['test_idx'][m]


def run_model(model_name, prep, seed):
    if model_name == 'idw':
        return run_idw(prep)
    elif model_name == 'ok':
        return run_ok(prep)
    elif model_name == 'rk':
        return run_rk(prep)
    elif model_name == 'rf':
        return run_rf(prep, seed)
    elif model_name == 'xgb':
        return run_xgb(prep, seed)
    raise ValueError(model_name)


if __name__ == "__main__":
    df = df_raw.copy()
    df["temp"] = df[TARGET_COL].astype(np.float32)
    df = df[(df["temp"] > -60) & (df["temp"] < 80)]

    yta = {}
    for yr in years_all:
        yr = int(yr)
        arr = np.full(n_stations, np.nan, dtype=np.float32)
        sub = df[df["year"] == yr]
        for st, t in zip(sub["station"].values, sub["temp"].values):
            arr[station_idx[st]] = np.float32(t)
        yta[yr] = arr

    summary_rows = []
    per_seed_rows = []
    site_long_rows = []
    site_wide_rows = []
    preds_pool = {m: [] for m in MODELS}
    trues_pool = {m: [] for m in MODELS}
    t_start = time.time()

    for density in DENSITIES:
        print(f"\n{'=' * 70}\n===== Density {density * 100:.0f}% =====\n{'=' * 70}")
        t0 = time.time()

        for seed in SEEDS:
            prep = build_prep(seed, density, df, yta, HISTORY_YEARS)
            model_results = {}
            seed_msgs = []

            for model_name in MODELS:
                try:
                    val_rmse, preds, yt, te_idx = run_model(model_name, prep, seed)
                except Exception as e:
                    print(f"  [WARN] {model_name} seed={seed} density={density} failed: {e}")
                    continue

                test_rmse = float(np.sqrt(mean_squared_error(yt, preds)))
                test_mae = float(mean_absolute_error(yt, preds))
                test_r2 = float(r2_score(yt, preds)) if len(yt) > 1 else float('nan')

                model_results[model_name] = dict(
                    val_rmse=val_rmse, preds=preds, yt=yt, te_idx=te_idx,
                    test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2,
                )

                per_seed_rows.append(dict(
                    model=model_name, density=density, seed=seed,
                    yr_test=prep['yr_test'], val_rmse=val_rmse,
                    test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2,
                ))
                preds_pool[model_name].extend(preds.tolist())
                trues_pool[model_name].extend(yt.tolist())
                seed_msgs.append(f"{model_name}:{test_rmse:.3f}")

            print(f"  Seed {seed}: yr={prep['yr_test']}  |  " + "  ".join(seed_msgs))

            if density == 0.20 and len(model_results) > 0:
                base_model = list(model_results.keys())[0]
                te_idx = model_results[base_model]['te_idx']
                yt_common = model_results[base_model]['yt']

                for model_name, res in model_results.items():
                    for k in range(len(te_idx)):
                        st = idx_to_station[int(te_idx[k])]
                        site_long_rows.append(dict(
                            model=model_name, seed=seed, yr_test=prep['yr_test'],
                            station=st,
                            lon=float(lons_raw[station_idx[st]]),
                            lat=float(lats_raw[station_idx[st]]),
                            elev=float(elevs_raw[station_idx[st]]),
                            true_temp=float(yt_common[k]),
                            pred_temp=float(res['preds'][k]),
                            abs_error=abs(float(yt_common[k]) - float(res['preds'][k])),
                        ))

                for k in range(len(te_idx)):
                    st = idx_to_station[int(te_idx[k])]
                    row = dict(
                        seed=seed, yr_test=prep['yr_test'], station=st,
                        lon=float(lons_raw[station_idx[st]]),
                        lat=float(lats_raw[station_idx[st]]),
                        elev=float(elevs_raw[station_idx[st]]),
                        true_temp=float(yt_common[k]),
                    )
                    for model_name in model_results:
                        p = float(model_results[model_name]['preds'][k])
                        row[f'pred_{model_name}'] = p
                        row[f'abs_err_{model_name}'] = abs(float(yt_common[k]) - p)
                    site_wide_rows.append(row)

        for model_name in MODELS:
            rows = [r for r in per_seed_rows
                    if r['density'] == density and r['model'] == model_name]
            if not rows:
                continue
            summary_rows.append(dict(
                model=model_name, density=density,
                RMSE_mean=float(np.mean([r['test_rmse'] for r in rows])),
                RMSE_std=float(np.std([r['test_rmse'] for r in rows])),
                MAE_mean=float(np.mean([r['test_mae'] for r in rows])),
                MAE_std=float(np.std([r['test_mae'] for r in rows])),
                R2_mean=float(np.mean([r['test_r2'] for r in rows])),
                R2_std=float(np.std([r['test_r2'] for r in rows])),
            ))
        print(f"  [Density {density * 100:.0f}% done in {time.time() - t0:.1f}s]")

    for model_name in MODELS:
        if not preds_pool[model_name]:
            continue
        p = np.array(preds_pool[model_name])
        t = np.array(trues_pool[model_name])
        summary_rows.append(dict(
            model=model_name, density='Total',
            RMSE_mean=float(np.sqrt(mean_squared_error(t, p))), RMSE_std=np.nan,
            MAE_mean=float(mean_absolute_error(t, p)), MAE_std=np.nan,
            R2_mean=float(r2_score(t, p)), R2_std=np.nan,
        ))

    summary_df = pd.DataFrame(summary_rows)
    summary_df['density_label'] = summary_df['density'].apply(
        lambda d: d if d == 'Total' else f"{int(d * 100)}%")
    summary_df = summary_df[['model', 'density_label', 'RMSE_mean', 'RMSE_std',
                             'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']].round(4)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df['density_label'] = per_seed_df['density'].apply(lambda d: f"{int(d * 100)}%")
    per_seed_df = per_seed_df[['model', 'density_label', 'seed', 'yr_test',
                               'val_rmse', 'test_rmse', 'test_mae', 'test_r2']].round(4)

    site_long_df = pd.DataFrame(site_long_rows).round(4)
    site_wide_df = pd.DataFrame(site_wide_rows).round(4)

    if len(site_wide_df) > 0:
        agg_rows = []
        for st, g in site_wide_df.groupby('station'):
            row = dict(
                station=st,
                lon=float(g['lon'].iloc[0]), lat=float(g['lat'].iloc[0]),
                elev=float(g['elev'].iloc[0]), n_seeds=len(g),
                true_temp_mean=float(g['true_temp'].mean()),
                true_temp_std=float(g['true_temp'].std()),
            )
            for m in MODELS:
                if f'pred_{m}' not in g.columns:
                    continue
                row[f'pred_{m}_mean'] = float(g[f'pred_{m}'].mean())
                row[f'bias_{m}'] = float((g[f'pred_{m}'] - g['true_temp']).mean())
                row[f'MAE_{m}'] = float(g[f'abs_err_{m}'].mean())
                row[f'RMSE_{m}'] = float(np.sqrt((g[f'abs_err_{m}'] ** 2).mean()))
            maes = {m: row[f'MAE_{m}'] for m in MODELS if f'MAE_{m}' in row}
            if maes:
                row['best_model_by_MAE'] = min(maes, key=maes.get)
            agg_rows.append(row)
        site_agg_df = pd.DataFrame(agg_rows).round(4)
    else:
        site_agg_df = pd.DataFrame()

    print(f"\n\n{'=' * 80}")
    print(f"Baseline 5 Models — {TARGET_NAME}")
    print(f"{'=' * 80}")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(f"baseline5_{TARGET_NAME}_multi_density_results.csv", index=False)
    per_seed_df.to_csv(f"baseline5_{TARGET_NAME}_per_seed_results.csv", index=False)
    site_long_df.to_csv(f"baseline5_{TARGET_NAME}_20pct_site_long.csv", index=False)
    site_wide_df.to_csv(f"baseline5_{TARGET_NAME}_20pct_site_wide.csv", index=False)
    site_agg_df.to_csv(f"baseline5_{TARGET_NAME}_20pct_site_agg.csv", index=False)

    print(f"\n\nSaved:")
    print(f"  baseline5_{TARGET_NAME}_multi_density_results.csv")
    print(f"  baseline5_{TARGET_NAME}_per_seed_results.csv")
    print(f"  baseline5_{TARGET_NAME}_20pct_site_long.csv  ({len(site_long_df)} rows)")
    print(f"  baseline5_{TARGET_NAME}_20pct_site_wide.csv  ({len(site_wide_df)} rows)")
    print(f"  baseline5_{TARGET_NAME}_20pct_site_agg.csv   ({len(site_agg_df)} rows)")
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
