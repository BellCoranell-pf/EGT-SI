import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykrige.ok import OrdinaryKriging

DATA_PATH = "TAVE.csv"
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = list(range(42, 52))
MAIN_VARIOGRAM = 'exponential'

TARGET_COLUMNS = {
    'Tave': 'Tave'
}

def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[
        'year', 'station_id', 'elevation', 'longitude', 'latitude', 'Tave'
    ]).copy()
    df['year'] = df['year'].astype(int)
    df['station_id'] = df['station_id'].astype(str)
    return df

df_full = load_data()
years_all = sorted(df_full['year'].unique())

station_counts = df_full.groupby('station_id')['year'].nunique()
common_stations = station_counts[station_counts == len(years_all)].index.tolist()
common_stations = sorted(common_stations)
df = df_full[df_full['station_id'].isin(common_stations)].copy()

print(f"Data loaded successfully: {len(common_stations)} common stations, {len(years_all)} years.")

def split_stations_by_density(stations, density, seed):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(stations)
    n = len(shuffled)
    n_val = int(n * density)
    n_test = int(n * density)
    n_train = n - n_val - n_test
    train_st = shuffled[:n_train]
    val_st = shuffled[n_train:n_train + n_val]
    test_st = shuffled[n_train + n_val:]
    return list(train_st), list(val_st), list(test_st)

def run_rk(seed, density, target_col):
    rng = np.random.RandomState(seed)
    yr_test = int(rng.choice(years_all))

    train_st, _, test_st = split_stations_by_density(common_stations, density, seed)

    df_year = df[df['year'] == yr_test]
    train_df = df_year[df_year['station_id'].isin(train_st)].copy()
    test_df = df_year[df_year['station_id'].isin(test_st)].copy()

    if len(train_df) < 5 or len(test_df) == 0:
        return None

    X_train = np.column_stack([np.ones(len(train_df)), train_df['elevation'].values])
    y_train = train_df[target_col].values.astype(np.float64)

    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    trend_train = coef[0] + coef[1] * train_df['elevation'].values
    trend_test = coef[0] + coef[1] * test_df['elevation'].values

    resid_train = y_train - trend_train

    x_tr = train_df['longitude'].values.astype(np.float64)
    y_tr = train_df['latitude'].values.astype(np.float64)
    x_te = test_df['longitude'].values.astype(np.float64)
    y_te = test_df['latitude'].values.astype(np.float64)

    try:
        ok = OrdinaryKriging(x_tr, y_tr, resid_train,
                             variogram_model=MAIN_VARIOGRAM,
                             verbose=False, enable_plotting=False)
        z_resid, _ = ok.execute('points', x_te, y_te)
        resid_pred = np.array(z_resid.data if hasattr(z_resid, 'data') else z_resid)
    except Exception:
        resid_pred = np.zeros(len(test_df))

    pred = trend_test + resid_pred
    true = test_df[target_col].values

    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    return rmse, mae, r2, true, pred

def main():
    per_seed_rows = []
    summary_rows = []

    for target_name, col_name in TARGET_COLUMNS.items():
        print(f"\n{'=' * 60}")
        print(f"Running RK baseline for: {target_name} ({col_name})")
        print(f"{'=' * 60}")

        all_preds_total = []
        all_trues_total = []

        for density in DENSITIES:
            print(f"  --- Density {density * 100:.0f}% ---")
            rmse_list, mae_list, r2_list = [], [], []
            dens_preds, dens_trues = [], []

            for seed in SEEDS:
                res = run_rk(seed, density, col_name)
                if res is None:
                    continue

                rmse, mae, r2, true_arr, pred_arr = res
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)

                dens_preds.extend(pred_arr.tolist())
                dens_trues.extend(true_arr.tolist())

                per_seed_rows.append({
                    'Target': target_name,
                    'Density': density,
                    'Seed': seed,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })
                print(f"    Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            if len(rmse_list) > 0:
                summary_rows.append({
                    'Target': target_name,
                    'Density': density,
                    'RMSE_mean': np.mean(rmse_list),
                    'RMSE_std': np.std(rmse_list),
                    'MAE_mean': np.mean(mae_list),
                    'MAE_std': np.std(mae_list),
                    'R2_mean': np.mean(r2_list),
                    'R2_std': np.std(r2_list),
                    'Overall_RMSE': np.sqrt(mean_squared_error(dens_trues, dens_preds)),
                    'Overall_MAE': mean_absolute_error(dens_trues, dens_preds),
                    'Overall_R2': r2_score(dens_trues, dens_preds)
                })

                all_preds_total.extend(dens_preds)
                all_trues_total.extend(dens_trues)

        if len(all_trues_total) > 0:
            total_rmse = np.sqrt(mean_squared_error(all_trues_total, all_preds_total))
            total_mae = mean_absolute_error(all_trues_total, all_preds_total)
            total_r2 = r2_score(all_trues_total, all_preds_total)

            summary_rows.append({
                'Target': target_name,
                'Density': 'Total',
                'RMSE_mean': total_rmse,
                'RMSE_std': np.nan,
                'MAE_mean': total_mae,
                'MAE_std': np.nan,
                'R2_mean': total_r2,
                'R2_std': np.nan,
                'Overall_RMSE': total_rmse,
                'Overall_MAE': total_mae,
                'Overall_R2': total_r2
            })

    df_per_seed = pd.DataFrame(per_seed_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_per_seed.to_csv("rk_per_seed.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv("rk_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("RK Baseline Results Summary (Simple Random Split)")
    print("=" * 80)
    for target_name in TARGET_COLUMNS.keys():
        print(f"\n>>> {target_name} <<<")
        sub = df_summary[df_summary['Target'] == target_name].copy()

        def fmt_density(x):
            if x == 'Total': return 'Total'
            return f"{int(x * 100)}%"

        sub['Density_pct'] = sub['Density'].apply(fmt_density)
        sub = sub[['Density_pct', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']]

        print(sub.round(4).to_string(index=False))

    print("\nSaved:")
    print("  rk_per_seed.csv")
    print("  rk_summary.csv")


if __name__ == "__main__":
    main()
