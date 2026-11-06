import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import spearmanr

DATA_PATH = "TAVE.csv"
DENSITY = 0.20
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
FIXED_YEAR = 2023
CKPT_DIR = "checkpoints"
OUT_DIR = f"attention_final1_d{int(DENSITY * 100)}_year{FIXED_YEAR}"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

EGT_HIDDEN = 32
EGT_OUT = 8
EGT_LAYERS = 2
EGT_HEADS = 4
EGT_DROPOUT = 0.1063

DIST_BINS = [0, 200, 500, 1000, 1500, 2000, 3000, 10000]
DIST_LABELS = ['0-200', '200-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    except:
        df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    df_raw.rename(columns={"station_id": "station", "longitude": "lon",
                           "latitude": "lat", "Tave": "temp"}, inplace=True)
    df = df_raw[["year", "station", "lon", "lat", "elevation", "temp"]].dropna().copy()
    df["year"] = df["year"].astype(int)
    df["station"] = df["station"].astype(str).str.strip()
    df["temp"] = df["temp"].astype(np.float32)
    return df[(df["temp"] > -60) & (df["temp"] < 80)]

df_full = load_data()
years_all = sorted(df_full["year"].unique())
station_year_counts = df_full.groupby("station")["year"].nunique()
common_stations = sorted(station_year_counts[station_year_counts == len(years_all)].index.tolist())
df = df_full[df_full["station"].isin(common_stations)].copy()

n_stations = len(common_stations)
station_idx = {st: i for i, st in enumerate(common_stations)}
station_info = df.groupby("station").first()[["lon", "lat", "elevation"]]
lons_raw = station_info.loc[common_stations, "lon"].values.astype(np.float32)
lats_raw = station_info.loc[common_stations, "lat"].values.astype(np.float32)
elevs_raw = station_info.loc[common_stations, "elevation"].values.astype(np.float32)

lon_mean, lon_std = lons_raw.mean(), lons_raw.std() + 1e-8
lat_mean, lat_std = lats_raw.mean(), lats_raw.std() + 1e-8
elev_mean, elev_std = elevs_raw.mean(), elevs_raw.std() + 1e-8
lon_norm = (lons_raw - lon_mean) / lon_std
lat_norm = (lats_raw - lat_mean) / lat_std
elev_norm = (elevs_raw - elev_mean) / elev_std
COORDS_arr = np.stack([lons_raw, lats_raw], axis=1)

year_to_idx = {int(y): i for i, y in enumerate(years_all)}
year_temp_array = {int(yr): np.full(n_stations, np.nan, dtype=np.float32) for yr in years_all}
for yr in years_all:
    sub = df[df["year"] == yr]
    for st, t in zip(sub["station"].values, sub["temp"].values):
        year_temp_array[int(yr)][station_idx[st]] = np.float32(t)

def haversine_matrix(lon, lat):
    R = 6371.0
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    dlon = lon_r[None, :] - lon_r[:, None]
    dlat = lat_r[None, :] - lat_r[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return (R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)

DIST_MATRIX = haversine_matrix(lons_raw, lats_raw)
DIST_TRAIN_NORM = DIST_MATRIX / (DIST_MATRIX.max() + 1e-8)
DIST_TRAIN_T = torch.tensor(DIST_TRAIN_NORM, dtype=torch.float32, device=DEVICE)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.distance_penalty = nn.Parameter(torch.tensor(5.0))

    def forward(self, src, dist_matrix, return_attn=False):
        attn_mask = -self.distance_penalty * dist_matrix
        if return_attn:
            src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask, need_weights=True,
                                        average_attn_weights=True)
        else:
            src2, _ = self.self_attn(src, src, src, attn_mask=attn_mask)
            attn = None
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        if return_attn: return src, attn
        return src

class EGTEncoder(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, n_layers, nhead, dropout, year_to_idx):
        super().__init__()
        self.year_to_idx = year_to_idx
        self.input_proj = nn.Linear(in_ch, hidden)
        self.year_emb = nn.Embedding(len(year_to_idx), hidden)
        self.coord_proj = nn.Linear(2, hidden)
        self.layers = nn.ModuleList(
            [SpatialAttentionLayer(hidden, nhead, hidden * 4, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, out_ch)

    def forward(self, x, year, coords, dist_matrix, return_attn=False):
        h = self.input_proj(x)
        yidx = self.year_to_idx.get(int(year), 0)
        yemb = self.year_emb(torch.tensor([yidx], dtype=torch.long, device=x.device))
        h = h + yemb.expand(h.size(0), -1) + self.coord_proj(coords)
        h = h.unsqueeze(0)
        attn_weights = None
        for layer in self.layers:
            if return_attn:
                h, attn = layer(h, dist_matrix, return_attn=True)
                attn_weights = attn if attn_weights is None else attn_weights + attn
            else:
                h = layer(h, dist_matrix)
        h = h.squeeze(0)
        if return_attn: return self.output_proj(h), attn_weights / len(self.layers)
        return self.output_proj(h)

def compute_idw_temp(target_idx, obs_idx, obs_temps):
    if len(obs_idx) == 0: return 0.0
    d = DIST_MATRIX[target_idx, obs_idx]
    w = 1.0 / (d ** 2 + 1e-12)
    mask = ~np.isnan(obs_temps)
    if not mask.any(): return 0.0
    return float((w[mask] * obs_temps[mask]).sum() / w[mask].sum())

def build_features(yr_test, train_st):
    train_idx = np.array([station_idx[st] for st in train_st], dtype=np.int64)
    train_st_set = set(train_st)
    df_other = df[df["year"] != yr_test]
    train_df = df_other[df_other["station"].isin(train_st)]
    train_clim = np.full(n_stations, np.nan, dtype=np.float32)
    for st, v in train_df.groupby("station")["temp"].mean().items():
        train_clim[station_idx[st]] = v
    clim_mean, clim_std = np.nanmean(train_clim[train_idx]), np.nanstd(train_clim[train_idx]) + 1e-8
    global_mean = np.nanmean(train_df["temp"])
    clim_arr = train_clim.copy()
    for i in np.setdiff1d(np.arange(n_stations), train_idx):
        clim_arr[i] = compute_idw_temp(i, train_idx, train_clim[train_idx])
    clim_arr = np.where(np.isnan(clim_arr), global_mean, clim_arr)

    df_test_year = df[df["year"] == yr_test]
    train_temps = np.full(n_stations, np.nan, dtype=np.float32)
    for st in train_st:
        row = df_test_year[df_test_year["station"] == st]
        if not row.empty: train_temps[station_idx[st]] = float(row["temp"].values[0])

    feats = []
    for st in common_stations:
        i = station_idx[st]
        if st in train_st_set:
            t_val = train_temps[i] if not np.isnan(train_temps[i]) else train_clim[i]
            has_obs = 1.0
        else:
            t_val = compute_idw_temp(i, train_idx, train_temps[train_idx])
            has_obs = 0.0
        feats.append(
            [(t_val - global_mean) / 20.0, lon_norm[i], lat_norm[i], elev_norm[i], (clim_arr[i] - clim_mean) / clim_std,
             has_obs, 0.0])
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(COORDS_arr, dtype=torch.float32)

if __name__ == "__main__":
    print(f"\n{'=' * 60}\nExtracting and Analyzing Attention\n{'=' * 60}")

    attn_matrices = []
    for seed in SEEDS:
        pth_path = os.path.join(CKPT_DIR, f"egt_si_density{int(DENSITY * 100)}_year{FIXED_YEAR}_seed{seed}.pth")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Checkpoint missing: {pth_path}.")
        loaded = torch.load(pth_path, map_location="cpu")
        x, c = build_features(FIXED_YEAR, loaded["train_st"])
        encoder = EGTEncoder(7, EGT_HIDDEN, EGT_OUT, EGT_LAYERS, EGT_HEADS, EGT_DROPOUT, year_to_idx).to(DEVICE)
        encoder.load_state_dict(loaded["encoder_state_dict"])
        encoder.eval()
        x = x.to(DEVICE)
        c = c.to(DEVICE)
        with torch.no_grad():
            _, attn = encoder(x, FIXED_YEAR, c, DIST_TRAIN_T, return_attn=True)
            attn_matrices.append(attn.squeeze(0).cpu().numpy())
        print(f"  Extracted attention for seed {seed}")

    attn_stack = np.stack(attn_matrices, axis=0)
    attn_mean = np.mean(attn_stack, axis=0)
    attn_std = np.std(attn_stack, axis=0)

    print("\nGenerating Fig 1: Spatial Attention Distribution Map...")
    in_strength_per_seed = [m.sum(axis=0) - np.diag(m) for m in attn_matrices]
    in_strength_per_seed = np.stack(in_strength_per_seed, axis=0)
    in_strength_mean = np.mean(in_strength_per_seed, axis=0)
    in_strength_std = np.std(in_strength_per_seed, axis=0)
    rank_quantiles = pd.Series(in_strength_mean).rank(pct=True).values

    df_site_attn = pd.DataFrame({
        'Station_ID': common_stations, 'Longitude': lons_raw, 'Latitude': lats_raw,
        'Elevation_m': elevs_raw, 'Incoming_Attention_Mean': in_strength_mean,
        'Incoming_Attention_Std': in_strength_std, 'Rank_Quantile': rank_quantiles
    }).sort_values(by='Incoming_Attention_Mean', ascending=False)
    df_site_attn.to_csv(os.path.join(OUT_DIR, "station_incoming_attention.csv"), index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([73, 136, 15, 54], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f0e1')
    ax.add_feature(cfeature.OCEAN, facecolor='#dce8f2')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    scatter = ax.scatter(lons_raw, lats_raw, c=rank_quantiles, cmap='Reds', s=40, edgecolor='black', linewidth=0.3,
                         transform=ccrs.PlateCarree(), zorder=3)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.08, fraction=0.046, shrink=0.8)
    cbar.set_label('Rank Quantile of Incoming Attention', fontsize=12)
    ax.set_title(
        f"Spatial Distribution of Received Attention\n(Year {FIXED_YEAR}, {EGT_LAYERS} Layers, {EGT_HEADS} Heads, Mean of {len(SEEDS)} Seeds)",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_Spatial_Attention_Distribution.png"), bbox_inches='tight')
    plt.close()

    print("Generating Fig 2: Distance-Attention Distribution Plot...")
    mask_tri = np.triu(np.ones_like(attn_mean, dtype=bool), k=1)
    df_plot = pd.DataFrame(
        {'distance': DIST_MATRIX[mask_tri], 'attn_mean': attn_mean[mask_tri], 'attn_std': attn_std[mask_tri]})
    df_plot['bin'] = pd.cut(df_plot['distance'], bins=DIST_BINS, labels=DIST_LABELS, right=False)
    stats = df_plot.groupby('bin', observed=False).agg(n_pairs=('attn_mean', 'size'), mean_attn=('attn_mean', 'mean'),
                                                       mean_std=('attn_std', 'mean')).reset_index()

    for i in range(len(DIST_LABELS)):
        if DIST_LABELS[i] not in stats['bin'].values:
            stats = pd.concat(
                [stats, pd.DataFrame([{'bin': DIST_LABELS[i], 'n_pairs': 0, 'mean_attn': np.nan, 'mean_std': np.nan}])],
                ignore_index=True)
    stats = stats.sort_values('bin', key=lambda x: x.map({v: i for i, v in enumerate(DIST_LABELS)})).reset_index(
        drop=True)
    stats.to_csv(os.path.join(OUT_DIR, "attention_distance_stats.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    valid_mask = stats['mean_attn'].notna().values
    mean_vals = stats['mean_attn'].values[valid_mask]
    std_vals = stats['mean_std'].values[valid_mask]
    valid_x = [i for i in range(len(DIST_LABELS)) if valid_mask[i]]

    ax.fill_between(valid_x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='#1f77b4')
    ax.plot(valid_x, mean_vals, '-o', color='#1f77b4', linewidth=2, markersize=8, label='Mean Attention (±1 SD)')
    uniform_attn = 1.0 / n_stations
    ax.axhline(uniform_attn, color='red', linestyle='--', linewidth=2, label=f'Uniform Baseline ({uniform_attn:.4f})')

    for i, row in stats.iterrows():
        if pd.notna(row['mean_attn']):
            ax.text(i, uniform_attn * 0.5, f"n={int(row['n_pairs'])}", ha='center', va='top', fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xticks(range(len(DIST_LABELS)))
    ax.set_xticklabels(DIST_LABELS)
    ax.set_yscale('log')
    ax.set_ylim(uniform_attn * 0.3, np.nanmax(mean_vals + std_vals) * 1.5)
    ax.set_ylabel('Attention Weight (Log Scale)', fontsize=12)
    ax.set_xlabel('Distance Bin (km)', fontsize=12)
    ax.set_title(f'Attention Weight Decay with Distance\n(Year {FIXED_YEAR}, Mean of {len(SEEDS)} Seeds)', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_Distance_Attention.png"), bbox_inches='tight')
    plt.close()

    print("Generating Fig 3: Cross-Seed Stability (Heatmap)...")
    in_strength_all = np.stack([m.sum(0) - np.diag(m) for m in attn_matrices], axis=1)
    n_seeds = len(SEEDS)
    spearman_matrix = np.zeros((n_seeds, n_seeds))
    for i in range(n_seeds):
        for j in range(n_seeds):
            rho, _ = spearmanr(in_strength_all[:, i], in_strength_all[:, j])
            spearman_matrix[i, j] = rho
    np.fill_diagonal(spearman_matrix, 1.0)
    pd.DataFrame(spearman_matrix, index=[f"Seed{s}" for s in SEEDS], columns=[f"Seed{s}" for s in SEEDS]).to_csv(
        os.path.join(OUT_DIR, "cross_seed_spearman_matrix.csv"))

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    im = ax.imshow(spearman_matrix, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_seeds))
    ax.set_yticks(range(n_seeds))
    ax.set_xticklabels([f"Seed {str(s)}" for s in SEEDS], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f"Seed {str(s)}" for s in SEEDS], fontsize=9)
    for i in range(n_seeds):
        for j in range(n_seeds):
            color = 'white' if spearman_matrix[i, j] < 0.85 else 'black'
            ax.text(j, i, f'{spearman_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label='Spearman Correlation Coefficient')
    ax.set_title(f'Cross-Seed Attention Stability\n(Spearman Correlation of In-Strength)', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_CrossSeed_Stability.png"), bbox_inches='tight')
    plt.close()

    print("Generating Fig 4: Attention Flow Map...")
    in_degree = attn_mean.sum(axis=0)
    top_30_idx = np.argsort(in_degree)[-30:][::-1]
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([73, 136, 15, 54], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f0e1')
    ax.add_feature(cfeature.OCEAN, facecolor='#dce8f2')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    ax.scatter(lons_raw, lats_raw, c='lightgray', s=10, transform=ccrs.PlateCarree(), alpha=0.6, zorder=1)
    for target_idx in top_30_idx:
        top_3_src = np.argsort(attn_mean[:, target_idx])[-3:]
        for src_idx in top_3_src:
            if attn_mean[src_idx, target_idx] < 0.005: continue
            ax.annotate("", xy=(lons_raw[target_idx], lats_raw[target_idx]),
                        xytext=(lons_raw[src_idx], lats_raw[src_idx]),
                        arrowprops=dict(arrowstyle="->", color='blue', lw=1.5, alpha=0.6,
                                        connectionstyle="arc3,rad=0.1"), transform=ccrs.PlateCarree())
        ax.scatter(lons_raw[target_idx], lats_raw[target_idx], c='red', s=40, edgecolor='k',
                   transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(f"Attention Flow Map (Top 30 Attended Stations, Year {FIXED_YEAR})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_Attention_Map.png"), bbox_inches='tight')
    plt.close()

    print("Generating Fig 5: Partition-Distance Attention Chart...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf_division = gpd.read_file('Geodivision.shp')
        gdf_points = gpd.GeoDataFrame({'Station_ID': common_stations},
                                      geometry=[Point(lon, lat) for lon, lat in zip(lons_raw, lats_raw)],
                                      crs="EPSG:4326")
        if gdf_division.crs != gdf_points.crs: gdf_division = gdf_division.to_crs(gdf_points.crs)
        merged = gpd.sjoin(gdf_points, gdf_division, how='inner', predicate='within')

        results = {}
        for region in ['Northern', 'Northwestern', 'Qinghai_Tibet', 'Southern']:
            region_stations = merged[merged['name'] == region]['Station_ID'].tolist()
            if not region_stations: continue
            region_indices = [station_idx[st] for st in region_stations if st in station_idx]
            region_attn = attn_mean[region_indices, :]
            region_dist = DIST_MATRIX[region_indices, :]
            bin_means = []
            for i in range(len(DIST_BINS) - 1):
                mask = (region_dist >= DIST_BINS[i]) & (region_dist < DIST_BINS[i + 1])
                bin_means.append(region_attn[mask].mean() if mask.any() else np.nan)
            results[region] = bin_means

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, (region, means) in enumerate(results.items()):
            valid_x = [i for i in range(len(DIST_LABELS)) if not np.isnan(means[i])]
            valid_y = [means[i] for i in valid_x]
            ax.plot(valid_x, valid_y, '-o', color=colors[idx % len(colors)], linewidth=2, markersize=6,
                    label=region.replace('_', '-'))
        uniform_attn = 1.0 / n_stations
        ax.axhline(uniform_attn, color='red', linestyle='--', linewidth=2,
                   label=f'Uniform Baseline ({uniform_attn:.4f})')
        ax.set_xticks(range(len(DIST_LABELS)))
        ax.set_xticklabels(DIST_LABELS)
        ax.set_yscale('log')
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        ax.set_ylabel('Mean Attention Weight', fontsize=12)
        ax.set_xlabel('Distance Bin (km)', fontsize=12)
        ax.set_title(f'Attention Weight by Region and Distance\n(Year {FIXED_YEAR}, Mean of {len(SEEDS)} Seeds)',
                     fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "Fig_Partition_Distance.png"), bbox_inches='tight')
        plt.close()

        latex_caption = f"Mean attention weights across partitions and distance bins, averaged over {len(SEEDS)} seeds. The uniform baseline is \(1/{n_stations} \\approx {1.0 / n_stations:.4f}\)."
        latex_code = r"""\begin{table}[htbp]
    \centering
""" + f"    \\caption{{{latex_caption}}}\n" + r"""    \label{tab:attn_partition_distance}
    \begin{tabular}{lccccccc}
    \toprule
    Target Partition & 0--200 & 200--500 & 500--1000 & 1000--1500 & 1500--2000 & 2000--3000 & 3000+ \\
    \midrule
"""
        for region, means in results.items():
            vals = " & ".join([f"{v:.5f}" if not np.isnan(v) else "--" for v in means])
            latex_code += f"    {region.replace('_', '-')} & {vals} \\\\\n"
        latex_code += r"""    \bottomrule
    \end{tabular}
\end{table}"""
        print("\n" + "=" * 60 + "\nLaTeX Code for Table 5:\n" + "=" * 60)
        print(latex_code)

    except Exception as e:
        print(f"\n[Warning] Partition analysis skipped: {e}")

    print(f"\nAll outputs saved to: {OUT_DIR}/")
