import pandas as pd
import geopandas as gpd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

df_all = pd.read_csv("station_model_mae_raw_with_seeds.csv")

raw_df = pd.read_csv("TAVE.csv")
raw_df.rename(columns={"station_id": "station", "longitude": "lon", "latitude": "lat"}, inplace=True)
station_info = raw_df.groupby("station")[["lon", "lat"]].first().reset_index()

df_all = df_all.merge(station_info, on="station", how="left")

if df_all['lon'].isna().any():
    print(f"Warning: {df_all['lon'].isna().sum()} records have missing coordinates. Please check if station IDs match.")
    df_all = df_all.dropna(subset=['lon', 'lat'])

gdf_division = gpd.read_file('Geodivision.shp')
gdf_points = gpd.GeoDataFrame(df_all, geometry=gpd.points_from_xy(df_all.lon, df_all.lat), crs="EPSG:4326")

if gdf_division.crs != gdf_points.crs:
    gdf_division = gdf_division.to_crs(gdf_points.crs)

merged = gpd.sjoin(gdf_points, gdf_division, how='inner', predicate='within')

region_col = 'name'
model_col = 'model'

seed_region_mae = merged.groupby([model_col, region_col, 'seed'])['abs_error'].mean().reset_index()
final_stats = seed_region_mae.groupby([model_col, region_col])['abs_error'].agg(['mean', 'std']).reset_index()

final_stats['MAE_formatted'] = final_stats.apply(lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1)
pivot_table = final_stats.pivot(index=region_col, columns=model_col, values='MAE_formatted')

rename_dict = {
    'EGT-SI': 'EGT-SI',
    'IDW': 'IDW',
    'RF': 'Random Forest',
    'XGBoost': 'XGBoost',
    'OK': 'Ordinary Kriging',
    'RK': 'RK',
    'ST-GCN': 'ST-GCN'
}
pivot_table.rename(columns=rename_dict, inplace=True)

col_order = ['EGT-SI', 'IDW', 'Random Forest', 'XGBoost', 'Ordinary Kriging', 'RK', 'ST-GCN']
pivot_table = pivot_table[col_order]

print("\n=== Strict 10-seed Run-to-Run Standard Deviation ===")
print(pivot_table)

pivot_table.to_csv('Table4_Strict_10_Seed_Std.csv', encoding='utf-8-sig')

print("\n=== LaTeX Code for Paper ===")
latex_str = "\\begin{table}[htbp]\n\\centering\n"
latex_str += r"\caption{Mean absolute error (MAE) averaged over test stations in each geographic region (units: \(^\circ\)C). Values are reported as mean \(\pm\) standard deviation across 10 random seeds.}" + "\n"
latex_str += "\\label{tab:regional_mae_mean}\n"
latex_str += "\\resizebox{\\textwidth}{!}{\n"
latex_str += "\\begin{tabular}{l" + "c" * len(pivot_table.columns) + "}\n"
latex_str += "\\toprule\n"
latex_str += "Region & " + " & ".join(pivot_table.columns) + " \\\\\n"
latex_str += "\\midrule\n"

for idx, row in pivot_table.iterrows():
    row_str = " & ".join([str(val).replace('±', r'\(\pm\)') for val in row.values])
    region_name = str(idx).replace('_', '-')
    latex_str += f"{region_name} & {row_str} \\\\\n"

latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}"
print(latex_str)
