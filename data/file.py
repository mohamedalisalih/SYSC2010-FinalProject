import pandas as pd

files = [
    "data/en_climate_hourly_ON_6080192_05-2010_P1H.csv",
    "data/en_climate_hourly_ON_6080192_06-2010_P1H copy.csv",
]

dfs = []

for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    df = df[["Date/Time (LST)", "Temp (°C)"]].copy()
    df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"], errors="coerce")
    df["Temp (°C)"] = pd.to_numeric(df["Temp (°C)"], errors="coerce")

    df = df.dropna(subset=["Date/Time (LST)", "Temp (°C)"]).reset_index(drop=True)
    dfs.append(df)

temp_df = pd.concat(dfs, ignore_index=True)
temp_df = temp_df.sort_values("Date/Time (LST)").drop_duplicates().reset_index(drop=True)

t0 = temp_df["Date/Time (LST)"].iloc[0]
temp_df["Time [s]"] = (temp_df["Date/Time (LST)"] - t0).dt.total_seconds()

out_df = temp_df[["Time [s]", "Temp (°C)"]].copy()
out_df.columns = ["Time [s]", "Temperature"]

out_df.to_csv("data/temp_big_clean.csv", index=False)

print(out_df.head())
print(out_df.tail())
print(f"Saved: data/temp_big_clean.csv with {len(out_df)} samples")