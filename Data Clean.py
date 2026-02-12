# %%
import pandas as pd
import os

# %%
base_dir = os.path.join("Dataset", "verify_BC_datasets")
years = range(2021, 2026)  # 2021 到 2024

dataframes = []
for year in years:
    file_path = os.path.join(base_dir, str(year), f"PM25_with_geo_{year}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"✅ {year} 年的檔案已成功讀取: {file_path}")
    else:
        print(f"⚠️ 找不到檔案: {file_path}")


# %%
for df in dataframes:
    print(df.dtypes)

# %%
# check if all merged datasets have the same columns
if dataframes:
    first_columns = set(dataframes[0].columns)
    for i, df in enumerate(dataframes):
        if set(df.columns) != first_columns:
            print(f"⚠️ Merged dataset {i} has different columns: {set(df.columns)}")
        else:
            print(f"✅ Merged dataset {i} has consistent columns.")

# %%
for df in dataframes:
    print(df.dtypes)

# %%
# Briefly check the first few rows of the merged datasets
dataframes[0].head()

# %%
dataframes[1].head()

# %%
dataframes[2].head()

# %%
dataframes[3].head()

# %%
dataframes[4].head()

# %%
# Convert 2025 Datetime Object to datatime format
dataframes[4]["Datetime"] = pd.to_datetime(dataframes[4]["Datetime"], errors="coerce")

# %%
dataframes[0].isna().sum()/dataframes[0].shape[0]

# %%
dataframes[1].isna().sum()/dataframes[1].shape[0]

# %%
dataframes[2].isna().sum()/dataframes[2].shape[0]

# %%
dataframes[3].isna().sum()/dataframes[3].shape[0]

# %%
dataframes[4].isna().sum()/dataframes[4].shape[0]

# %%
# print each year length
for i, df in enumerate(dataframes):
    print(f"Year {2021 + i}: {len(df)} records")

# %%
for i, df in enumerate(dataframes):
    # Check unique stations in each year
    unique_stations = df["Station"].nunique()
    print(f"Year {2021 + i}: {unique_stations} unique stations")

# %%
# Find the common stations across all years
common_stations = set(dataframes[0]["Station"].unique())
for df in dataframes[1:]:
    common_stations &= set(df["Station"].unique())
print(f"Common stations across all years: {len(common_stations)}")

# print not the common stations in each year
for i, df in enumerate(dataframes):
    unique_stations = set(df["Station"].unique())
    not_common = unique_stations - common_stations
    print(f"Year {2021 + i}: {len(not_common)} stations not in common: {sorted(not_common)}")

# %%
# only contain the common stations in each year
for i, df in enumerate(dataframes):
    unique_stations = set(df["Station"].unique())
    
    not_common = unique_stations - common_stations
    print(f"Year {2021 + i}: {len(not_common)} stations not in common: {sorted(not_common)}")
    
    dataframes[i] = df[df["Station"].isin(common_stations)]
    
    

# %%
# check the missing percentage and number of records after filtering to common stations
for i, df in enumerate(dataframes):
    missing_percentage = df.isna().sum()/df.shape[0]
    print(f"Year {2021 + i} missing percentage:\n{missing_percentage}\n")
    print(f"Year {2021 + i} number of records: {df.shape[0]}\n")

# %%
# Clean the dataset make sure that each dataframe have year less than or equal to 2025

for i, df in enumerate(dataframes):
    # Filter out records with year greater than 2025
    dataframes[i] = df[df["Datetime"].str.contains("202[1-5]")]
    print(f"Year {2021 + i} missing percentage:\n{missing_percentage}\n")
    print(f"Year {2021 + i} number of records: {df.shape[0]}\n")


