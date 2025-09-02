# ben
mendeteksi dan membuang outlier

import pandas as pd
import numpy as np

# ——————————————————————
# → 1. Baca dataset Anda
# Jika dataset sudah di-upload ke Colab, misalnya iris.csv, df akan dibaca:
df = pd.read_csv("iris.csv")

# ——————————————————————
# → 2. Inspeksi awal
print("Info dataset:")
print(df.info())
print("\nContoh data awal:")
print(df.head())

print("\nDescriptive statistics:")
print(df.describe())

print("\nJumlah nilai kosong tiap kolom:")
print(df.isnull().sum())

# ——————————————————————
# → 3. Identifikasi kolom numerik
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nKolom numerik:", num_cols)

# ——————————————————————
# → 4. Fungsi deteksi dan penghapusan outlier via IQR
def detect_outliers_iqr(data, cols):
    outlier_indices = {}
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask_out = (data[col] < lower) | (data[col] > upper)
        outlier_indices[col] = data[mask_out].index.tolist()
        print(f"{col}: {len(outlier_indices[col])} outlier")
    return outlier_indices

def remove_outliers_iqr(data, cols):
    df_clean = data.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

# ——————————————————————
# → 5. Deteksi outlier
print("\nMendeteksi outlier:")
outliers = detect_outliers_iqr(df, num_cols)

# Cetak beberapa indeks dan nilai yang aneh
for col, idx_list in outliers.items():
    if len(idx_list)>0:
        sample_idx = idx_list[:3]
        print(f"\nContoh outlier di kolom {col}:")
        print(df.loc[sample_idx, col])

# ——————————————————————
# → 6. Hapus outlier
df_clean = remove_outliers_iqr(df, num_cols)
print(f"\nUkuran sebelum = {df.shape}, setelah buang outlier = {df_clean.shape}")

# ——————————————————————
# → 7. Opsional: lihat distribusi lagi setelah bersih
print("\nDeskriptif setelah bersih:")
print(df_clean.describe())
