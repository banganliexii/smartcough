import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Untuk handle data imbalance
try:
    from imblearn.over_sampling import SMOTE
    imblearn_installed = True
except ImportError:
    imblearn_installed = False

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    return df

# Load dataset
df = pd.read_excel(r'C:\SMARTCOUGH\csv\metadata_compiled.csv.xlsx', engine='openpyxl')
df = downcast_dtypes(df)

print("Info awal dataset:")
print(df.info())
print("\nSample data:")
print(df.sample(5))

# CLEANING DASAR 

# 1. Drop kolom tidak perlu (edit list kolom_tidak_perlu jika ada kolom yang ingin di-drop)
kolom_tidak_perlu = []
if kolom_tidak_perlu:
    df.drop(columns=[c for c in kolom_tidak_perlu if c in df.columns], inplace=True)

# 2. Konversi tipe data kolom penting sebelum validasi
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
if 'day' in df.columns:
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
if 'month' in df.columns:
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

# 3. Handle missing value
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 4. Hapus duplikat
df.drop_duplicates(inplace=True)

# 5. Bersihkan string pada kolom object
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# 6. Standardisasi kolom 'gender' sebelum encoding
if 'gender' in df.columns:
    df['gender'] = df['gender'].astype(str).str.lower().replace({
        'true': 'male',
        'false': 'female',
        'laki-laki': 'male',
        'perempuan': 'female',
        'pria': 'male',
        'wanita': 'female',
        '1': 'male',
        '0': 'female'
    })
    # Jika masih ada selain male/female/other, jadikan 'other'
    df['gender'] = df['gender'].where(df['gender'].isin(['male','female','other']), 'other')

print("\nDistribusi gender setelah standarisasi:")
print(df['gender'].value_counts())

# 7. Konversi datetime dan ekstrak fitur waktu
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

# 8. Validasi range kolom
if 'age' in df.columns:
    print("Jumlah data sebelum validasi umur:", df.shape)
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    print("Jumlah data setelah validasi umur:", df.shape)
if 'day' in df.columns:
    df = df[(df['day'] >= 1) & (df['day'] <= 31)]
if 'month' in df.columns:
    df = df[(df['month'] >= 1) & (df['month'] <= 12)]

# 9. Advanced Outlier Handling (log transform jika distribusi sangat skewed)
num_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
for col in num_cols:
    # log transform hanya untuk kolom numerik yang bukan 'day', 'month', 'year', 'age'
    if col not in ['day', 'month', 'year', 'age'] and np.abs(df[col].skew()) > 2:  
        print(f"Log transform pada kolom: {col}")
        df[col] = np.log1p(df[col] - df[col].min() + 1)

# 10. Outlier removal dengan IQR
def remove_outlier_iqr(df, columns):
    for col in columns:
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64'] and col not in ['day', 'month', 'year', 'age']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outlier_iqr(df, num_cols)

# 11. Simpan kolom age asli sebelum scaling
if 'age' in df.columns:
    df['age_asli'] = df['age']

# 12. Normalisasi / Scaling (hanya numerik selain 'day','month','year','age_asli','age')
scale_cols = [c for c in num_cols if c not in ['day', 'month', 'year', 'age', 'age_asli']]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("Normalisasi data numerik selesai.")

# EDA SINGKAT
print("\nStatistik deskriptif:")
print(df.describe(include='all'))

if scale_cols:
    df[scale_cols].hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    plt.show()

# Periksa distribusi kategorik (edit sesuai kolom kategorik)
categorical_cols = ['gender', 'status']
for col in categorical_cols:
    if col in df.columns:
        print(f"\nDistribusi {col}:")
        print(df[col].value_counts())
        sns.countplot(x=col, data=df)
        plt.show()

if scale_cols:
    plt.figure(figsize=(16,12))
    sns.heatmap(df[scale_cols].corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# ENCODING KATEGORIK 
cat_cols = [col for col in categorical_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nSample kolom hasil encoding:")
print([col for col in df_encoded.columns if col.startswith("gender")])

# HANDLE IMBALANCE LABEL (opsional) 
if 'status' in df.columns:
    df_encoded['status_encoded'] = df['status'].astype('category').cat.codes
    from collections import Counter
    print("Distribusi label (status_encoded):", Counter(df_encoded['status_encoded']))
    # Drop kolom yang tidak dipakai dan kolom object
    X = df_encoded.drop(['status', 'status_encoded', 'uuid', 'datetime', 'age_asli'], axis=1, errors='ignore')
    y = df_encoded['status_encoded']

    # PASTIKAN HANYA NUMERIK di X
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols):
        print("Kolom non-numerik terdeteksi pada X:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    # SMOTE (oversampling) kalau imbalance
    if imblearn_installed:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("Setelah SMOTE oversampling:", Counter(y))
    else:
        print("imblearn belum terinstall, skip SMOTE oversampling.")

    # MODELING 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print("Akurasi model setelah advanced preprocessing:", clf.score(X_test, y_test))

# Data baru yang sudah bersih
# Simpan kolom age_asli (bukan age yang sudah diskalakan)
kolom_export = list(df_encoded.columns)
if 'age' in kolom_export and 'age_asli' in kolom_export:
    kolom_export.remove('age')  # Jangan ekspor age scaled
df_encoded[kolom_export].to_csv('data_advanced_preprocessed.csv', index=False)
print("Data hasil preprocessing lanjutan disimpan sebagai 'data_advanced_preprocessed.csv'")