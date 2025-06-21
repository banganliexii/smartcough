import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cek apakah imblearn (SMOTE) sudah terpasang agar bisa menangani data imbalance
try:
    from imblearn.over_sampling import SMOTE
    imblearn_installed = True
except ImportError:
    imblearn_installed = False

def downcast_dtypes(df):
    # Mengoptimalkan tipe data float64 dan int64 menjadi tipe yang lebih ringan agar menghemat memori
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    return df

# Memuat dataset utama dari file Excel
df = pd.read_excel(r'C:\SMARTCOUGH\csv\metadata_compiled.csv.xlsx', engine='openpyxl')
df = downcast_dtypes(df)

print("Menampilkan informasi awal dataset:")
print(df.info())
print("\nMenampilkan contoh data secara acak:")
print(df.sample(5))

# ======== MELAKUKAN PEMROSESAN DATA DASAR ========

# 1. Kalau ada kolom tidak diperlukan, silakan masukkan namanya ke list kolom_tidak_perlu lalu hapus dari dataframe
kolom_tidak_perlu = []
if kolom_tidak_perlu:
    df.drop(columns=[c for c in kolom_tidak_perlu if c in df.columns], inplace=True)

# 2. Pastikan tipe data kolom penting sudah benar sebelum melakukan validasi nilai
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
if 'day' in df.columns:
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
if 'month' in df.columns:
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

# 3. Menangani missing value: untuk numerik gunakan median, untuk kategorik gunakan modus
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 4. Menghapus data duplikat agar tidak menyebabkan bias pada analisis atau model
df.drop_duplicates(inplace=True)

# 5. Membersihkan spasi di kolom string/object agar data menjadi konsisten
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# 6. Menstandarkan kolom gender supaya nilainya konsisten dan mudah dianalisis
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
    # Mengubah nilai selain male/female/other menjadi 'other'
    df['gender'] = df['gender'].where(df['gender'].isin(['male','female','other']), 'other')

print("\nMenampilkan distribusi gender setelah standarisasi:")
print(df['gender'].value_counts())

# 7. Kalau ada kolom datetime, konversi ke format datetime lalu ekstrak tahun, bulan, dan hari
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

# 8. Melakukan validasi nilai pada kolom umur, hari, dan bulan supaya hanya memakai data wajar
if 'age' in df.columns:
    print("Menampilkan jumlah data sebelum validasi umur:", df.shape)
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    print("Menampilkan jumlah data setelah validasi umur:", df.shape)
if 'day' in df.columns:
    df = df[(df['day'] >= 1) & (df['day'] <= 31)]
if 'month' in df.columns:
    df = df[(df['month'] >= 1) & (df['month'] <= 12)]

# 9. Melakukan log transform jika kolom numerik sangat skewed supaya distribusi data lebih normal
num_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
for col in num_cols:
    # Hanya melakukan log transform untuk kolom numerik yang bukan tanggal/waktu/umur
    if col not in ['day', 'month', 'year', 'age'] and np.abs(df[col].skew()) > 2:  
        print(f"Melakukan log transform pada kolom: {col}")
        df[col] = np.log1p(df[col] - df[col].min() + 1)

# 10. Menghapus outlier dengan metode IQR agar data lebih bersih dan robust
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

# 11. Menyimpan kolom umur asli sebelum melakukan scaling agar mudah dianalisis di tahap selanjutnya
if 'age' in df.columns:
    df['age_asli'] = df['age']

# 12. Melakukan normalisasi/standarisasi untuk kolom numerik selain tanggal/waktu/umur asli
scale_cols = [c for c in num_cols if c not in ['day', 'month', 'year', 'age', 'age_asli']]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("Berhasil melakukan normalisasi data numerik.")

# ======== MELAKUKAN EDA SINGKAT UNTUK MEMAHAMI DATA ========
print("\nMenampilkan statistik deskriptif:")
print(df.describe(include='all'))

if scale_cols:
    df[scale_cols].hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    plt.show()

# Menampilkan distribusi data kategorik (silakan edit kolom sesuai dataset)
categorical_cols = ['gender', 'status']
for col in categorical_cols:
    if col in df.columns:
        print(f"\nMenampilkan distribusi {col}:")
        print(df[col].value_counts())
        sns.countplot(x=col, data=df)
        plt.show()

# Memvisualisasikan korelasi antar fitur numerik
if scale_cols:
    plt.figure(figsize=(16,12))
    sns.heatmap(df[scale_cols].corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# ======== MELAKUKAN ENCODING FITUR KATEGORIK ========
cat_cols = [col for col in categorical_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nMenampilkan contoh kolom hasil encoding:")
print([col for col in df_encoded.columns if col.startswith("gender")])

# ======== MENANGANI IMBALANCE LABEL (OPSIONAL) ========
if 'status' in df.columns:
    df_encoded['status_encoded'] = df['status'].astype('category').cat.codes
    from collections import Counter
    print("Menampilkan distribusi label (status_encoded):", Counter(df_encoded['status_encoded']))
    # Menghapus kolom yang tidak digunakan dan bertipe object agar siap untuk modeling
    X = df_encoded.drop(['status', 'status_encoded', 'uuid', 'datetime', 'age_asli'], axis=1, errors='ignore')
    y = df_encoded['status_encoded']

    # Memastikan semua kolom pada X adalah numerik sebelum modeling
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols):
        print("Menemukan kolom non-numerik pada X:", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    # Jika imblearn terpasang, lakukan oversampling menggunakan SMOTE jika label tidak seimbang
    if imblearn_installed:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("Menampilkan distribusi label setelah SMOTE oversampling:", Counter(y))
    else:
        print("imblearn belum terinstall, sehingga melewati proses SMOTE oversampling.")

    # ======== MELAKUKAN MODELING SINGKAT DENGAN RANDOM FOREST ========
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print("Menampilkan akurasi model setelah preprocessing lanjutan:", clf.score(X_test, y_test))

# Data hasil proses sudah bersih dan siap digunakan
# Simpan hasil akhir, gunakan kolom age_asli (bukan age scaled) agar interpretasi lebih mudah
kolom_export = list(df_encoded.columns)
if 'age' in kolom_export and 'age_asli' in kolom_export:
    kolom_export.remove('age')  # Hindari mengekspor age yang sudah diskalakan
df_encoded[kolom_export].to_csv('data_advanced_preprocessed.csv', index=False)
print("Berhasil menyimpan data hasil preprocessing lanjutan sebagai 'data_advanced_preprocessed.csv'")