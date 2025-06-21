import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# === KONFIGURASI ===
DATA_DIR = "audio/dataset_audio"
CSV_PATH = "C:\\SMARTCOUGH\\csv\\data_advanced_preprocessed.csv"
N_MFCC = 40
MAX_LEN = 130
SAMPLE_RATE = 16000
SEED = 42
AUGMENT = True
MIN_SAMPLES_PER_CLASS = 50  # Naikkan minimum sample per kelas untuk kualitas lebih baik
BATCH_SIZE = 16  # Gunakan batch size lebih besar
VALIDATION_SPLIT = 0.15  # Sisihkan data validasi terpisah

np.random.seed(SEED)
tf.random.set_seed(SEED)

if not os.path.exists(DATA_DIR):
    print(f"❌ Folder '{DATA_DIR}' tidak ditemukan.")
    exit(1)

# --- MUAT DATA ---
df = pd.read_csv(CSV_PATH)
df['uuid'] = df['uuid'].astype(str).str.strip()

# --- EKSTRAK HANYA 5 TIPE BATUK TARGET ---
rows_ct = []
for i in range(1, 5):
    severity_col = f'severity_{i}' if f'severity_{i}' in df.columns else None
    cols = ['uuid', f'cough_type_{i}', f'quality_{i}']
    if severity_col:
        cols.append(severity_col)
    rows_ct.append(
        df[cols].rename(
            columns={
                f'cough_type_{i}': 'label_raw',
                f'quality_{i}': 'quality',
                f'severity_{i}': 'severity'
            }
        ).assign(source='cough')
    )
long_df = pd.concat(rows_ct, axis=0, ignore_index=True).reset_index(drop=True)

# --- MAPPING LABEL ---
label_map = {
    'dry': 'dry_cough',
    'wet': 'wet_cough',
    'allergy': 'allergy_cough',
    'COVID-19': 'covid_cough',
    'sneeze': 'sneeze_flu'
}
long_df['label'] = long_df['label_raw'].map(label_map)

print("Cek distribusi label_raw sebelum mapping:")
print(long_df['label_raw'].value_counts(dropna=False))
print("Cek distribusi label setelah mapping:")
print(long_df['label'].value_counts(dropna=False))

# --- FILTER: hanya ambil 5 kelas target & kualitas audio ok/good ---
long_df = long_df[long_df['label'].notnull()].reset_index(drop=True)
long_df = long_df[long_df['quality'].isin(['good', 'ok'])].reset_index(drop=True)
print("Total data batuk yang digunakan:", len(long_df))

# --- HAPUS KELAS MINORITAS EKSTRIM ---
class_counts = long_df['label'].value_counts()
print("Distribusi kelas sebelum filtering:")
print(class_counts)

keep_labels = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
long_df = long_df[long_df['label'].isin(keep_labels)].reset_index(drop=True)
print(f"Kelas yang dipakai (>= {MIN_SAMPLES_PER_CLASS} sample):", keep_labels.tolist())

# Pastikan jumlah kelas cukup
if len(keep_labels) < 2:
    print("❌ Tidak cukup kelas untuk klasifikasi. Minimal butuh 2 kelas.")
    exit(1)

def to_severity_num(x):
    if pd.isnull(x): return 0
    try: return int(x)
    except: return 0
long_df['severity'] = long_df['severity'].apply(to_severity_num)

# --- TEMUKAN FILE AUDIO ---
def get_audio_path(uuid):
    allowed_ext = ('.webm', '.ogg', '.wav', '.mp3', '.m4a')
    for fname in os.listdir(DATA_DIR):
        if uuid in fname and fname.lower().endswith(allowed_ext):
            return os.path.join(DATA_DIR, fname)
    return None

long_df['file_path'] = long_df['uuid'].apply(get_audio_path)
long_df = long_df[long_df['file_path'].notnull()].reset_index(drop=True)
print(f"Total data batuk yang lolos filter: {len(long_df)}")
if len(long_df) == 0:
    print("❌ Tidak ada data audio batuk yang lolos filter.")
    exit()

# --- ENCODING LABEL ---
label_list = sorted(long_df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(label_list)}
print(f"Total kelas: {len(label_list)}")
print(f"Daftar label: {label_list}")

# --- PEMBAGIAN DATA TRAIN/VAL/TEST YANG LEBIH BAIK ---
file_paths = long_df['file_path'].tolist()
labels = [label2id[label] for label in long_df['label']]

# Split pertama: Train+Val vs Test
X_temp, X_test_fp, y_temp, y_test = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# Split kedua: Train vs Val
X_train_fp, X_val_fp, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-0.2), stratify=y_temp, random_state=SEED
)

print("Distribusi label train:", {label_list[i]:c for i,c in Counter(y_train).items()})
print("Distribusi label val:", {label_list[i]:c for i,c in Counter(y_val).items()})
print("Distribusi label test:", {label_list[i]:c for i,c in Counter(y_test).items()})

# --- AUGMENTASI YANG DITINGKATKAN ---
def augment_audio(y, sr):
    """Terapkan teknik augmentasi audio modern"""
    choices = ['noise', 'pitch', 'stretch', 'speed', 'volume', 'none']
    weights = [0.2, 0.2, 0.15, 0.15, 0.2, 0.1]
    choice = np.random.choice(choices, p=weights)
    
    if choice == 'noise':
        noise_factor = np.random.uniform(0.002, 0.01)
        noise = np.random.randn(len(y))
        y_aug = y + noise_factor * noise
        return np.clip(y_aug, -1.0, 1.0)
    elif choice == 'pitch':
        n_steps = np.random.uniform(-1.5, 1.5)
        return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    elif choice == 'stretch':
        rate = np.random.uniform(0.85, 1.15)
        y_aug = librosa.effects.time_stretch(y, rate)
        if len(y_aug) > len(y):
            y_aug = y_aug[:len(y)]
        else:
            y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)), mode='constant')
        return y_aug
    elif choice == 'speed':
        rate = np.random.uniform(0.95, 1.05)
        y_aug = librosa.effects.time_stretch(y, rate)
        return y_aug[:len(y)] if len(y_aug) > len(y) else np.pad(y_aug, (0, len(y)-len(y_aug)), 'constant')
    elif choice == 'volume':
        gain = np.random.uniform(0.8, 1.2)
        return np.clip(y * gain, -1.0, 1.0)
    else:
        return y

def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, sr=SAMPLE_RATE, augment=False):
    """Ekstrak fitur MFCC dengan normalisasi yang baik dan augmentasi opsional"""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if len(y) < sr * 0.1:  # Lewati audio yang sangat singkat (< 0.1s)
            return None
            
        if augment:
            y = augment_audio(y, sr)
            
        # Ekstrak MFCC dan fitur delta
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Terapkan normalisasi global
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        # Lakukan padding atau potong untuk panjang tetap
        if mfccs.shape[1] < max_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- STRATEGI AUGMENTASI CERDAS ---
y_train_arr = np.array(y_train)
class_counts_train = Counter(y_train)
max_samples = max(class_counts_train.values())
print(f"Jumlah sampel terbanyak di satu kelas: {max_samples}")

# Hitung multiplier augmentasi untuk setiap kelas
aug_multipliers = {}
for label_id in range(len(label_list)):
    current_count = class_counts_train.get(label_id, 0)
    if current_count > 0:
        # Targetkan balancing kelas minimal 70% dari kelas terbesar
        target_count = max(int(max_samples * 0.7), current_count)
        multiplier = max(1, target_count // current_count)
        aug_multipliers[label_id] = min(multiplier, 8)  # Batasi maksimal 8x
    else:
        aug_multipliers[label_id] = 1

print("Multiplier augmentasi:", {label_list[i]: aug_multipliers[i] for i in aug_multipliers})

# --- EKSTRAKSI FITUR SECARA BATCH ---
def extract_features_batch(file_paths, labels, augment=False, desc=""):
    X, y_new = [], []
    gagal = 0
    
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(labels), desc=desc):
        # Proses sampel asli
        mfcc = extract_mfcc(fp)
        if mfcc is not None:
            X.append(mfcc)
            y_new.append(lbl)
            
            # Tambahkan augmented sample khusus train
            if augment and lbl in aug_multipliers:
                n_aug = aug_multipliers[lbl] - 1  # -1 karena sudah ada original
                for _ in range(n_aug):
                    mfcc_aug = extract_mfcc(fp, augment=True)
                    if mfcc_aug is not None:
                        X.append(mfcc_aug)
                        y_new.append(lbl)
        else:
            gagal += 1
    
    if gagal > 0:
        print(f"⚠️ {gagal} file gagal diproses")
    
    return np.array(X), np.array(y_new)

print("🔄 Proses ekstraksi fitur MFCC...")
X_train, y_train = extract_features_batch(X_train_fp, y_train, augment=AUGMENT, desc="Training")
X_val, y_val = extract_features_batch(X_val_fp, y_val, augment=False, desc="Validation")
X_test, y_test = extract_features_batch(X_test_fp, y_test, augment=False, desc="Testing")

# Tambahkan dimensi channel
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"✅ Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("Distribusi FINAL - Train:", {label_list[i]:c for i,c in Counter(y_train).items()})
print("Distribusi FINAL - Val:", {label_list[i]:c for i,c in Counter(y_val).items()})
print("Distribusi FINAL - Test:", {label_list[i]:c for i,c in Counter(y_test).items()})

# --- HITUNG CLASS WEIGHTS UNTUK TRAINING ---
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", {label_list[i]: class_weight_dict[i] for i in class_weight_dict})

# === ARSITEKTUR MODEL YANG DITINGKATKAN ===
def create_improved_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Blok konvolusi pertama
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Blok konvolusi kedua
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Blok konvolusi ketiga
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Pakai global pooling, bukan flatten
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layer dengan regularisasi
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='label')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

model = create_improved_model((N_MFCC, MAX_LEN, 1), len(label_list))

# Gunakan optimizer yang lebih baik
optimizer = keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- CALLBACKS YANG DITINGKATKAN ---
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True, 
        mode='max',
        min_delta=0.001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=7, 
        min_lr=1e-7, 
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("🚀 Mulai proses training model klasifikasi batuk...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Muat model terbaik
model = keras.models.load_model('best_model.keras')

# --- EVALUASI MENYELURUH ---
print("🔍 Evaluasi model pada data test:")
test_results = model.evaluate(X_test, y_test, verbose=0)
test_acc = test_results[1]
print(f"✅ Akurasi test: {test_acc:.4f}")

print("🔍 Evaluasi model pada data validasi:")
val_results = model.evaluate(X_val, y_val, verbose=0)
val_acc = val_results[1]
print(f"✅ Akurasi validasi: {val_acc:.4f}")

# --- ANALISIS MENDALAM ---
print("\n=== ANALISIS PERFORMA MODEL ===")
y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)

# Analisis test set
print("\n--- HASIL DI TEST SET ---")
cm_test = confusion_matrix(y_test, y_pred_test, labels=list(range(len(label_list))))
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=label_list, digits=4))

# Analisis validation set
print("\n--- HASIL DI VALIDATION SET ---")
cm_val = confusion_matrix(y_val, y_pred_val, labels=list(range(len(label_list))))
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_pred_val, target_names=label_list, digits=4))

# Visualisasikan confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion matrix test set
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
           xticklabels=label_list, yticklabels=label_list, ax=axes[0])
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Label Asli')
axes[0].set_title('Confusion Matrix - Test Set')

# Confusion matrix validation set
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', 
           xticklabels=label_list, yticklabels=label_list, ax=axes[1])
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Label Asli')
axes[1].set_title('Confusion Matrix - Validation Set')

plt.tight_layout()
plt.show()

# Plot history training
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Analisis overfitting
train_acc = max(history.history['accuracy'])
val_acc_final = max(history.history['val_accuracy'])
overfitting_gap = train_acc - val_acc_final

print(f"\n=== ANALISIS OVERFITTING ===")
print(f"Akurasi Training Akhir: {train_acc:.4f}")
print(f"Akurasi Validasi Terbaik: {val_acc_final:.4f}")
print(f"Gap Overfitting: {overfitting_gap:.4f}")

if overfitting_gap > 0.1:
    print("⚠️ Model terindikasi overfitting (gap > 0.1)")
elif overfitting_gap > 0.05:
    print("⚠️ Model sedikit overfitting (gap > 0.05)")
else:
    print("✅ Model tidak overfitting")

# --- SIMPAN MODEL ---
model.save("smartcough_model_cough5class_improved.keras")
print("✅ Model disimpan sebagai smartcough_model_cough5class_improved.keras")

# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("smartcough_model_cough5class_improved.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model berhasil disimpan sebagai smartcough_model_cough5class_improved.tflite")

# --- RINGKASAN AKHIR ---
print(f"\n{'='*50}")
print("RINGKASAN HASIL TRAINING")
print(f"{'='*50}")
print(f"Total kelas: {len(label_list)}")
print(f"Daftar kelas: {label_list}")
print(f"Jumlah data training: {len(y_train)} sampel")
print(f"Jumlah data validasi: {len(y_val)} sampel") 
print(f"Jumlah data testing: {len(y_test)} sampel")
print(f"Akurasi test: {test_acc:.4f}")
print(f"Akurasi validasi: {val_acc:.4f}")
print(f"Gap overfitting: {overfitting_gap:.4f}")

if test_acc > 0.8 and overfitting_gap < 0.1:
    print("✅ MODEL BERKUALITAS BAIK - Siap untuk deployment!")
elif test_acc > 0.7:
    print("⚠️ MODEL CUKUP BAIK - Masih bisa ditingkatkan")
else:
    print("❌ MODEL PERLU PERBAIKAN - Accuracy terlalu rendah")