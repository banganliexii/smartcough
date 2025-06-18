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

# === CONFIG ===
DATA_DIR = "audio/dataset_audio"
CSV_PATH = "C:\\SMARTCOUGH\\csv\\data_advanced_preprocessed.csv"
N_MFCC = 40
MAX_LEN = 130
SAMPLE_RATE = 16000
SEED = 42
AUGMENT = True
MIN_SAMPLES_PER_CLASS = 50  # Increased from 20 to 50 for better quality
BATCH_SIZE = 16  # Increased batch size
VALIDATION_SPLIT = 0.15  # Separate validation set

np.random.seed(SEED)
tf.random.set_seed(SEED)

if not os.path.exists(DATA_DIR):
    print(f"❌ Folder '{DATA_DIR}' tidak ditemukan.")
    exit(1)

# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH)
df['uuid'] = df['uuid'].astype(str).str.strip()

# --- EXTRACT ONLY 5 TARGET COUGH TYPES ---
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

# --- LABEL MAPPING ---
label_map = {
    'dry': 'dry_cough',
    'wet': 'wet_cough',
    'allergy': 'allergy_cough',
    'COVID-19': 'covid_cough',
    'sneeze': 'sneeze_flu'
}
long_df['label'] = long_df['label_raw'].map(label_map)

print("Value counts label_raw sebelum map:")
print(long_df['label_raw'].value_counts(dropna=False))
print("Value counts label setelah map:")
print(long_df['label'].value_counts(dropna=False))

# --- FILTER: hanya 5 kelas target, audio kualitas ok/good ---
long_df = long_df[long_df['label'].notnull()].reset_index(drop=True)
long_df = long_df[long_df['quality'].isin(['good', 'ok'])].reset_index(drop=True)
print("Total audio batuk dipakai:", len(long_df))

# --- HAPUS KELAS MINORITAS EKSTRIM ---
class_counts = long_df['label'].value_counts()
print("Distribusi kelas sebelum filter:")
print(class_counts)

keep_labels = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
long_df = long_df[long_df['label'].isin(keep_labels)].reset_index(drop=True)
print(f"Kelas yang dipakai (>= {MIN_SAMPLES_PER_CLASS} sample):", keep_labels.tolist())

# Check if we have enough classes
if len(keep_labels) < 2:
    print("❌ Tidak cukup kelas untuk klasifikasi. Minimal butuh 2 kelas.")
    exit(1)

def to_severity_num(x):
    if pd.isnull(x): return 0
    try: return int(x)
    except: return 0
long_df['severity'] = long_df['severity'].apply(to_severity_num)

# --- CARI FILE AUDIO ---
def get_audio_path(uuid):
    allowed_ext = ('.webm', '.ogg', '.wav', '.mp3', '.m4a')
    for fname in os.listdir(DATA_DIR):
        if uuid in fname and fname.lower().endswith(allowed_ext):
            return os.path.join(DATA_DIR, fname)
    return None

long_df['file_path'] = long_df['uuid'].apply(get_audio_path)
long_df = long_df[long_df['file_path'].notnull()].reset_index(drop=True)
print(f"Total data batuk digunakan: {len(long_df)}")
if len(long_df) == 0:
    print("❌ Tidak ada data audio batuk yang lolos filter.")
    exit()

# --- LABEL ENCODING ---
label_list = sorted(long_df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(label_list)}
print(f"Total kelas: {len(label_list)}")
print(f"Label list: {label_list}")

# --- IMPROVED DATA SPLITTING (TRAIN/VAL/TEST) ---
file_paths = long_df['file_path'].tolist()
labels = [label2id[label] for label in long_df['label']]

# First split: Train+Val vs Test
X_temp, X_test_fp, y_temp, y_test = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# Second split: Train vs Val
X_train_fp, X_val_fp, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-0.2), stratify=y_temp, random_state=SEED
)

print("Distribusi label train:", {label_list[i]:c for i,c in Counter(y_train).items()})
print("Distribusi label val:", {label_list[i]:c for i,c in Counter(y_val).items()})
print("Distribusi label test:", {label_list[i]:c for i,c in Counter(y_test).items()})

# --- IMPROVED AUGMENTATION ---
def augment_audio(y, sr):
    """Improved augmentation techniques"""
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
    """Improved MFCC extraction with better normalization"""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if len(y) < sr * 0.1:  # Skip very short audio (< 0.1s)
            return None
            
        if augment:
            y = augment_audio(y, sr)
            
        # Extract MFCC with delta features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Global normalization
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        # Padding/truncating
        if mfccs.shape[1] < max_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- SMART AUGMENTATION STRATEGY ---
y_train_arr = np.array(y_train)
class_counts_train = Counter(y_train)
max_samples = max(class_counts_train.values())
print(f"Maksimum sampel per kelas: {max_samples}")

# Calculate augmentation multiplier for each class
aug_multipliers = {}
for label_id in range(len(label_list)):
    current_count = class_counts_train.get(label_id, 0)
    if current_count > 0:
        # Target: balance classes to at least 70% of max class
        target_count = max(int(max_samples * 0.7), current_count)
        multiplier = max(1, target_count // current_count)
        aug_multipliers[label_id] = min(multiplier, 8)  # Cap at 8x
    else:
        aug_multipliers[label_id] = 1

print("Augmentation multipliers:", {label_list[i]: aug_multipliers[i] for i in aug_multipliers})

# --- EXTRACT FEATURES ---
def extract_features_batch(file_paths, labels, augment=False, desc=""):
    X, y_new = [], []
    failed_count = 0
    
    for fp, lbl in tqdm(zip(file_paths, labels), total=len(labels), desc=desc):
        # Original sample
        mfcc = extract_mfcc(fp)
        if mfcc is not None:
            X.append(mfcc)
            y_new.append(lbl)
            
            # Augmentation for training only
            if augment and lbl in aug_multipliers:
                n_aug = aug_multipliers[lbl] - 1  # -1 because we already have original
                for _ in range(n_aug):
                    mfcc_aug = extract_mfcc(fp, augment=True)
                    if mfcc_aug is not None:
                        X.append(mfcc_aug)
                        y_new.append(lbl)
        else:
            failed_count += 1
    
    if failed_count > 0:
        print(f"⚠️ Failed to process {failed_count} files")
    
    return np.array(X), np.array(y_new)

print("🔄 Ekstrak fitur MFCC...")
X_train, y_train = extract_features_batch(X_train_fp, y_train, augment=AUGMENT, desc="Training")
X_val, y_val = extract_features_batch(X_val_fp, y_val, augment=False, desc="Validation")
X_test, y_test = extract_features_batch(X_test_fp, y_test, augment=False, desc="Testing")

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"✅ Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("Distribusi FINAL - Train:", {label_list[i]:c for i,c in Counter(y_train).items()})
print("Distribusi FINAL - Val:", {label_list[i]:c for i,c in Counter(y_val).items()})
print("Distribusi FINAL - Test:", {label_list[i]:c for i,c in Counter(y_test).items()})

# --- COMPUTE CLASS WEIGHTS ---
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", {label_list[i]: class_weight_dict[i] for i in class_weight_dict})

# === IMPROVED MODEL ARCHITECTURE ===
def create_improved_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # First conv block
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Second conv block
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Third conv block
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Global pooling instead of flatten
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='label')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

model = create_improved_model((N_MFCC, MAX_LEN, 1), len(label_list))

# Use different optimizer with better settings
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

# --- IMPROVED CALLBACKS ---
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

print("🚀 Mulai training model klasifikasi batuk...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Load best model
model = keras.models.load_model('best_model.keras')

# --- COMPREHENSIVE EVALUATION ---
print("🔍 Evaluasi model pada data test:")
test_results = model.evaluate(X_test, y_test, verbose=0)
test_acc = test_results[1]
print(f"✅ Test accuracy: {test_acc:.4f}")

print("🔍 Evaluasi model pada data validation:")
val_results = model.evaluate(X_val, y_val, verbose=0)
val_acc = val_results[1]
print(f"✅ Validation accuracy: {val_acc:.4f}")

# --- DETAILED ANALYSIS ---
print("\n=== ANALISIS PERFORMA MODEL ===")
y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)

# Test set analysis
print("\n--- TEST SET RESULTS ---")
cm_test = confusion_matrix(y_test, y_pred_test, labels=list(range(len(label_list))))
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=label_list, digits=4))

# Validation set analysis
print("\n--- VALIDATION SET RESULTS ---")
cm_val = confusion_matrix(y_val, y_pred_val, labels=list(range(len(label_list))))
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_pred_val, target_names=label_list, digits=4))

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
           xticklabels=label_list, yticklabels=label_list, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix - Test Set')

# Validation confusion matrix
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', 
           xticklabels=label_list, yticklabels=label_list, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix - Validation Set')

plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Check for overfitting
train_acc = max(history.history['accuracy'])
val_acc_final = max(history.history['val_accuracy'])
overfitting_gap = train_acc - val_acc_final

print(f"\n=== OVERFITTING ANALYSIS ===")
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Best Validation Accuracy: {val_acc_final:.4f}")
print(f"Overfitting Gap: {overfitting_gap:.4f}")

if overfitting_gap > 0.1:
    print("⚠️ Model mungkin overfitting (gap > 0.1)")
elif overfitting_gap > 0.05:
    print("⚠️ Model sedikit overfitting (gap > 0.05)")
else:
    print("✅ Model tidak overfitting")

# --- SAVE MODELS ---
model.save("smartcough_model_cough5class_improved.keras")
print("✅ Model disimpan sebagai smartcough_model_cough5class_improved.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("smartcough_model_cough5class_improved.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model berhasil disimpan sebagai smartcough_model_cough5class_improved.tflite")

# --- FINAL SUMMARY ---
print(f"\n{'='*50}")
print("RINGKASAN HASIL TRAINING")
print(f"{'='*50}")
print(f"Total kelas: {len(label_list)}")
print(f"Kelas: {label_list}")
print(f"Data training: {len(y_train)} sampel")
print(f"Data validasi: {len(y_val)} sampel") 
print(f"Data testing: {len(y_test)} sampel")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Overfitting gap: {overfitting_gap:.4f}")

if test_acc > 0.8 and overfitting_gap < 0.1:
    print("✅ MODEL BERKUALITAS BAIK - Siap untuk deployment!")
elif test_acc > 0.7:
    print("⚠️ MODEL CUKUP BAIK - Masih bisa ditingkatkan")
else:
    print("❌ MODEL PERLU PERBAIKAN - Accuracy terlalu rendah")