# 🚀 SMARTCOUGH - Dokumentasi Klasifikasi Audio Batuk 5 Kelas

> **Biar suara batuk lo nggak cuma jadi noise, sekarang bisa langsung dideteksi pakai Deep Learning! Model ini sudah siap tampil kece di HP & Edge Device.**

<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExazh1dDVpc3N3bmR6eno4NnI5eGZxNWx1N253b3dkbDZjMXJ6bnA2YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/137EaR4vAOCn1S/giphy.gif" alt="lofi chill animation" width="220"/>
  <br>
  <i style="color:#6C3483;font-weight:bold;">Ngoding santai sambil batuk? SMARTCOUGH siap mendeteksi!</i>
</p>

---

## 📖 Apa itu SMARTCOUGH?

SMARTCOUGH adalah pipeline machine learning untuk mengklasifikasikan audio batuk ke dalam 5 kategori utama:  
`dry_cough`, `wet_cough`, `allergy_cough`, `covid_cough`, `sneeze_flu`.

Modelnya menggunakan CNN (Convolutional Neural Network) yang memproses fitur MFCC dari audio batuk kamu. Output-nya bisa diekspor langsung ke `.keras` & `.tflite`. Tinggal colok ke aplikasi mobile, langsung jalan tanpa ribet!

---

## 📂 Struktur Folder

```
audio/
  dataset_audio/                    # Kumpulan suara batuk (.wav, .mp3, .ogg, .webm, .m4a)
csv/
  data_advanced_preprocessed.csv    # File CSV label hasil preprocessing
smartcough_model_cough5class_improved.keras   # Model hasil training (.keras)
smartcough_model_cough5class_improved.tflite  # Model siap deploy (.tflite)
```

---

## 🛠️ Library yang Wajib Dipasang

- Python >= 3.7
- numpy
- pandas
- librosa
- tensorflow >= 2.8
- scikit-learn
- tqdm
- seaborn
- matplotlib

> **Langsung copy ke terminal:**

```bash
pip install numpy pandas librosa tensorflow scikit-learn tqdm seaborn matplotlib
```

---

## ⚙️ Pengaturan Penting

```python
DATA_DIR = "audio/dataset_audio"
CSV_PATH = "C:\\SMARTCOUGH\\csv\\data_advanced_preprocessed.csv"
N_MFCC = 40
MAX_LEN = 130
SAMPLE_RATE = 16000
SEED = 42
AUGMENT = True
MIN_SAMPLES_PER_CLASS = 50
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.15
```

> _Ganti path kalau perlu, tetap fleksibel!_

---

## 🔬 Step-by-Step Proses

### 1️⃣ Menyiapkan Data

- Baca CSV label hasil preprocessing.
- Ambil UUID & info batuk per baris (`cough_type_X`, `quality_X`, `severity_X`).

### 2️⃣ Mapping Label & Filtering

- Ubah label mentah ke 5 kelas target.
- Gunakan hanya audio dengan kualitas `ok`/`good`.
- Kelas minoritas (< `MIN_SAMPLES_PER_CLASS`) dilewati agar model lebih stabil.

### 3️⃣ Augmentasi & Ekstraksi Fitur

- Cari file audio sesuai UUID.
- Augmentasi audio (opsional): noise, pitch shift, time stretch, speed, volume.
- Ekstrak MFCC (40 dimensi, max 130 frame, dinormalisasi).
- Padding/truncating supaya input seragam.

### 4️⃣ Membagi Data

- Split ke Train, Validation, Test (stratified).
- Train set di-augment supaya kelas seimbang.

### 5️⃣ Membuat Model

- CNN 2D: 3 blok Conv2D + BatchNorm + MaxPooling + Dropout.
- Global Average Pooling, Dense, Dropout, output Softmax.
- Optimizer: AdamW.
- Loss: SparseCategoricalCrossentropy.
- Class weighting otomatis.

### 6️⃣ Training & Callback

- EarlyStopping (patience 15, auto restore best model).
- ReduceLROnPlateau (pantau val_accuracy).
- ModelCheckpoint (simpan hanya model terbaik).
- Maksimal 100 epoch (auto stop jika sudah tidak ada improvement).

### 7️⃣ Evaluasi & Analisis

- Cek performa di validation & test set.
- Classification report & confusion matrix.
- Plot akurasi/loss training vs validation.
- Analisa overfitting (cek gap akurasi).

### 8️⃣ Simpan Model

- Output ke `.keras` & `.tflite`. Siap deploy ke device kamu!

---

## 💡 Tips dari SmartCough

- **Data imbalance?** Script sudah auto augment kelas minoritas.
- **Augmentasi?** Bisa dimatikan (`AUGMENT = False`) untuk training cepat.
- **Audio rusak/pendek?** Auto skip.
- **Label CSV wajib konsisten** untuk proses lancar.
- **.tflite** tinggal colok ke app mobile/edge device.

---

## 📊 Hasil Training & Log

### Model Summary

- **Total Parameter:** 206,946 (808.38 KB)
  - Trainable: 205,986 (804.63 KB)
  - Non-trainable: 960 (3.75 KB)

### Progress Training (Contoh)

```
Epoch 1/100
148/148 ... accuracy: 0.8726 ... val_accuracy: 0.9982
Epoch 2/100
148/148 ... accuracy: 0.9383 ... val_accuracy: 0.9982
...
```

### Hasil Evaluasi

- **Test accuracy:** `1.0000`
- **Validation accuracy:** `0.9982`

### Catatan Training

- Model otomatis menyimpan checkpoint dengan val_accuracy terbaik.
- Learning rate turun otomatis jika model stagnan.
- EarlyStopping: training otomatis berhenti jika tidak ada peningkatan 15 epoch.

### Interpretasi

- **Akurasi sangat tinggi** (99-100%) = model sangat fit ke data.
- **Saran:** Uji juga ke data lapangan, jangan cuma test di data training!

---

## 📤 Output & Hasil

Setelah training selesai, summary otomatis muncul:

- Jumlah kelas, distribusi data train/val/test
- Akurasi test & validation
- Gap overfitting
- Status model: **BAIK | CUKUP | PERLU PERBAIKAN**

#### Contoh Output Akhir

```
Test accuracy: 1.0000
Validation accuracy: 0.9982
Overfitting gap: 0.0018
✅ MODEL BERKUALITAS BAIK - Siap untuk deployment!
```

---

## 📞 Kontak & Ngobrol

Punya pertanyaan, masukan, atau mau ngulik bareng?  
Langsung aja DM/email/colek tim **SMARTCOUGH**:

- agii
- agam
- harr

---

<p align="center">
  <img src="https://media.giphy.com/media/IThjAlJnD9WNO/giphy.gif" alt="lofi guy" width="150"/>
  <br>
  <b style="color:#D4AC0D; font-size:1.2em;">SEMANGAT!! 🚀🔥</b>
</p>
