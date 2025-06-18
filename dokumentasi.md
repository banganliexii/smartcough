# 🚀 SMARTCOUGH - Dokumentasi Klasifikasi Audio Batuk 5 Kelas

> **Biar suara batuk lo gak cuma jadi noise, sekarang bisa langsung dideteksi pake Deep Learning! Model ini udah siap tempur buat HP & Edge Device, bro!**

<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExazh1dDVpc3N3bmR6eno4NnI5eGZxNWx1N253b3dkbDZjMXJ6bnA2YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/137EaR4vAOCn1S/giphy.gif" alt="lofi chill animation" width="240"/>
  <br><i>Ngoding santai sambil batuk? SMARTCOUGH tetap stand by deteksiin! ☕🎧🦠</i>
</p>

---

## 📖 Apa itu SMARTCOUGH?

Jadi gini, **SMARTCOUGH** basically adalah pipeline machine learning yang ngebantu lo ngelabelin audio batuk ke 5 kategori utama:  
`dry_cough`, `wet_cough`, `allergy_cough`, `covid_cough`, `sneeze_flu`.  
Modelnya sendiri pake CNN (Convolutional Neural Network) yang ngolah fitur MFCC dari audio batuk lo. Output-nya bisa diekspor langsung ke `.keras` & `.tflite`. Tinggal colok ke mobile app lo, langsung jalan, no drama!

---

## 📂 Struktur Folder (Simple & Rapi)

```
audio/
  dataset_audio/                    # Kumpulan suara batuk (.wav, .mp3, .ogg, .webm, .m4a)
csv/
  data_advanced_preprocessed.csv    # File CSV label hasil preprocessing
smartcough_model_cough5class_improved.keras   # Model hasil training (.keras)
smartcough_model_cough5class_improved.tflite  # Model buat deploy (.tflite)
```

---

## 🛠️ Tools & Library Wajib

- Python >= 3.7
- numpy
- pandas
- librosa
- tensorflow >= 2.8
- scikit-learn
- tqdm
- seaborn
- matplotlib

> **Gak usah ribet, langsung copy ke terminal:**

```bash
pip install numpy pandas librosa tensorflow scikit-learn tqdm seaborn matplotlib
```

---

## ⚙️ Setting-an Penting

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

> _Gampang kok, tinggal ganti path kalo mau pindah lokasi data!_

---

## 🔬 Step-by-Step Proses (No Drama, No Ribet)

### 1️⃣ Data Preparation

- Baca CSV label hasil preprocessing.
- Ambil UUID & info batuk per baris (`cough_type_X`, `quality_X`, `severity_X`). Gampang.

### 2️⃣ Label Mapping & Filtering

- Ubah label mentah ke 5 kelas target.
- Filter audio yang kualitasnya cuma `ok`/`good` (no bad vibes here).
- Kelas minoritas (< `MIN_SAMPLES_PER_CLASS`)? Skip aja biar model lo gak bias.

### 3️⃣ Augmentasi & Ekstraksi Fitur

- Cari file audio sesuai UUID (auto detect).
- Augmentasi audio (optional, bisa dimatiin juga): noise, pitch shift, time stretch, speed, volume biar model makin robust.
- Ekstrak MFCC (40 dimensi, max 130 frame, udah dinormalisasi).
- Padding/truncating biar input seragam, anti ribet-ribet club.

### 4️⃣ Split Data

- Data dibagi adil: Train, Validation, Test (stratified, biar gak berat sebelah).
- Train set di-augment, biar semua kelas dapet jatah yang fair.

### 5️⃣ Bikin Modelnya

- CNN 2D, 3 blok Conv2D + BatchNorm + MaxPooling + Dropout.
- Global Average Pooling, Dense, Dropout, output Softmax.
- Optimizer: AdamW (udah ada weight decay, anti cepat bosen)
- Loss: SparseCategoricalCrossentropy.
- Class weighting otomatis, biar kelas minoritas tetap dihargai.

### 6️⃣ Training & Callback

- EarlyStopping (patience 15, auto restore best model)
- ReduceLROnPlateau (monitor val_accuracy, biar model makin pinter)
- ModelCheckpoint (save best model only)
- Maksimal 100 epoch, auto stop kalo udah stuck.

### 7️⃣ Evaluasi & Analisis

- Cek performa di validation & test set.
- Classification report & confusion matrix, detail tiap kelas.
- Plot akurasi/loss training vs validation.
- Analisa overfitting (gap akurasi, biar model gak cuma jago kandang).

### 8️⃣ Save Model

- Output ke `.keras` & `.tflite`. Tinggal deploy, langsung gaskeun ke device lo!

---

## 💡 Tips & Lifehack ala SmartCough

- **Data Imbalance?** Chill, script udah auto augment kelas minoritas (no bias, no cry).
- **Augmentasi?** Bisa dimatiin (`AUGMENT = False`) kalo mau training lebih cepet.
- **Audio rusak/pendek?** Auto skip, anti baper.
- **Label harus konsisten** di CSV, biar gak error di tengah jalan.
- **Mobile/Edge?** `.tflite` tinggal colok ke app, plug and play!

---

## 📊 Hasil Training & Log (Biar Gak Kudet)

### 🏗️ Model Summary

- **Total Parameter:** 206,946 (808.38 KB)
  - Trainable: 205,986 (804.63 KB)
  - Non-trainable: 960 (3.75 KB)

### 🚦 Training Progress (Real Example)

```
Epoch 1/100
148/148 ━━━━━━━━━━━━━━━━━━━━ 26s 155ms/step - accuracy: 0.8726 - loss: 0.8716 - val_accuracy: 0.9982 - val_loss: 0.0219
Epoch 2/100
148/148 ━━━━━━━━━━━━━━━━━━━━ 21s 144ms/step - accuracy: 0.9383 - loss: 3.0762 - val_accuracy: 0.9982 - val_loss: 0.0206
...
Epoch 8/100
148/148 ━━━━━━━━━━━━━━━━━━━━ 21s 144ms/step - accuracy: 0.8577 - loss: 0.5519 - val_accuracy: 0.5115 - val_loss: 5.1902
...
Epoch 16/100
148/148 ━━━━━━━━━━━━━━━━━━━━ 26s 176ms/step - accuracy: 0.9069 - loss: 0.1754 - val_accuracy: 0.9876 - val_loss: 0.0476
```

### 🏅 Hasil Evaluasi

- **Test accuracy:** `1.0000` (serius, ini beneran)
- **Validation accuracy:** `0.9982`

### 📈 Catatan Training

- Model auto-save checkpoint yang val_accuracy-nya paling oke.
- Learning rate turun otomatis kalo model udah mulai mager.
- EarlyStopping: training auto stop kalo udah gak improve 15 epoch.

### 🔎 Interpretasi

- **Akurasi tinggi banget** (99-100%) = model udah fit banget sama data.
- **Saran:** Coba juga di data lapangan biar gak cuma jago kandang, bro. Real world test itu penting!

---

## 📤 Output & Hasil

Setelah training, summary bakal keluar otomatis:

- Jumlah kelas, distribusi data train/val/test
- Akurasi test & validation
- Overfitting gap (buat ngecek model ngibul apa enggak)
- Status model: **BAIK | CUKUP | PERLU PERBAIKAN**

#### Output Akhir (REAL NYA)

```
Test accuracy: 1.0000
Validation accuracy: 0.9982
Overfitting gap: 0.0018
✅ MODEL BERKUALITAS BAIK - Siap untuk deployment!
```

---

## 📞 Kontak & Ngobrol

Punya pertanyaan, masukan, atau pengen ngulik bareng?  
Feel free buat DM, email, atau colek aja tim **SMARTCOUGH**

- agii
- agam
- harr

---

<p align="center">
  <img src="https://media.giphy.com/media/IThjAlJnD9WNO/giphy.gif" alt="lofi guy" width="200"/>
  <br>
  <b>SEMANGAT!! 🚀🔥</b>
</p>
