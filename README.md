# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

**Nama :** Rahmatdi

**Username Dicoding:** rahmatdi_rIZz

**Repository GitHub:** [Tautan ke Repository GitHub](https://github.com/3xlagi/mlops-jantung-deploy)


## 1. Informasi Dataset
Proyek ini menggunakan **[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)** yang bersumber dari Kaggle. Dataset ini berisi data rekam medis pasien yang digunakan untuk memprediksi kemungkinan terjadinya gagal jantung.
* **Jumlah Data:** 918 baris
* **Fitur Kategorikal (5):** `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`.
* **Fitur Numerik (6):** `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`.
* **Target/Label:** `HeartDisease` (0: Normal, 1: Memiliki Penyakit Jantung).

## 2. Persoalan yang Ingin Diselesaikan
Penyakit jantung kronis merupakan salah satu penyebab kematian tertinggi. Identifikasi dini sering kali terhambat karena banyaknya variabel medis yang harus dianalisis oleh dokter. Jika keterlambatan deteksi terus terjadi, risiko fatal bagi pasien akan semakin tinggi.

## 3. Solusi Machine Learning dan Target
* **Solusi:** Membangun model *Machine Learning* (klasifikasi biner) yang mampu memprediksi apakah seorang pasien memiliki risiko penyakit jantung atau tidak, berdasarkan 11 input variabel medis. Pipeline MLOps dibangun secara *end-to-end* menggunakan **TensorFlow Extended (TFX)** untuk memastikan skalabilitas.
* **Target:** Menghasilkan model dengan tingkat akurasi (Binary Accuracy) di atas 85% pada data validasi, serta memastikan model tersebut siap melayani *request* secara *real-time* di lingkungan *cloud* (*Production*).

## 4. Metode Pengolahan Data dan Arsitektur Model
* **Pengolahan Data:** Dilakukan melalui komponen `Transform` pada TFX. Fitur numerik diskalakan menggunakan *MinMax Scaler* (skala 0-1) agar komputasi lebih stabil. Fitur kategorikal diubah menjadi representasi matriks menggunakan teknik *One-Hot Encoding*.
* **Hyperparameter Tuning:** Menggunakan komponen **Tuner** (KerasTuner - RandomSearch) untuk mencari kombinasi arsitektur dan *learning rate* yang paling optimal secara otomatis.
* **Arsitektur Model Terbaik:** Menggunakan *Deep Learning* (Keras Functional API) berdasarkan hasil *tuning*:
  - Input Layer: Menyatukan (*concatenate*) seluruh fitur numerik dan kategorikal (total dimensi input 25).
  - Hidden Layer 1: Dense (64 neuron, ReLU) + Dropout (0.1).
  - Hidden Layer 2: Dense (32 neuron, ReLU).
  - Hidden Layer 3: Dense (32 neuron, ReLU).
  - Output Layer: Dense (1 neuron) dengan fungsi aktivasi *Sigmoid* (untuk klasifikasi biner).
  - Optimizer: Adam dengan Learning Rate 0.01.
* **Metrik Evaluasi:** `BinaryAccuracy`, `AUC`, `Precision`, `Recall`, dan `ExampleCount` (menggunakan ambang batas / *threshold* 0.5).

## 5. Performa Model
Dalam pengembangan *pipeline* ini, terdapat mekanisme evaluasi otomatis untuk memastikan hanya model terbaik yang masuk ke tahap *production*.

1. **Model Baseline (Deployment Saat Ini):** Pada iterasi awal, model berhasil mencapai performa yang sangat baik dengan **Training Accuracy ~99%** dan **Validation Accuracy ~85%**. Model ini mendapatkan restu dari komponen `Evaluator` (*Blessing = True*) dan berhasil di-*push* ke dalam *serving model* untuk di-*deploy* ke Railway.
2. **Model Hasil Tuning:** Setelah dilakukan pencarian arsitektur baru menggunakan komponen `Tuner`, model baru menghasilkan *Validation Accuracy* sekitar **~83%**. 
3. **Mekanisme Proteksi (MLOps):** Karena performa model baru (~83%) tidak berhasil melampaui performa *baseline* model lama (~85%), komponen `Evaluator` dengan cerdas menolak model baru tersebut (*Blessing = False*). Komponen `Pusher` kemudian membatalkan ekspor model baru, sehingga menjaga API di *production* tetap menggunakan model versi lama yang lebih stabil dan akurat.

## 6. Model Deployment
* **Platform:** Sistem di-deploy menggunakan **Railway** (Platform as a Service).
* **Metode:** Model yang diekspor dari TFX dibungkus ke dalam Docker Container menggunakan *image* bawaan `tensorflow/serving`. *Port* REST API (8501) diteruskan (*forwarded*) melalui konfigurasi lingkungan *cloud*.

**Tautan Web App (Endpoint):** [https://mlops-jantung-deploy-production.up.railway.app/v1/models/rahmatdi-model](https://mlops-jantung-deploy-production.up.railway.app/v1/models/rahmatdi-model)

## 7. Hasil Monitoring
Sistem telah dimonitor secara terus-menerus menggunakan kombinasi **Prometheus** (untuk *metrics scraping*) dan **Grafana** (untuk visualisasi). 
* Berdasarkan *dashboard* yang dibuat, sistem berhasil merekam *Total Requests*, *Error Requests*, dan memantau *Rata-rata Waktu Respons* (Average Response Time).
* Hasil prediksi via REST API (menggunakan data *dummy*) mengembalikan respons kode 200 (sukses), yang menandakan bahwa TF-Serving dan konfigurasi metrik berjalan dengan sangat optimal.