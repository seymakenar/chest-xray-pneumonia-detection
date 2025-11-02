# Chest X-Ray Pneumonia Detection

Bu proje, göğüs röntgeni görüntülerinden zatürre (Pneumonia) tespiti yapan bir derin öğrenme modelidir. Model, Transfer Learning tekniğiyle ResNet18 mimarisi kullanılarak geliştirilmiştir.

---

Model Bilgisi

| Özellik | Açıklama |
|--------|---------|
Model | ResNet18 (PyTorch)
Amaç | Normal / Pnömoni sınıflandırma
Doğruluk | ~%79 (Test verisi)
Veri Seti | Kaggle Chest X-Ray Pneumonia Dataset

---

Veri Seti

Veri seti lisanslı olduğu için repoda paylaşılmamıştır. Aşağıdaki linkten indirip `data/chest_xray/` dizinine yerleştiriniz:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Klasör yapısı şu şekilde olmalıdır:

data/chest_xray/
├── train/
├── test/
└── val/

---

Kurulum ve Çalıştırma

Sanal ortam oluşturma
python -m venv venv

Sanal ortamı aktifleştirme (Windows)
venv\Scripts\activate

Gerekli kütüphanelerin yüklenmesi
pip install -r requirements.txt

Model eğitimi
python src/train_model.py

Model testi
python test_model.py

Tek resim ile tahmin
python src/predict.py sample_images/test1.jpg

Proje Yapısı
├── data/                # Dataset (local)
├── models/              # Eğitilmiş model
├── sample_images/       # Test görüntüleri
├── src/
│   ├── train_model.py   # Eğitim scripti
│   └── predict.py       # Tahmin scripti
├── test_model.py        # Test scripti
└── requirements.txt

Model Dosyası

Bu projede kullanılan eğitilmiş model dosyasını aşağıdaki bağlantıdan indirebilirsiniz:

Model indirme linki:
https://drive.google.com/file/d/12AyFzI6HYCSANyIEfRaVqwxLEDXZ1NbC/view?usp=drive_link

İndirdikten sonra models/ klasörü içine koymanız gerekir:

project_root/
└── models/
    └── chest_xray_model.pth

Notlar

Derin öğrenme ve transfer learning uygulanmıştır.
PyTorch ve Torchvision kullanılmıştır.
DataLoader ve görüntü dönüşümleri uygulanmıştır.
GPU/CPU üzerinde çalıştırılabilir.

Geliştirici
Şeyma Kenar

Bu proje akademik araştırma / ödev kapsamında hazırlanmıştır.




