# 🧑‍💻 İnsan Tanıma Yapay Zekası (Human Detection AI)

CNN (ResNet50) tabanlı, görüntülerden insan yüzü tanıyabilen makine öğrenmesi uygulaması.

---

## 🎯 Kullanılan Teknolojiler ve Yapılar

- **Model:** ResNet50 (Transfer Learning ve Fine-Tuning yöntemiyle eğitildi)
- **Framework:** PyTorch, FastAPI, Streamlit
- **Docker:** Uygulama konteynerize edilerek hazır hale getirildi.
- **Bulut:** AWS EC2 (Ubuntu)
- **Veri Seti:** https://www.kaggle.com/datasets/aliasgartaksali/human-and-non-human
---

## 🚀 Proje Yapısı

```
.
├── Dockerfile
├── requirements.txt
├── streamlit_app.py
├── app
│   └── app.py
└── model
    └── final_model.pth
```

---

## 📦 Kurulum ve Çalıştırma

### Docker (Önerilen)

Docker konteynerini doğrudan Docker Hub üzerinden çekerek çalıştırabilirsiniz:

```bash
docker pull alknumt/detector_app:latest
docker run -d -p 8000:8000 -p 8501:8501 alknumt/detector_app:latest
```

veya kendi Docker imajınızı oluşturmak için:

```bash
docker build -t human-detector-app .
docker run -d -p 8000:8000 -p 8501:8501 human-detector-app
```

---

## 🌐 Kullanım

- **Streamlit Arayüzü:** Tarayıcınızdan uygulamanıza ulaşmak için:
```
http://http://13.48.30.60:8501
```

---

## 📌 Kullanılan Versiyonlar

| Teknoloji    | Versiyon   |
|--------------|------------|
| Python       | 3.9        |
| FastAPI      | 0.110.1    |
| Streamlit    | 1.32.2     |
| PyTorch      | 2.2.1      |
| Docker       | latest     |
| AWS EC2      | Ubuntu     |

---

## 📸 Uygulama Ekran Görüntüleri

>![kadın](<Ekran görüntüsü 2025-03-07 181125.png>)

>![araba](<Ekran görüntüsü 2025-03-07 181110.png>)

>![çocuk](<Ekran görüntüsü 2025-03-07 180847.png>)

>![apple](<Ekran görüntüsü 2025-03-07 181048.png>)

---

## 📧 İletişim ve Destek

Sorularınız ve önerileriniz için iletişime geçebilirsiniz:

- LinkedIn: [LinkedIn Profiliniz](https://www.linkedin.com/in/umutalkan42/)
- E-posta: `alkanumut848@gmail.com`

---