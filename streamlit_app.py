import streamlit as st
import requests
from PIL import Image
import io

# 🎨 Streamlit Sayfa Tasarımı
st.set_page_config(page_title="Human Detector AI", page_icon="🧑‍💻", layout="centered")

# 🎯 Başlık ve Açıklama
st.title("🧑‍💻 İnsan Tanıma Yapay Zekası")
st.markdown(
    """
    **Yapay Zeka destekli insan algılama modeli** kullanarak bir resmin insan yüzü olup olmadığını tespit edin.  
    Aşağıdan bir görüntü yükleyin ve sonucu anında görün!
    """
)

# 🚀 Yan Menü (Sidebar)
st.sidebar.header("⚙️ Ayarlar")
st.sidebar.markdown("Bu uygulama bir **CNN Modeli** kullanmaktadır.")

# 🌐 API URL (Canlı veya Lokal)
API_URL = "http://13.48.30.60:8000/predict/"

# 📤 Kullanıcıdan Görüntü Yüklemesini İste
uploaded_file = st.file_uploader("📸 Bir resim yükleyin:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🎨 Görseli Göster
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Yüklenen Görsel", use_container_width=True)

    # 🚀 Yükleme Sonrası Buton
    if st.button("📊 Analiz Et!"):
        # ⏳ Bekleme mesajı
        with st.spinner("🔍 Model tahmin yapıyor, lütfen bekleyin..."):
            try:
                # 🎯 Görüntüyü API'ye gönder
                img_bytes = io.BytesIO()
                image.convert("RGB").save(img_bytes, format="JPEG")  # RGBA HATASI ÇÖZÜLDÜ!
                response = requests.post(API_URL, files={"file": img_bytes.getvalue()})
                response.raise_for_status()  # HTTP hatalarını yakala

                # 📌 API Yanıtını Kontrol Et
                result = response.json()
                prediction = result.get("prediction", "Bilinmiyor")
                confidence = result.get("confidence", 0.0)

                # 🎨 Sonucu Göster
                if prediction == "Human":
                    st.success(f"✅ **İnsan Algılandı!** (Güven Skoru: {confidence:.2f})")
                else:
                    st.error(f"❌ **İnsan Algılanmadı!** (Güven Skoru: {confidence:.2f})")

                # 📊 Skoru ilerleme çubuğu ile göster
                st.progress(confidence)

            except requests.exceptions.ConnectionError:
                st.error("⚠️ API çalışmıyor! Lütfen önce API sunucusunu başlatın.")
            except Exception as e:
                st.error(f"⚠️ Hata oluştu: {e}")
