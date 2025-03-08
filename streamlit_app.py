import streamlit as st
import requests
from PIL import Image
import io
import time
import threading
import psutil  # 🛠 RAM Kullanımı Kontrolü için

# 🚀 RAM Kullanımını Kontrol Ederek Cache Temizleme
def clear_cache_if_needed():
    while True:
        ram_usage = psutil.virtual_memory().percent
        if ram_usage > 85:  # RAM %85'i aşarsa cache temizle
            st.cache_data.clear()
            print("⚠️ RAM Yüksek! Cache Temizlendi!")
        time.sleep(600)  # 10 dakikada bir kontrol et

# 🛠 RAM Temizleme Thread'i (1 Kez Çalıştırılır)
if "cache_clear_thread" not in st.session_state:
    thread = threading.Thread(target=clear_cache_if_needed, daemon=True)
    thread.start()
    st.session_state["cache_clear_thread"] = True

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

# 🌐 API URL'yi Dinamik Yap
default_api_url = "http://51.21.193.154:8000/predict/"
API_URL = st.sidebar.text_input("🌐 API URL", default_api_url)

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
                st.error("🚨 API’ye bağlanılamıyor! Sunucu çalışıyor mu?")
            except requests.exceptions.Timeout:
                st.error("⏳ API çok yavaş yanıt veriyor! Sunucuyu kontrol edin.")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ API Hatası: {e}")
            except Exception as e:
                st.error(f"❌ Beklenmeyen bir hata oluştu: {str(e)}")
