import subprocess
import time

# Docker container ve image isimleri
CONTAINER_NAME = "detector_container"
IMAGE_NAME = "alknumt/detector_app:latest"

# RAM sınırları
MEMORY_LIMIT = "512m"
MEMORY_RESERVATION = "256m"

# API Sağlamlık Kontrol URL'si
API_URL = "http://localhost:8000"

def is_api_running():
    """API sunucusunun çalışıp çalışmadığını kontrol eder."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{API_URL}/docs"],
            capture_output=True, text=True
        )
        return result.stdout.strip() == "200"  # API /docs endpoint'ine cevap veriyorsa çalışıyordur
    except Exception as e:
        print(f"[X] API kontrol edilirken hata oluştu: {e}")
        return False

def is_container_running():
    """Docker konteynerinin çalışıp çalışmadığını kontrol eder."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        return CONTAINER_NAME in result.stdout.strip()
    except Exception as e:
        print(f"[X] Konteyner kontrol edilirken hata oluştu: {e}")
        return False

def restart_container():
    """Eğer API çökmüşse, konteyneri yeniden başlatır."""
    print("[!!] API çöktü! Konteyner yeniden başlatılıyor...")
    result = subprocess.run(["docker", "restart", CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if result.returncode == 0:
        print("[+] Konteyner başarıyla yeniden başlatıldı.")
    else:
        print("[X] Konteyner yeniden başlatılamadı. Yeni bir tane başlatılıyor...")
        start_container()

def start_container():
    """Eğer konteyner tamamen çökmüşse, yeni bir tane başlatır."""
    print("[!!] Konteyner çalışmıyor! Yeni bir tane başlatılıyor...")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Eskiyi kaldır

    result = subprocess.run([
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "--memory=" + MEMORY_LIMIT, "--memory-reservation=" + MEMORY_RESERVATION,
        "-p", "8000:8000", "-p", "8501:8501", IMAGE_NAME
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0:
        print("[+] Yeni konteyner başarıyla başlatıldı.")
    else:
        print("[X] Konteyner başlatılamadı! Logları kontrol et.")

if __name__ == "__main__":
    while True:
        if is_container_running():
            if not is_api_running():
                restart_container()
        else:
            start_container()
        
        time.sleep(30)  # Her 30 saniyede bir kontrol et
