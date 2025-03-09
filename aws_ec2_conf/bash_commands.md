# Docker FastAPI Watchdog Kurulum Rehberi

## 1) Watchdog Python Scriptini Çalıştırılabilir Yapın
Aşağıdaki komut ile `watchdog.py` dosyasını çalıştırılabilir hale getirin:
```bash
chmod +x /home/ubuntu/watchdog.py
```

## 2) Systemd Servis Dosyasını Oluşturun
Aşağıdaki komut ile systemd servis dosyanızı açın:
```bash
sudo nano /etc/systemd/system/docker-watchdog.service
```

### 2.1) Aşağıdaki İçeriği Dosyaya Ekleyin:
```ini
[Unit]
Description=Docker FastAPI Watchdog
After=network.target docker.service
Requires=docker.service

[Service]
ExecStart=/usr/bin/python3 /home/ubuntu/watchdog.py
Restart=always
User=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 3) Systemd Servisini Yükleyin ve Başlatın
Servisi etkinleştirmek ve çalıştırmak için aşağıdaki komutları sırasıyla çalıştırın:
```bash
sudo systemctl daemon-reload
sudo systemctl enable docker-watchdog
sudo systemctl start docker-watchdog
```

## 4) Servis Durumunu Kontrol Edin
Watchdog'un çalıştığını doğrulamak için:
```bash
sudo systemctl status docker-watchdog
```

Bu adımları uyguladıktan sonra, FastAPI konteyneriniz çöktüğünde Watchdog otomatik olarak onu yeniden başlatacaktır.

