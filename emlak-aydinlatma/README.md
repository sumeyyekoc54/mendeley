# Emlak Fotoğraf Aydınlatma (MVP)

Karanlık iç mekân emlak fotoğraflarını **içeriği değiştirmeden** doğal şekilde aydınlatan Windows-dostu Python uygulaması.

- Yalnızca **OpenCV + NumPy + Pillow** (deterministik). Üretken AI / diffusion **yok**.
- Lab L kanalında pozlama, yumuşak gölge eğrisi, highlight roll-off ve pencere koruması.
- Türkçe **Tkinter** arayüzü; isteğe bağlı headless CLI.

## Windows’ta çalıştırma

1. [Python 3.10+](https://www.python.org/downloads/) kurun (kurulumda “Add Python to PATH” işaretli olsun).
2. PowerShell veya cmd:

```bat
cd emlak-aydinlatma
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

3. Arayüzden 1–3 JPEG/PNG seçin → **İşle / Önizle** → beğenirseniz **Kaydet**.
4. Kayıtlar, kaynak klasörün yanındaki `duzenlenmis_fotograflar\` altına `<ad>_AYDINLIK.jpg` olarak yazılır (JPEG kalite ~95, piksel boyutu aynı). Orijinaller üzerine yazılmaz.

### Headless (toplu) işleme

```bat
python main.py --cli foto1.jpg foto2.jpg --out duzenlenmis_fotograflar --mode otomatik
```

### Testler

```bat
pip install pytest
python -m pytest tests -q
```

## HEIC / HEIF

Varsayılan kurulumda HEIC yoktur. Gerekirse:

```bat
pip install pillow-heif
```

Arayüz HEIC desteği varsa durum çubuğunda “HEIC: hazır” gösterir.

## Modlar

| Mod | Açıklama |
|-----|----------|
| Otomatik | Histogram analizine göre orta-hafif güç |
| Hafif | Küçük pozlama / gölge |
| Orta | Dengeli |
| Güçlü | Daha belirgin; yerel ton biraz artabilir |
| Şiddet kaydırıcısı | Seçilen modun çarpanı (≈0.4–1.4) |

**Pencere koruması** (varsayılan açık): parlak bölgelerde yumuşak maske ile yanmayı sınırlar.  
**Beyaz dengesi**: isteğe bağlı, hafif gri-dünya; varsayılan kapalı (duvar/mobilya renkleri korunur).

## Bilinen sınırlar (dürüst)

- Kaynakta yanmış (clip) pencereler / gökyüzü **geri getirilemez**; koruma yalnızca daha fazla yanmayı azaltır.
- Aşırı ISO gürültülü gölgeler güçlendirilince gürültü artabilir — Hafif mod kullanın.
- Bu MVP “profesyonel stüdyo kalitesi” iddiasında değildir; doğal, kontrollü aydınlatma iyileştirmesi sunar.
- `.exe` paketleme henüz yok (onay sonrası düşünülebilir).

## Proje yapısı

```
emlak-aydinlatma/
  main.py
  requirements.txt
  src/emlak_aydinlatma/
    analysis.py   # luminans / histogram analizi
    enhance.py    # pozlama, gölge, highlight, WB
    quality.py    # ölçülebilir kalite kapıları
    io_exif.py    # EXIF yön + kayıt
    pipeline.py   # modlar + orkestrasyon
    ui_app.py     # Tkinter
  tests/
```
