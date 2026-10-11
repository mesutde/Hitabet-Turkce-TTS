# -*- coding: utf-8 -*-
"""Rust normalizer portu icin test bataryasi: gercek ciktidan bekleneler.

Kapsam: sayi/ondalik/yuzde/sira, tarih/saat/para/telefon, kisaltma/birim,
unvan, romen rakami, aralik, e-posta/adres, noktalama bicimleri.
BILEREK DISARIDA: Ingilizce/marka kelimeler (cmudict tabanli, v1'de belgeli eksik).
"""
import json
from pathlib import Path

CASES = [
    # sayilar
    "3 elma aldım.",
    "2024 yılında doğdu.",
    "1.250.000 TL tutarındaki ödeme.",
    "Toplam tutar 1.250,75 TL.",
    "%95 başarı oranı.",
    "Sonuç %12,5 arttı.",
    "3. katta oturuyor.",
    "2. Abdülhamid dönemi.",
    "12'nci deneme.",
    "Sıra numaranız 12. Lütfen bekleyin.",
    "5 üzerinden 4 puan.",
    "2026/1847 oranı.",
    "3/4 su bardağı un.",
    "7/24 hizmet veriyoruz.",
    "45/3 numaralı daire.",
    "Telefonu 05321234567.",
    "Bize 0212 555 12 34 numarasından ulaşın.",
    "+90 532 123 45 67 arayın.",
    "19. yüzyıl edebiyatı.",
    "II. Abdülhamid tahttaydı.",
    "Warcraft III oynuyoruz.",
    "Sınavdan 100 üzerinden 085 aldım.",
    "Aradaki fark -5 derece.",
    "3-5 gün içinde gelir.",
    "Sezon 2003-04 çok iyiydi.",
    "Skor 2-1 bitti.",
    # tarih / saat
    "Randevunuz 14 Ekim Salı günü saat 15:30'da.",
    "Kargo 9 Ekim 2026, Perşembe gelecek.",
    "Toplantı 2026-10-09 tarihinde.",
    "Yarın saat 09:45'te İstanbul'dan kalkıyor.",
    "Film 2 saat 15 dakika 30 saniye sürdü.",
    "Sabah 08:00:00'de başlıyoruz.",
    "Pazartesi ile Cuma arası açığız.",
    "Pzt ve Cum günleri kapalıyız.",
    # para
    "₺12,50 ödedim.",
    "100 dolar bozdurdum.",
    "50 avro gönderdi.",
    # kisaltma / unvan / birim
    "Dr. Ayşe Yılmaz ile görüşün.",
    "Prof. Dr. Mehmet Bey geldi.",
    "5 kg un ve 2 L su aldım.",
    "90 km/s hızla gidiyordu.",
    "Oda 30 metrekare.",
    "Kan grubun A Rh+ mi?",
    "Günde 2x1 tablet alın.",
    "Tansiyonum 120/80 mmHg çıktı.",
    "SGK, SMS ve TBMM kısaltmaları.",
    "THY uçağıyla uçtum.",
    "CEO toplantıya katıldı.",
    "K vitamini faydalıdır.",
    "Ahmet K. aradı.",
    "No: 12 D: 5 adresindeyim.",
    "Kimlik No gerekiyor.",
    "Sayfa s. 45'e bakın.",
    "Tel: 0212 555 12 34 / Faks: 0212 555 12 35",
    # e-posta / adres
    "destek@ornek.com adresine yazın.",
    "Sitemiz www.ornek.com adresinde.",
    "https://ornek.com/blog/yazi oku.",
    "Kadıköy/İstanbul hattı.",
    "Atatürk Cad. No: 5",
    # noktalama / sembol
    "Ekmek & su aldım.",
    "%100 yerli üretim!",
    "Sıcaklık 25°C oldu.",
    "A4 kağıt ver.",
    "COVID-19 atlattım.",
    "Saat 14:30'da buluşalım mı?",
    "MERHABA DÜNYA NASILSIN",
    "X-ray çektirdim.",
    "2 GB'lık dosya indirdim.",
    "IBAN TR12 0001 0000 0000 0000 0000 01 nedir?",
]


def main():
    from antalia_mini.text import to_model_text
    out = []
    for text in CASES:
        try:
            spoken = to_model_text(text)
        except Exception as e:  # pragma: no cover
            spoken = f"HATA:{type(e).__name__}"
        out.append({"in": text, "out": spoken})
        print(f"IN : {text!r}\nOUT: {spoken!r}\n", flush=True)
    path = Path("models/antalia_ref/normalizer_cases.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"batarya yazildi: {path.absolute()} ({len(out)} vaka)", flush=True)


if __name__ == "__main__":
    main()
