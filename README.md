# AzuraForge Core Engine ⚙️

**AzuraForge Core**, AzuraForge ekosisteminin kalbidir. Bu kütüphane, "The AzuraForge Way" felsefesinin en saf halini temsil eder: **Sıfırdan İnşa ve Derin Anlayış.**

## 🎯 Ana Sorumluluklar

*   **`Tensor` Nesnesi:** NumPy (ve opsiyonel olarak CuPy) üzerinde çalışan, dinamik hesaplama grafiği oluşturabilen çok boyutlu bir dizi nesnesi sağlar.
*   **Otomatik Türev (Geri Yayılım):** `Tensor` nesneleri üzerinde yapılan tüm matematiksel işlemlerin (toplama, çarpma, matris çarpımı vb.) gradyanlarını otomatik olarak hesaplayan bir geri yayılım (backpropagation) motoru içerir.
*   **Temel Aktivasyon Fonksiyonları:** `ReLU`, `Sigmoid`, `Tanh` gibi temel sinir ağı fonksiyonlarının hem ileri (forward) hem de geri (backward) geçişlerini implemente eder.
*   **Donanım Soyutlaması:** `AZURAFORGE_DEVICE` ortam değişkenine göre işlemlerin CPU (NumPy) veya GPU (CuPy) üzerinde çalışmasını sağlar.

Bu kütüphane, dış dünyaya minimum bağımlılıkla, temel prensipleri anlaşılarak inşa edilmiştir ve platformdaki tüm AI işlemlerinin temelini oluşturur.

---

## 🏛️ Ekosistemdeki Yeri

Bu motor, AzuraForge ekosisteminin en alt katmanıdır ve `azuraforge-learner` tarafından kullanılır. Projenin genel mimarisini, vizyonunu ve geliştirme rehberini anlamak için lütfen ana **[AzuraForge Platform Dokümantasyonuna](https://github.com/AzuraForge/platform/tree/main/docs)** başvurun.

---

## 🛠️ Geliştirme ve Test

Bu kütüphane, genellikle diğer AzuraForge servisleri tarafından bir bağımlılık olarak kullanılır. Yerel geliştirme ortamı kurulumu için ana platformun **[Geliştirme Rehberi](https://github.com/AzuraForge/platform/blob/main/docs/DEVELOPMENT_GUIDE.md)**'ni takip edin.

Birim testlerini çalıştırmak (`LSTM` katmanının gradyan doğruluğunu test etmek vb.) için bu repo dizinindeyken `pytest` komutunu kullanabilirsiniz.