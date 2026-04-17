# Multivariate Energy Consumption Forecasting (XGBoost + Chronos)

Bu proje, cok degiskenli enerji tuketim tahmini icin iki farkli yaklasimi birlestiren moduler bir Time Series Analysis (TSA) altyapisi sunar:

- XGBoost: hizli, tabular ve aciklanabilir nokta tahmini (baseline)
- Amazon Chronos-T5: transformer tabanli, olasiliksal tahmin ve belirsizlik bantlari

Amaç, CUDA uyumlu bir is akisinda (RTX 4060/5050, 8GB VRAM siniri) dogru, izlenebilir ve uretime hazir bir pipeline olusturmaktir.

## 1) Proje Ozet Mimarisi

Pipeline katmanlari:

1. Konfigurasyon ve runtime optimizasyonu (`src/config.py`)
2. Veri indirme, temizleme, ozellik muhendisligi ve veri formatlama (`src/data_loader.py`)
3. Model katmani:
   - XGBoost (`src/xgboost_model.py`)
   - Chronos (`src/chronos_model.py`)
4. Degerlendirme ve gorsellestirme (`src/evaluator.py`)
5. Operasyonel yardimcilar (logger, klasor setup, VRAM izleme, timer) (`src/utils.py`)
6. Uc uca orkestrasyon (`main.py`)

## 2) Dizin Yapisi

```text
Time Series Analysis/
├── data/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── xgboost_model.py
│   ├── chronos_model.py
│   ├── evaluator.py
│   └── utils.py
├── main.py
├── requirements.txt
└── venv/
```

## 3) Kurulum

### 3.1 Python Ortami

Onerilen surum: Python 3.10+

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.2 GPU/CUDA Notlari

- `requirements.txt` icinde PyTorch CUDA wheel index tanimli.
- XGBoost tarafinda `tree_method=hist` ve `device=cuda` zorunlu.
- Chronos tarafinda `torch_dtype=torch.bfloat16` kullanimi VRAM yonetimi icin kritik.

## 4) Hizli Calistirma

```bash
python main.py
```

Beklenen ana ciktilar:

- `outputs/training.log`
- `outputs/model_comparison.csv`
- `outputs/xgboost/metrics.json`
- `outputs/xgboost/forecast.png`
- `outputs/chronos/metrics.json`
- `outputs/chronos/probabilistic_forecast.png`
- `outputs/hybrid/metrics.json`
- `outputs/hybrid/forecast.png`
- `models/xgboost_tsa_model.json`

## 5) Konfigurasyon Rehberi (`src/config.py`)

Bu modul, tum proje parametrelerini merkezi sekilde yonetir.

### 5.1 Siniflar

#### `PathsConfig`

- `__post_init__()`
  - Amac: Proje klasor hiyerarsisini tek noktadan olusturmak.
  - Urettigi yollar: `data/raw`, `data/processed`, `models/xgboost`, `models/chronos`, `outputs/xgboost`, `outputs/chronos`, `outputs/hybrid`, `outputs/metrics`, `outputs/plots`, `logs`.

- `all_directories() -> tuple[Path, ...]`
  - Amac: Runtime oncesi olusturulmasi gereken tum dizinleri donmek.

#### `RuntimeConfig`

- `device` (property)
  - Amac: CUDA varsa `cuda`, yoksa `cpu` secmek.

- `apply_torch_optimizations()`
  - Amac: PyTorch CUDA allocator ve cuDNN ayarlariyla throughput/memory dengesi kurmak.
  - Uygulananlar:
    - `PYTORCH_CUDA_ALLOC_CONF`
    - `torch.backends.cudnn.benchmark = True`
    - TF32 / matmul precision ayarlari

#### `XGBoostConfig`

- `to_dict() -> dict[str, Any]`
  - Amac: XGBoost egitimi icin parametreleri tek adimda sozluk formatina cevirme.

#### `ChronosConfig`

- Alanlar:
  - `model_id = amazon/chronos-t5-base`
  - `torch_dtype = torch.bfloat16`
  - `context_length = 512`
  - `prediction_length = 24`
  - `batch_size = 8`

#### `PipelineConfig`

- Alanlar:
  - `test_size = 0.2`
  - `sliding_window_size = 512`
  - `max_lag_features = 48`
  - `seed = 42`

#### `AppConfig`

- `initialize()`
  - Amac: Dizinleri olusturmak, runtime optimizasyonlarini uygulamak, seed sabitlemek.

- `init_directories()`
  - Amac: `all_directories()` listesindeki klasorleri fiziksel olarak olusturmak.

### 5.2 Fonksiyon

- `seed_everything(seed: int)`
  - Amac: Python, NumPy ve PyTorch tarafinda deterministik tohumlama.

### 5.3 Module-level Exports

`BASE_DIR`, `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`, `DEVICE`, `TEST_SIZE`, `PREDICTION_LENGTH`, `XGBOOST_PARAMS` gibi sabitler proje genelinde import edilir.

## 6) Veri Katmani Rehberi (`src/data_loader.py`)

`DataLoader`, ingestion -> preprocess -> feature engineering -> split -> scaling adimlarini tek sinifta birlestirir.

### 6.1 Yardimci Tipler

- `Scaler = StandardScaler | MinMaxScaler`

### 6.2 `DataLoader` Metotlari

- `__init__(...)`
  - Amac: Config, hedef kolonu, datetime kolonu ve scaler secimini ayarlamak.
  - Not: Ham veri cache yolu: `DATA_DIR/raw_data.csv`.

- `_build_scaler(kind)`
  - Amac: `standard` veya `minmax` scaler nesnesi uretmek.

- `download_data() -> Path`
  - Amac: Veriyi `Config.DATASET_URL` kaynagindan indirmek.
  - Caching: Dosya varsa tekrar indirmez.

- `load_raw_data() -> pd.DataFrame`
  - Amac: Cache dosyasini DataFrame olarak yuklemek.

- `preprocess_data(data) -> pd.DataFrame`
  - Amac:
    - datetime parse + index set
    - duplicate timestamp temizligi
    - numeric coercion
    - zaman bazli interpolation + ffill/bfill

- `_lag_steps() -> list[int]`
  - Amac: lag adimlarini uretmek (kritik: 1, 2, 24 dahil).

- `_rolling_windows(max_lag) -> list[int]`
  - Amac: rolling pencere adaylarini secmek (3, 6, 12, 24).

- `_add_time_features(frame)`
  - Amac: saat, gun, ay, quarter, day_of_year, weekend bayragi gibi zamansal ozellikler eklemek.

- `_add_lag_features(frame)`
  - Amac: hedef degiskenin gecikmeli kolonlarini eklemek.

- `_add_rolling_features(frame)`
  - Amac: leakage onlemek icin `shift(1)` ile rolling mean/std uretmek.

- `engineer_features(data) -> pd.DataFrame`
  - Amac: tum ozellik muhendisligi adimlarini uygulayip NaN satirlari atmak.

- `split_train_test(data) -> (train_df, test_df)`
  - Amac: zaman sirasini bozmadan kronolojik ayirma.

- `_split_features_target(data) -> (X, y)`
  - Amac: feature matrisi ve hedef vektorunu ayirmak.

- `inverse_transform_target(values) -> np.ndarray`
  - Amac: scale edilmis tahminleri orijinal olcege geri cevirmek.

- `get_xgboost_data() -> (X_train, y_train, X_test, y_test)`
  - Amac: XGBoost icin leakage-safe tabular train/test setlerini hazirlamak.
  - Kritik: scaler `fit` sadece train setinde yapilir.

- `_build_chronos_windows(series) -> (contexts, horizons)`
  - Amac: 1D seri uzerinden Chronos context/horizon pencereleri uretmek.

- `get_chronos_data() -> (train_contexts, train_horizons, test_contexts, test_horizons)`
  - Amac: Chronos icin tensor formatli pencere setleri dondurmek.

## 7) XGBoost Katmani Rehberi (`src/xgboost_model.py`)

`XGBoostModel`, model yasam dongusunu tek sinifta toplar.

### 7.1 `XGBoostModel` Metotlari

- `__init__(config=Config, early_stopping_rounds=50)`
  - Amac: `XGBRegressor` olusturmak ve CUDA parametrelerini enforce etmek.

- `train(X_train, y_train, X_val, y_val)`
  - Amac: validation set ile early stopping destekli egitim.

- `predict(X_test) -> np.ndarray`
  - Amac: modelden 1D NumPy tahmin dondurmek.

- `evaluate(y_true, y_pred) -> dict[str, float]`
  - Amac: MSE, RMSE, MAE hesaplamak.

- `save_model(model_name="xgboost_tsa_model.json") -> Path`
  - Amac: modeli JSON formatinda kaydetmek.

- `load_model(model_path) -> xgb.XGBRegressor`
  - Amac: kayitli modeli geri yuklemek.

- `get_feature_importance(feature_names) -> pd.DataFrame`
  - Amac: `weight` ve `gain` skorlarini orijinal kolon isimleriyle eslemek ve siralamak.

## 8) Chronos Katmani Rehberi (`src/chronos_model.py`)

`ChronosModel`, olasiliksal forecast ve belirsizlik hesaplamalarina odaklanir.

### 8.1 Tipler

- `ContextInput = Union[torch.Tensor, list[torch.Tensor], np.ndarray]`
- `ForecastOutput = tuple[np.ndarray, np.ndarray, np.ndarray]`

### 8.2 `ChronosModel` Metotlari

- `__init__(config=Config, num_samples=20, confidence_level=0.9)`
  - Amac: Chronos pipeline'i VRAM dostu parametrelerle hazirlamak.

- `_initialize_pipeline()`
  - Amac: `from_pretrained` ile modeli yuklemek.
  - Kritik parametreler:
    - `model_id = Config.CHRONOS_MODEL_ID`
    - `device_map = Config.DEVICE`
    - `torch_dtype = torch.bfloat16`

- `_to_1d_tensor(series)`
  - Amac: giris serisini 1D float tensor formatina normalize etmek.

- `_prepare_context(context_series)`
  - Amac: tekli veya batched context girdisini pipeline uyumlu hale getirmek.

- `_ensure_sample_axis(samples)`
  - Amac: tahmin cikisini `[batch, num_samples, horizon]` formatina normalize etmek.

- `_compute_prediction_bands(samples)`
  - Amac:
    - medyan tahmin
    - alt/ust guven bandi (`torch.quantile`)
    - NumPy formatina donus

- `predict(context_series) -> ForecastOutput`
  - Amac: olasiliksal tahmin almak ve `(median, low, high)` dondurmek.

## 9) Degerlendirme Katmani Rehberi (`src/evaluator.py`)

`ModelEvaluator`, hem metrik hem de grafik uretimini tek noktadan yonetir.

### 9.1 `ModelEvaluator` Metotlari

- `__init__(config=Config)`
  - Amac: output klasorunu garanti etmek ve profesyonel plot stilini ayarlamak.

- `_to_numpy_1d(values)`
  - Amac: girisleri 1D NumPy dizisine cevirmek.

- `_validate_equal_length(*arrays)`
  - Amac: tum dizi uzunluklarini dogrulamak.

- `_safe_mape(y_true, y_pred)`
  - Amac: sifira bolme riskini epsilon ile yoneterek MAPE hesaplamak.

- `_serialize_metrics(metrics, model_name)`
  - Amac: metrikleri JSON dosyasina yazmak.

- `calculate_metrics(y_true, y_pred, model_name)`
  - Amac: MSE, RMSE, MAE, MAPE hesaplayip dondurmek ve kaydetmek.

- `plot_predictions(y_true, y_pred, timestamps, model_name)`
  - Amac: nokta tahmini grafigi cizmek ve kaydetmek (`dpi=300`).

- `plot_probabilistic_predictions(y_true, median_pred, low_band, high_band, timestamps, model_name)`
  - Amac: medyan tahmin + guven bandi (`fill_between`) grafigi uretmek.

- `compare_models(metrics_dict)`
  - Amac: modelleri DataFrame olarak yan yana karsilastirmak, konsola yazdirmak, CSV kaydetmek.

## 10) Operasyonel Yardimcilar Rehberi (`src/utils.py`)

`utils.py`, calisma ortami operasyonlarini merkeziler.

### 10.1 Fonksiyonlar

- `setup_directories() -> tuple[Path, Path, Path]`
  - Amac: `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR` ile birlikte `outputs/xgboost`, `outputs/chronos`, `outputs/hybrid` klasorlerini olusturmak.

- `_has_stdout_handler(logger)`
  - Amac: duplicate stdout handler kontrolu.

- `_has_file_handler(logger, file_path)`
  - Amac: duplicate file handler kontrolu.

- `get_logger(logger_name) -> logging.Logger`
  - Amac: stdout + `outputs/training.log` hedeflerine ayni anda log yazan logger olusturmak.

- `log_gpu_memory(logger)`
  - Amac: CUDA aciksa allocated/reserved/total VRAM degerlerini GB cinsinden loglamak.
  - Uyari kosulu: reserved VRAM 7.2GB ve uzeri.

- `_resolve_decorator_logger(func, args, kwargs)`
  - Amac: decorator icin logger secimini non-intrusive sekilde yapmak.

- `time_it(func)`
  - Amac: fonksiyon calisma suresini olcup logger uzerinden yazmak.

## 11) Orkestrasyon Akisi (`main.py`)

`main.py`, pipeline'i try-except-finally ile guvenli sekilde calistirir.

### 11.1 Yardimci Fonksiyonlar

- `_split_train_validation(x_train, y_train, validation_ratio=0.2)`
  - Amac: early stopping icin train/validation ayrimi.

- `_first_horizon_step(values)`
  - Amac: multi-horizon cikislardan ilk adim tahmini secmek.

- `_compute_chronos_timestamps(processed_df, data_loader, expected_length)`
  - Amac: Chronos one-step tahminleriyle hizali timestamp dizisi uretmek.

- `_align_prediction_frames(...)`
  - Amac: XGBoost ve Chronos ciktilarini ortak timestamp uzayinda birlestirmek.
  - Ek islev: kesismeyen durumlarda tail-based fallback.

### 11.2 `main()` Uc Uca Siralama

1. Dizin setup + logger baslatma
2. Veri ingestion + preprocessing
3. XGBoost data hazirlama ve egitim
4. XGBoost model kaydi ve tahmin
5. GPU cache temizleme (`gc.collect`, `torch.cuda.empty_cache`) + VRAM log
6. Chronos olasiliksal tahmin
7. Hibrit tahmin (`0.5 * xgb + 0.5 * chronos`)
8. Metrik hesaplama ve karsilastirma
9. Nokta ve olasiliksal grafiklerin kaydi
10. Hata durumunda exception loglama, finally blokta toplam sure loglama

## 12) Cikti Dosyalari ve Anlamlari

- `xgboost/metrics.json`, `chronos/metrics.json`, `hybrid/metrics.json`: model bazli metrikler (`mse`, `rmse`, `mae`, `mape`, `r2`)
- `model_comparison.csv`: tum modellerin yan yana metrik tablosu
- `xgboost/forecast.png`, `hybrid/forecast.png`: nokta tahmini grafikleri
- `chronos/probabilistic_forecast.png`: medyan + guven bandi grafigi
- `training.log`: pipeline adim loglari, hatalar, sure ve VRAM kayitlari

## 13) Konfigurasyon ve Veri Notlari

Mevcut kod tabani, `src.config` icerisinde hem module-level sabitlerle hem de fallback `Config` class yaklasimiyla calisacak sekilde yazilmistir.

Dikkat edilmesi gereken alanlar:

- DataLoader, veri kaynaginda asagidaki kolonlari bekler:
  - datetime kolonu: varsayilan `timestamp`
  - hedef kolon: varsayilan `target`
- Farkli kolon adlari icin `DataLoader(target_column=..., datetime_column=...)` parametreleri kullanilabilir.
- `Config.DATASET_URL` tanimi yapilmadiysa `download_data()` hata verir.

## 14) Sorun Giderme

### `Config.DATASET_URL is empty`

- Cozum: `src/config.py` icinde dataset URL tanimlayin veya DataLoader akisini local dosya ile guncelleyin.

### `Import could not be resolved` (IDE warning)

- Kodda pyright/type ignore notlari bulunur; runtime icin paketin kurulu olmasi gerekir.
- Cozum: `pip install -r requirements.txt`

### CUDA OOM riski

- Chronos `num_samples` degerini dusurun.
- Context uzunlugunu azaltin.
- Egitim/tahmin oncesi `torch.cuda.empty_cache()` cagrilarini koruyun.
- `training.log` icerisindeki VRAM satirlarini takip edin.

## 15) Gelistirme Onerileri

- `Config` icin tekil, explicit bir class kontrati tanimlayarak fallback ihtiyacini azaltma
- Unit testler:
  - Data leakage testleri
  - Timestamp hizalama testleri
  - Metric regression testleri
- CI pipeline:
  - lint + type-check + smoke run
- Model registry entegrasyonu (MLflow vb.)

---

Bu README, kod tabanindaki mevcut modul/fonksiyon imzalarina gore hazirlanmistir ve proje buyudukce referans dokuman olarak guncellenmelidir.
