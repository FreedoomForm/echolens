# tools/ — генерация ассетов EchoLens

Скрипты, которые один раз собирают то, что лежит в `assets/`: звуковую
библиотеку FSD50K, квантизованные нейро-модели, классификатор материалов.

Подробные инструкции: [../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md).
Кратко:

```bash
pip install -r requirements.txt   # numpy, soundfile (+ ffmpeg в PATH)
export ECHOLENS_REPO=/path/to/echolens
export ECHOLENS_DATA=/path/to/data
python3 fsd_build_candidates.py   # кандидаты клипов по классам/материалам
python3 fsd_pipeline.py download  # скачать + прогнать гейты качества
python3 fsd_events.py             # one-shot акценты
python3 fsd_pipeline.py select    # разложить в assets + manifest + атрибуция
```

| Файл | Назначение |
|---|---|
| `fsd_build_candidates.py` | Маппинг 426 классов + 15 материалов → метки FSD50K; фильтр лицензий CC0/CC-BY |
| `fsd_pipeline.py` | Пайплайн: загрузка → SNR-гейты → лупы → нормализация → OGG; select пишет manifest.json + ATTRIBUTIONS.md |
| `fsd_events.py` | Короткие акценты «объект появился» (0.9 с) |
| `quantize_conditioner.py` | int8-квантизация T5-conditioner (нейро-опция) |
| `train_minc_classifier.py` | MobileNetV3-Small на MINC-2500 → TFLite uint8 |
| `make_prompt_tokens.py` | Генерация MaterialPrompts.kt (T5-токены материалов) |
| `litert_models.md` | Сигнатуры LiteRT-моделей и порядок провижининга |
