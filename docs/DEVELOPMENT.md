# Разработка EchoLens

Сборка, CI, релизы, генерация ассетов и моделей. Для архитектуры —
docs/ARCHITECTURE.md, для звукового слоя — docs/SOUND_DESIGN.md.

## Сборка

```bash
git clone https://github.com/FreedoomForm/echolens.git
cd echolens
./gradlew assembleRelease          # APK: app/build/outputs/apk/release/
./gradlew testReleaseUnitTest      # JVM-тесты (Angles, LabelTable, RuText)
```

Требуется JDK 17+ и Android SDK (platform 34). `local.properties` с путём
`sdk.dir` — при необходимости.

## CI и релизы

`.github/workflows/android.yml`: на каждый push → unit-тесты + сборка
release-APK; на тег `v*` → то же + автоматический GitHub Release с APK
`Echolens-<тег>.apk` (generate_release_notes).

Порядок релиза:
```bash
# 1. Поднять versionCode/versionName в app/build.gradle.kts
# 2. Обновить README «Дорожная карта» (отметить сделанное)
git commit -m "v1.8.0: <что вошло>"
git tag v1.8.0 && git push origin main v1.8.0
```

Подпись: `signing/ci-release.jks` — публичный CI-ключ для sideload-APK
(намеренно в репозитории; замените на свой для продакшена).

## Карта репозитория: что здесь лежит и зачем

| Путь | Зачем |
|---|---|
| `app/src/main/java/com/echolens/app/` | Код приложения: `vision/` (камера→детекции), `core/` (домен и таблицы), `head/` (гиро), `audio/` (движок звука), `audio/neural/` (опциональная генерация), `service/`, `MainActivity` |
| `app/src/main/assets/*.onnx` | Детекторы YOLOE/YOLO26 + глубина — встроены, приложение офлайн с первого запуска |
| `app/src/main/assets/sounds/` | Библиотека записей FSD50K + `manifest.json` + `ATTRIBUTIONS.md` (по-файловая атрибуция) |
| `app/src/test/` | JVM-тесты чистых модулей (без Android-зависимостей) |
| `tools/` | Скрипты генерации ассетов: звуковая библиотека из FSD50K, квантизация нейро-моделей, обучение классификатора материалов |
| `docs/` | Документация «зачем»: архитектура, исследования, звук, разработка |
| `THIRD_PARTY_LICENSES.md` | Лицензии всех сторонних компонентов (модели, датасет, библиотеки) |
| `signing/` | CI-keystore (публичный тестовый) |

## Скрипты tools/

Все скрипты конфигурируются переменными окружения; по умолчанию данные
ожидаются в `../data` относительно репозитория:

| Скрипт | Что делает | Входы → Выходы |
|---|---|---|
| `fsd_build_candidates.py` | Отбор кандидатов: маппинг 426 классов + 15 материалов → 200 меток FSD50K, фильтр лицензий | `data/{dev,eval}.csv`, `*_clips_info.json`, `pp_pnp.json` → `data/candidates.json` |
| `fsd_pipeline.py` | Скачивание → гейты качества (SNR/клиппинг/тишина) → лупы → нормализация → OGG; стадия `select` раскладывает в assets и пишет `manifest.json` + `ATTRIBUTIONS.md` | `data/candidates.json` → `assets/sounds/` |
| `fsd_events.py` | One-shot акценты 0.9 с по материалам | FSD50K → `assets/sounds/events/` |
| `quantize_conditioner.py` | Weight-only int8 квантизация T5-conditioner (нейро-опция) | tflite → tflite int8 |
| `train_minc_classifier.py` | Обучение MINC-23 MobileNetV3-Small → uint8 TFLite | MINC-2500 → `minc_classifier.tflite` |
| `make_prompt_tokens.py` | Генерация `MaterialPrompts.kt` (T5-токены 15 материалов) | tokenizer → Kotlin-файл |

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r tools/requirements.txt   # numpy, soundfile; нужен также ffmpeg в PATH
export ECHOLENS_REPO="$PWD" ECHOLENS_DATA="$PWD/../data"
cd tools
python3 fsd_build_candidates.py
python3 fsd_pipeline.py download
python3 fsd_events.py
python3 fsd_pipeline.py select
```

## Таблица классов (LabelTable)

`core/LabelTable.kt` — 426 параллельных массивов (EN-метка = индекс класса
yoloe26s.onnx; RU-имя; материал; типовая высота; важность). Таблица
экспортирована из словаря YOLOE один раз и аннотирована вручную. Правила
изменения:

1. Порядок EN-меток менять нельзя — он запечён в ONNX.
2. Первые 80 записей — COCO-80 в порядке Ultralytics (общие с fallback-детектором).
3. RU-имена согласованы с `core/TextureLibrary.kt` (словарь fallback-моделей).
4. Изменение таблицы покрывается `LabelTableTest` — запустите тесты.

## Добавление звукового класса

1. Проверить, есть ли класс в `LabelTable` (если нет — добавить с материалом).
2. Собрать записи в `assets/sounds/class/<en_name>/` (5–10 вариантов) или
   расширить `fsd_build_candidates.py` маппингом и прогнать пайплайн.
3. Обновить `manifest.json` (ключи = EN-метки с подчёркиваниями).
4. `assembleRelease` + тесты; проверить звучание нового класса в «Проверке звука».

## Известные техдолги (осознанные)

- Строки статусов нейро-пайплайна — RU hardcoded (опциональный тумблер,
  основная аудитория RU; миграция при появлении второго языка).
- `AudioSceneService` объединяет камеру/TTS/вибро — разделить после
  инструментальных тестов.
- Инструментальные тесты (эмулятор) — не заведены; JVM-покрытие растёт
  (`app/src/test`).
