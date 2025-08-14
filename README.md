# All-in-One Pack: ADetailer + 2x ControlNet (Enhanced v2.0)

Професійний інструмент для покращення портретів з використанням Stable Diffusion WebUI, ADetailer та двох ControlNet моделей. **Версія 2.0** включає покращену обробку помилок, паралельну обробку, валідацію конфігурації та графічний інтерфейс.

## Особливості

- **A-Pass**: Автоматичне створення масок обличчя та контурних карт
- **B-Pass**: Покращення портретів з використанням ControlNet та ADetailer
- **Два ControlNet**: SoftEdge SDXL для форми та Canny/Lineart для контурів
- **ADetailer**: Автоматичне покращення деталей обличчя
- **Batch Processing**: Обробка множини зображень з прогрес-барами
- **Quality Control**: Автоматична перевірка якості результатів

## Структура проєкту

```
.
├── README.md
├── .gitignore
└── runpod-pack/
    ├── bootstrap.sh              # Автоматична установка та налаштування
    ├── models_auto.py            # Автоматичне завантаження моделей
    ├── README.md                 # Інструкції по запуску
    └── preset-contour/
        ├── batch.py              # 🆕 Покращений batch обробка з прогрес-барами
        ├── config.yaml           # 🆕 Розширена конфігурація
        ├── config_validator.py   # 🆕 Валідатор конфігурації
        ├── config_gui.py         # 🆕 Графічний редактор конфігурації
        ├── requirements.txt      # 🆕 Оновлені залежності
        ├── run_a_pass.py         # A-Pass: створення масок та контурів
        └── run_b_pass.py         # B-Pass: покращення портретів
```

## Швидкий старт

1. **Розархівуйте проєкт** в `/workspace`
2. **Запустіть установку**: `cd runpod-pack && bash bootstrap.sh`
3. **Помістіть зображення** в `preset-contour/input/`
4. **Запустіть обробку**: `python batch.py --workers 2 --log-level INFO`

## 🆕 Нові команди

### Batch обробка з покращеними опціями
```bash
# Базова обробка
python batch.py

# Паралельна обробка з 4 робочими процесами
python batch.py --workers 4

# Детальне логування
python batch.py --log-level DEBUG

# Валідація конфігурації
python config_validator.py config.yaml

# Графічний редактор конфігурації
python config_gui.py
```

### Покращені параметри конфігурації
```yaml
general:
  max_retries: 3          # Кількість спроб при помилці
  retry_delay: 2          # Затримка між спробами (секунди)
  timeout_seconds: 300    # Таймаут для операцій

quality:
  enable_quality_check: true
  min_face_detection_confidence: 0.5
  max_blur_threshold: 100
  min_sharpness_score: 0.6

logging:
  level: "INFO"
  file_logging: true
  log_file: "adetailer_batch.log"
```

## Налаштування

### Основні параметри (config.yaml)
- `controlnet_weight` (0.55–0.65)
- `controlnet2_weight` (0.35–0.55)  
- `ad_denoise` (0.14–0.20)
- `steps` (26–32)
- `cfg` (4.0–4.8)

### 🆕 Параметри якості
- `min_face_detection_confidence`: Мінімальна впевненість детекції обличчя
- `max_blur_threshold`: Максимальний поріг розмиття
- `min_sharpness_score`: Мінімальний бал різкості

### 🆕 Параметри логування
- `level`: Рівень логування (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `file_logging`: Увімкнення/вимкнення логування у файл
- `log_file`: Шлях до файлу логування

## Залежності

- **Python 3.8+**
- **Stable Diffusion WebUI**
- **ControlNet extension**
- **ADetailer extension**
- **Pillow, PyYAML, requests**
- **🆕 tqdm** (прогрес-бари)
- **🆕 typing-extensions** (типізація)

## 🆕 Валідація конфігурації

Автоматична перевірка конфігурації з детальними повідомленнями про помилки:

```bash
python config_validator.py config.yaml
```

## 🆕 Графічний інтерфейс

Зручний редактор конфігурації з вкладками:

```bash
python config_gui.py
```

**Вкладки:**
- **General**: Backend, endpoint, retry налаштування
- **B-Pass**: Prompts, параметри обробки
- **I/O**: Директорії введення/виведення
- **Quality**: Налаштування якості
- **Logging**: Параметри логування

## Приклади використання

### Обробка одного зображення
```bash
python run_a_pass.py --input photo.jpg --workdir work/
python run_b_pass.py --workdir work/ --output output/
```

### Batch обробка з прогрес-баром
```bash
python batch.py --workers 2 --log-level INFO
```

### Валідація перед запуском
```bash
python config_validator.py config.yaml
python batch.py --config config.yaml
```

## 🆕 Обробка помилок

- **Автоматичні retry** з експоненціальним backoff
- **Детальне логування** помилок та попереджень
- **Graceful degradation** при часткових збоях
- **Статистика успішності** обробки

## Ліцензія

MIT License

## 🆕 Changelog v2.0

- ✅ Покращене логування з прогрес-барами
- ✅ Retry механізм для надійності
- ✅ Паралельна обробка зображень
- ✅ Валідація конфігурації
- ✅ Графічний інтерфейс налаштувань
- ✅ Розширена конфігурація
- ✅ Автоматичне тестування якості
- ✅ Покращена обробка помилок
