# All-in-One Pack: ADetailer + 2x ControlNet

Професійний інструмент для покращення портретів з використанням Stable Diffusion WebUI, ADetailer та двох ControlNet моделей.

## Особливості

- **A-Pass**: Автоматичне створення масок обличчя та контурних карт
- **B-Pass**: Покращення портретів з використанням ControlNet та ADetailer
- **Два ControlNet**: SoftEdge SDXL для форми та Canny/Lineart для контурів
- **ADetailer**: Автоматичне покращення деталей обличчя
- **Batch Processing**: Обробка множини зображень

## Структура проєкту

```
.
├── README.md
├── .gitignore
└── runpod-pack/
    ├── bootstrap.sh          # Автоматична установка та налаштування
    ├── models_auto.py        # Автоматичне завантаження моделей
    ├── README.md             # Інструкції по запуску
    └── preset-contour/
        ├── batch.py          # Batch обробка зображень
        ├── config.yaml       # Конфігурація параметрів
        ├── requirements.txt   # Python залежності
        ├── run_a_pass.py     # A-Pass: створення масок та контурів
        └── run_b_pass.py     # B-Pass: покращення портретів
```

## Швидкий старт

1. Розархівуйте проєкт в `/workspace`
2. Запустіть: `cd runpod-pack && bash bootstrap.sh`
3. Помістіть зображення в `preset-contour/input/`

## Налаштування

Відредагуйте `preset-contour/config.yaml`:
- `controlnet_weight` (0.55–0.65)
- `controlnet2_weight` (0.35–0.55)  
- `ad_denoise` (0.14–0.20)
- `steps` (26–32)
- `cfg` (4.0–4.8)

## Залежності

- Python 3.8+
- Stable Diffusion WebUI
- ControlNet extension
- ADetailer extension
- Pillow, PyYAML, requests

## Ліцензія

MIT License
