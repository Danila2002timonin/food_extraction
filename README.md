# Food Extractor


## Структура проекта

```
food_extraction/
├── __init__.py                    # Инициализация пакета
├── main.py                        # Основная логика запуска
├── run.py                         # Файл запуска из корня проекта
├── detector/                      # Модуль детектора объектов
│   ├── __init__.py
│   ├── hugging_face_detector.py   # Основной класс детектора
│   ├── object_detection.py        # Функции обнаружения объектов
│   └── visualization.py           # Функции визуализации
├── ai_services/                   # Модуль для работы с AI сервисами 
│   ├── __init__.py
│   ├── gpt_service.py             # Интеграция с OpenAI GPT-4o
│   ├── semantic_matching.py       # Функции семантического сопоставления
│   ├── stability_ai.py            # Интеграция со Stability AI
│   └── data/                      # Данные конфигурации в JSON формате
│       ├── food_container_mapping.json    # Сопоставления типов блюд и контейнеров
│       ├── food_items.json               # Список пищевых элементов
│       ├── tableware_items.json          # Список столовой посуды
│       ├── forbidden_matches.json        # Запрещенные сочетания блюд и посуды
│       └── multi_piece_dish_items.json   # Блюда из нескольких частей
└── utils/                         # Вспомогательные утилиты
    ├── __init__.py
    └── image_utils.py             # Функции для работы с изображениями
```

## Установка

Создайте виртуальное окружение и установите зависимости с помощью uv:

```bash
# Установите uv, если он еще не установлен
pip install uv

# Создайте виртуальное окружение и активируйте его
uv venv
source .venv/bin/activate   # для Windows: .venv\Scripts\activate

# Установите зависимости
uv pip install -r requirements.txt
```

## Использование

### Базовое использование

Запустите скрипт с помощью командной строки:

```bash
# Извлечение объекта по текстовому описанию
uv run run.py --image path/to/image.jpg --prompt "pizza"

# Автоматическое определение основного блюда
uv run run.py --image path/to/image.jpg --auto
```

### Дополнительные опции

```bash
# Удаление фона изображения
uv run run.py --image path/to/image.jpg --prompt "steak" --remove-bg

# Расширение изображения (требуется API Stability AI)
uv run run.py --image path/to/image.jpg --prompt "cake" --extend --extend-left 100 --extend-right 100

# Полный набор опций
uv run run.py --image path/to/image.jpg --prompt "salad" --output result.png --model facebook/detr-resnet-101 --threshold 0.2 --debug --remove-bg --extend --extend-left 50 --extend-right 50 --extend-up 30 --extend-down 30
```

## Параметры командной строки

- `--image` - Путь к входному изображению
- `--prompt` - Текстовое описание объекта для извлечения
- `--auto` - Автоматическое определение основного блюда
- `--output` - Путь для сохранения выходного изображения (по умолчанию: extracted_object.png)
- `--api_key` - Ключ API OpenAI (если не установлен в переменных окружения)
- `--model` - Название модели Hugging Face для обнаружения объектов
- `--threshold` - Порог уверенности для обнаружения (0.0-1.0)
- `--debug` - Включение режима отладки с визуализацией
- `--remove-bg` - Использование Stability AI для удаления фона
- `--extend` - Использование Stability AI для расширения изображения
- `--extend-left` - Пиксели для расширения с левой стороны
- `--extend-right` - Пиксели для расширения с правой стороны
- `--extend-up` - Пиксели для расширения сверху
- `--extend-down` - Пиксели для расширения снизу
- `--stability-api-key` - Ключ API Stability AI (если не установлен в переменных окружения)


## Переменные окружения

Следующие переменные окружения могут быть использованы вместо передачи их как аргументов:

- `OPENAI_API_KEY` - Ключ API OpenAI
- `STABLE_DIFFUSION_API_KEY` - Ключ API Stability AI

Их можно установить в файле `.env` в корне проекта. 