# Анализ кандидатов на детекцию людей

## Условия

- **Окружение**: CPU (без GPU), Python 3.13, PyTorch 2.12+cpu
- **Требование**: ≥5 FPS в Colab (с GPU), лучше чем оригинальный DeepSORT на каждом видео
- **Нужно**: минимум 3 модели с переключением перед запуском
- **Бонус**: модели из более чем 2 разных источников — +1 балл

---

## 1. YOLOv8 / YOLO11 (Ultralytics)

**Источник**: [Ultralytics](https://github.com/ultralytics/ultralytics)

| Модель | mAP50-95 | CPU ONNX (ms) | T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|--------|----------|---------------|-------------------|------------|-----------|
| YOLOv8n | 37.3 | 80.4 | 1.47 | 3.2 | 8.7 |
| YOLOv8s | 44.9 | 128.4 | 2.66 | 11.2 | 28.6 |
| YOLOv8m | 50.2 | 234.7 | 5.86 | 25.9 | 78.9 |
| YOLO11n | 39.5 | 56.1 | 1.5 | 2.6 | 6.5 |
| YOLO11s | 47.0 | 90.0 | 2.5 | 9.4 | 21.5 |
| YOLO11m | 51.5 | 183.2 | 4.7 | 20.1 | 68.0 |

**Плюсы**:
- Лучшее соотношение скорость/точность на CPU
- Простой Python API (`ultralytics` пакет)
- Поддержка сегментации (YOLOv8-seg / YOLO11-seg) — для дополнительной задачи
- COCO-предобученные, класс "person" (ID 0)
- Лёгкая интеграция: `model = YOLO("yolo11n.pt"); results = model(frame)`

**Минусы**:
- Требует NMS (постобработка)
- На CPU только nano/small модели достижимы для ≥5 FPS

**Оценка FPS на CPU** (оценочно по ONNX-замерам):
- YOLO11n: ~18 FPS (56ms/frame) ✅
- YOLO11s: ~11 FPS (90ms/frame) ✅
- YOLOv8n: ~12 FPS (80ms/frame) ✅
- YOLOv8s: ~8 FPS (128ms/frame) ✅

---

## 2. RT-DETR (Ultralytics / HuggingFace)

**Источник**: [Ultralytics](https://docs.ultralytics.com/models/rtdetr/) + [HuggingFace](https://huggingface.co/PekingU/rtdetr_r50vd)

| Модель | mAP50-95 | T4 TensorRT (FPS) | Params (M) | FLOPs (G) |
|--------|----------|-------------------|------------|-----------|
| RT-DETR-R18 | 46.5 | 217 | 20 | 60 |
| RT-DETR-R34 | 48.9 | 161 | 31 | 92 |
| RT-DETR-R50 | 53.1 | 108 | 42 | 136 |
| RT-DETR-L (HGNetv2) | 53.0 | 114 | 32 | 110 |
| RT-DETR-X (HGNetv2) | 54.8 | 74 | 67 | 234 |

**Плюсы**:
- NMS-free (end-to-end, без постобработки)
- Лучше точность чем YOLO при сравнимом размере
- Гибкая настройка скорости (уменьшение числа decoder layers без переобучения)
- Доступен через два источника: Ultralytics и HuggingFace transformers

**Минусы**:
- На CPU значительно медленнее YOLO (тяжёлый transformer decoder)
- Большие модели — высокая потребность памяти
- Нет встроенной сегментации (только детекция)

**Оценка FPS на CPU**: ~1-3 FPS (тяжёлый transformer, вероятно слишком медленно для CPU)
**На GPU в Colab**: RT-DETR-R18 ~217 FPS, RT-DETR-L ~114 FPS ✅

---

## 3. DETR (HuggingFace)

**Источник**: [HuggingFace transformers](https://huggingface.co/docs/transformers/model_doc/detr)

| Модель | mAP50-95 | Платформа | Особенности |
|--------|----------|-----------|-------------|
| DETR-R50 | 42.0 | GPU | 100 queries, ~28 FPS на T4 |
| DETR-R101 | 43.5 | GPU | Тяжелее, медленнее |

**Плюсы**:
- End-to-end, NMS-free
- Хорошая глобальная контекстная обработка (transformer)
- Поддержка паноптической сегментации (DETR-panoptic)
- Легко использовать через HuggingFace transformers

**Минусы**:
- Оригинальный DETR значительно медленнее RT-DETR
- По исследованию: "DETR was found to be too computationally expensive" для real-time
- На CPU неприменим для real-time (<1 FPS)
- Меньшая точность чем RT-DETR при большем времени инференса

**Оценка FPS на CPU**: <1 FPS ❌
**На GPU в Colab**: ~28 FPS (R50) — приемлемо

---

## 4. Faster R-CNN / Cascade R-CNN (mmdetection)

**Источник**: [mmdetection](https://github.com/open-mmlab/mmdetection)

| Модель | Backbone | box AP | Inf FPS (GPU) | Person AP |
|--------|----------|--------|---------------|-----------|
| Faster R-CNN | R-50-FPN | 37.4 | 21.4 | 55.8 (person-specific) |
| Faster R-CNN | R-101-FPN | 39.4 | 15.6 | — |
| Cascade R-CNN | R-50-FPN | 40.4 | ~20 | — |
| Cascade R-CNN | R-101-FPN | 42.3 | ~14 | — |

**Плюсы**:
- Person-specific модель: Faster R-CNN R-50-FPN, AP=55.8 на person
- Двухстадийный детектор — высокая точность локализации
- Разный архитектурный подход (region-based, не anchor-free/transformer)
- Разнообразие источников (mmdetection — отдельный репозиторий)

**Минусы**:
- Медленный на CPU (двухстадийный, RPN + RoIHead)
- Требует mmdetection framework (тяжёлые зависимости, конфликт версий)
- На GPU: 21 FPS — ниже YOLO и RT-DETR
- Сложная установка mmdetection (mmcv, mmengine)

**Оценка FPS на CPU**: ~1-2 FPS ❌
**На GPU в Colab**: ~21 FPS (R-50) ✅

---

## Сводная таблица

| Модель | Источник | mAP | CPU FPS | GPU FPS | Сегментация | Сложность интеграции |
|--------|----------|-----|---------|---------|-------------|---------------------|
| YOLO11n | Ultralytics | 39.5 | ~18 ✅ | ~670 | ✅ (seg) | Низкая |
| YOLO11s | Ultralytics | 47.0 | ~11 ✅ | ~400 | ✅ (seg) | Низкая |
| YOLOv8n | Ultralytics | 37.3 | ~12 ✅ | ~680 | ✅ (seg) | Низкая |
| YOLOv8s | Ultralytics | 44.9 | ~8 ✅ | ~376 | ✅ (seg) | Низкая |
| RT-DETR-L | Ultralytics + HF | 53.0 | ~1-3 ❌ | 114 ✅ | ❌ | Средняя |
| RT-DETR-R18 | Ultralytics + HF | 46.5 | ~2-4 ❌ | 217 ✅ | ❌ | Средняя |
| DETR-R50 | HuggingFace | 42.0 | <1 ❌ | ~28 ✅ | ✅ (panoptic) | Средняя |
| Faster R-CNN R-50 | mmdetection | 37.4 | ~1-2 ❌ | 21 ✅ | ❌ | Высокая |
| Cascade R-CNN R-50 | mmdetection | 40.4 | ~1-2 ❌ | ~20 ✅ | ❌ | Высокая |

---

## Рекомендации

### Для интеграции (минимум 3 модели):

1. **YOLO11s (Ultralytics)** — лучший баланс точности и скорости на CPU. mAP 47.0, ~11 FPS на CPU. Поддержка сегментации.
2. **YOLOv8n (Ultralytics)** — самый быстрый вариант, mAP 37.3, ~12 FPS на CPU. Другая архитектура от YOLO11.
3. **YOLO11n (Ultralytics)** — лёгкий вариант, mAP 39.5, ~18 FPS на CPU.

### Для сегментации (+1 балл):
- YOLO11-seg или YOLOv8-seg — нативная поддержка instance segmentation

### Для real-time на CPU (≥5 FPS):
- YOLO11n (~18 FPS) ✅
- YOLO11s (~11 FPS) ✅
- YOLOv8n (~12 FPS) ✅
- YOLOv8s (~8 FPS) ✅

### Итоговый выбор для реализации:

| # | Модель | Источник | Назначение |
|---|--------|----------|------------|
| 1 | YOLO11s | Ultralytics | Основная модель (точность + скорость) |
| 2 | YOLOv8n | Ultralytics | Быстрый вариант (real-time на CPU) |
| 3 | YOLO11n | Ultralytics | Лёгкий вариант (максимальный FPS) |
| 4 (доп.) | YOLO11-seg | Ultralytics | Сегментация для доп. задачи |
