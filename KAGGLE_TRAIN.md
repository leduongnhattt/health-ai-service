# Kaggle Train Guide (health-ai-service)

Mục tiêu: train **`model.pkl`** trên Kaggle (không train trên máy cá nhân), sau đó tải về và đặt vào đúng chỗ để `health-ai-service` tự load.

> Lưu ý: Kaggle phù hợp để **train/export**. Không dùng Kaggle để host API public ổn định cho app gọi.

## 1) Chuẩn bị dataset CSV (train thật)

### 1.1. Schema bắt buộc (10 cột)

CSV phải có header và các cột:

- `age` (int)
- `weight` (float, kg)
- `height` (float, cm)
- `gender` (`male` | `female` | `other`)
- `activity` (`sedentary` | `light` | `moderate` | `active` | `very-active`)
- `goal` (`weight-loss` | `weight-gain` | `muscle-gain` | `maintenance` | `health-improvement`)
- `calories` (number)
- `protein` (number, g)
- `carb` (number, g)
- `fat` (number, g)

Alias được trainer tự rename:

- `activityLevel` → `activity`
- `healthGoal` → `goal`
- `carbs` → `carb`

### 1.2. Dataset mẫu

Repo đã có file mẫu 10 dòng:

- `health-ai-service/data/nutrition_dataset.sample10.csv`

## 2) Tạo Kaggle Dataset (chứa CSV)

1. Kaggle → **Datasets** → **New Dataset**
2. Upload file CSV của bạn (ví dụ `nutrition_train.csv`)
3. Save

Sau đó Kaggle sẽ mount file vào dạng:

- `/kaggle/input/<dataset-name>/nutrition_train.csv`

## 3) Tạo Kaggle Notebook để train

1. Kaggle → **Code** → **New Notebook**
2. “Add data” → chọn Dataset CSV ở bước (2)
3. Upload folder `health-ai-service/` vào notebook (hoặc tối thiểu upload `train.py` + `requirements.txt`)

## 4) Chạy train (cell bash)

Trong notebook, chạy:

```bash
pip install -r health-ai-service/requirements.txt
python health-ai-service/train.py --csv "/kaggle/input/<dataset-name>/nutrition_train.csv"
```

Kết quả train sẽ tạo file:

- `health-ai-service/model/model.pkl`

## 5) Copy model sang Output để download

Kaggle chỉ hiển thị file trong `/kaggle/working` ở tab Output.

```bash
mkdir -p /kaggle/working/model_out
cp health-ai-service/model/model.pkl /kaggle/working/model_out/model.pkl
ls -lh /kaggle/working/model_out/model.pkl
```

Sau đó bạn download:

- `model_out/model.pkl`

## 6) Dùng `model.pkl` ở local hoặc deploy

### 6.1. Local

Đặt file vào:

- `health-ai-service/model/model.pkl`

Rồi chạy FastAPI như bình thường. Service sẽ tự load nếu file tồn tại.

### 6.2. Deploy

Bạn có thể:

- đặt đúng path `health-ai-service/model/model.pkl`, hoặc
- set env `MODEL_PATH` trỏ tới file `.pkl`.

