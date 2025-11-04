
<div align="center">
 
#  CSSM
**Efficient Remote Sensing Change Detection with Change State Space Models**

[**E.Ghazaei**](https://scholar.google.com/citations?user=R-ghC00AAAAJ&hl=en), [**E.Aptoula**](https://sites.google.com/view/erchan-aptoula/) 

 Faculty of Engineering and Natural Sciences (VPALab), Sabanci University, Istanbul, Turkiye

[[Paper Link](https://arxiv.org/abs/2504.11080)] 
</div>



## 🛎️Updates
* **` Notice🐍🐍`**: CSSM has been accepted by [IEEE GRSL](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8859)! We'd appreciate it if you could give this repo a ⭐️**star**⭐️ and stay tuned!!
* **` Nov 05th, 2025`**: The CSSM model and training code uploaded. You are welcome to use them!!




---


## 🚀 Overview


* [**CSSM**]() serves as an efficient and state-of-the-art (SOTA) benchmark for binary change detection.



<p align="center">
  
<img width="1395" height="579" alt="Screenshot from 2025-11-03 16-28-31" src="https://github.com/user-attachments/assets/dccfdfc5-98b4-443d-b170-07e5e3ec551d" />
</p>


---


## 📦 Requirements
```bash
pip install torch torchvision
pip install numpy matplotlib
pip install opencv-python
# Add other dependencies as needed
```


---

## 📁 Dataset Preparation

This project supports three main change detection datasets:
- **LEVIR-CD+** - [Download](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection)
- **SYSU-CD** - [Download](https://github.com/liumency/SYSU-CD)
- **WHU-CD** - [Download](http://gpcv.whu.edu.cn/data/building_dataset.html)

---

### Dataset Structure

#### For LEVIR-CD:
```
your_dataset/
├── train/
│   ├── A/          # Pre-change images
│   ├── B/          # Post-change images
│   └── label/      # Ground truth masks
├── test/
│   ├── A/
│   ├── B/
│   └── label/
└── val/
    ├── A/
    ├── B/
    └── label/
```


#### For SYSU-CD:
```
your_dataset/
├── train/
│   ├── time1/          # Pre-change images
│   ├── time2/          # Post-change images
│   └── label/      # Ground truth masks
├── test/
│   ├── time1/
│   ├── time2/
│   └── label/
└── val/
    ├── time1/
    ├── time2/
    └── label/
```

#### For WHU-CD:
```
WHU-CD/
├── A/              # Pre-change images
├── B/              # Post-change images
├── label/          # Ground truth masks
├── train_list.txt  # List of training samples
├── test_list.txt   # List of test samples
└── val_list.txt    # List of validation samples
```

The text files should contain image names (one per line):
```
image_001.png
image_002.png
image_003.png
...
```
---

## 🚂 Training

### LEVIR-CD Dataset
```bash
python main.py \
    --dataset levir \
    --train_path /path/to/LEVIR-CD/train \
    --test_path /path/to/LEVIR-CD/test \
    --val_path /path/to/LEVIR-CD/val \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001
```

### SYSU-CD Dataset
```bash
python main.py \
    --dataset sysu \
    --train_path /path/to/SYSU-CD/train \
    --test_path /path/to/SYSU-CD/test \
    --val_path /path/to/SYSU-CD/val \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0001
```

### WHU-CD Dataset
```bash
python main.py \
    --dataset whu \
    --train_path /path/to/WHU-CD \
    --train_txt /path/to/train_list.txt \
    --test_txt /path/to/test_list.txt \
    --val_txt /path/to/val_list.txt \
    --batch_size 64 \
    --epochs 50
```
---

## ⚙️ Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset type: `levir`, `sysu`, or `whu` | `--dataset levir` |
| `--train_path` | Path to training data | `--train_path /data/train` |
| `--test_path` | Path to test data (not for WHU) | `--test_path /data/test` |
| `--val_path` | Path to validation data (not for WHU) | `--val_path /data/val` |

### WHU-CD Specific Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--train_txt` | Training sample list file | `--train_txt train_list.txt` |
| `--test_txt` | Test sample list file | `--test_txt test_list.txt` |
| `--val_txt` | Validation sample list file | `--val_txt val_list.txt` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 64 | Batch size for training |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--step_size` | 10 | Learning rate scheduler step size |
| `--save_dir` | ./checkpoints | Directory to save model checkpoints |
| `--model_name` | best_model.pth | Filename for saved model |
| `--seed` | 42 | Random seed for reproducibility |
| `--num_workers` | 4 | Number of data loading workers |

---

## 🔧 Advanced Usage Examples

### Custom Save Directory and Model Name
```bash
python main.py \
    --dataset levir \
    --train_path /data/LEVIR-CD/train \
    --test_path /data/LEVIR-CD/test \
    --val_path /data/LEVIR-CD/val \
    --save_dir ./experiments/levir_exp1 \
    --model_name levir_model.pth \
    --epochs 100
```

### Different Learning Rate Schedule
```bash
python main.py \
    --dataset sysu \
    --train_path /data/SYSU-CD/train \
    --test_path /data/SYSU-CD/test \
    --val_path /data/SYSU-CD/val \
    --lr 0.0005 \
    --step_size 20 \
    --epochs 150
```

### Smaller Batch Size (for limited GPU memory)
```bash
python main.py \
    --dataset levir \
    --train_path /data/train \
    --test_path /data/test \
    --val_path /data/val \
    --batch_size 16 \
    --num_workers 2
```

---

## 📤 Output

During training, the script will:
- Display training loss for each batch
- Show validation metrics (IoU, confusion matrix) after each epoch
- Save the best model based on validation IoU
- Display learning rate and epoch time

### Model Checkpoint

The best model is automatically saved to:
```
{save_dir}/{model_name}
```

Default: `./checkpoints/best_model.pth`

---

## 🔍 Troubleshooting

### Paths with Spaces

If your paths contain spaces, wrap them in quotes:
```bash
python main.py \
    --dataset levir \
    --train_path "/path/with spaces/train" \
    --test_path "/path/with spaces/test" \
    --val_path "/path/with spaces/val"
```

### CUDA Out of Memory

Reduce batch size:
```bash
python main.py --dataset levir ... --batch_size 16
```

### Missing WHU Text Files

For WHU dataset, ensure all three text files are provided:
```bash
python main.py \
    --dataset whu \
    --train_path /data/WHU-CD \
    --train_txt train_list.txt \
    --test_txt test_list.txt \
    --val_txt val_list.txt
```

---

## 💡 Getting Help

View all available arguments:
```bash
python main.py --help
```

---

## 📧 Contact

If you have any questions, please contact Elman Ghazaei at elman.ghazaei@sabanciuniv.edu

---

## Qualitative Analysis:





<p align="center">
<img width="1379" height="357" alt="Screenshot from 2025-11-03 16-38-52" src="https://github.com/user-attachments/assets/c63690af-fd07-40af-b991-2b5b33ff53af" />
</p>

---
# Results

<img width="1365" height="780" alt="Screenshot from 2025-11-03 18-02-18" src="https://github.com/user-attachments/assets/2dabac8d-9ab5-467d-9dbe-6aa5266b2e5f" />



