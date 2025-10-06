# 🫁 Lung Histopathology Classification with VGG16

This project applies **Transfer Learning with VGG16** to classify lung histopathology images  
(e.g., adenocarcinoma, squamous cell carcinoma, and normal tissue).  
It includes data preparation, model training, evaluation, and Grad-CAM visualization.

---

## 📂 Project Structure
```
lung-histo-vgg16/
│ lung_cancer_vgg16_refactor.py # main training script
│ requirements.txt # dependencies
│ .gitignore
│ README.md
│
└───artifacts/
├── model/ # trained models (.keras, .h5)
└── reports/ # confusion matrix, Grad-CAM outputs
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone git@github.com:clavinci94/lung-histo-vgg16.git
cd lung-histo-vgg16
```
## 2. Create virtual environment (recommended)
```
python3 -m venv .venv
source .venv/bin/activate
```
## 3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```



