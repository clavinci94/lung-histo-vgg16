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
## 🔎 Grad-CAM Visualization
To visualize class activation maps (heatmaps), uncomment the Grad-CAM section in
lung_cancer_vgg16_refactor.py:
```
sample_img = test_df.iloc[0]["filepaths"]
cam_path = grad_cam(model, sample_img, layer_name="block5_conv3")
print(f"Grad-CAM saved to: {cam_path}")
```
## 🖥️ Requirements
Python 3.11+
TensorFlow 2.16.2 (tensorflow-macos + tensorflow-metal on Apple Silicon)
numpy, pandas, scikit-learn, matplotlib, seaborn, opencv-python

Install with:
```
pip install -r requirements.txt
```
📊 Results
Confusion matrix and classification report are automatically generated.
Final accuracy depends on dataset split and number of epochs.
Example output:
## 📊 Example Result

![Confusion Matrix](<img width="1050" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/89efc7bd-5b09-46c4-81c1-f4cb61839e96" />)



👤 Author
Developed by Claudio




