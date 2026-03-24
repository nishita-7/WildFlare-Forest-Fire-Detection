## WildFlare: Forest Fire Detection using CNN 🔥

A CNN-based binary image classifier that detects forest fires from images, built from scratch without any transfer learning. Trained on ~1500 images, achieves 96% test accuracy.

---

## What this project does

Upload any forest image and the model predicts whether it contains fire or not. It also generates a **Grad-CAM heatmap** showing which parts of the image the model focused on — useful for understanding what the CNN actually learned.

---

## Results

- **Test Accuracy:** 96.32%
- **ROC AUC:** 0.989
- Both fire and no-fire classes score ~0.96 F1

Train, validation and test metrics are all close to each other which suggests the model generalises well and isn't overfitting.

---

## Project structure

```
├── 01_train.ipynb       # build & train the CNN, visualize feature maps
├── 02_evaluate.ipynb    # confusion matrix, ROC, Grad-CAM, all evaluation graphs
├── app.py               # Streamlit app
├── models/              # saved model weights
├── outputs/             # all generated graphs
└── requirements.txt
```

---

## Model

Custom CNN from scratch — 4 convolutional blocks, each with Conv → BatchNorm → ReLU → MaxPool → Dropout, followed by Global Average Pooling and a Dense head. No pretrained weights used.

---

## Setup

```bash
git clone https://github.com/yourusername/forest-fire-detection.git
cd forest-fire-detection

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Add the dataset under `dataset/TrainingandValidation/` and `dataset/Testing/` (fire and nofire subfolders), then run the notebooks in order.

To launch the app:
```bash
streamlit run app.py
```

---

## Dataset

Forest fire image dataset sourced from Kaggle — 760 training images per class, 190 test images per class.

---

## Tech

Python · TensorFlow · Streamlit · OpenCV · scikit-learn

---

*Made by Nishita N*
