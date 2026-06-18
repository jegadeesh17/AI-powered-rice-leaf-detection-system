# AI-Powered Rice Leaf Disease Detection System

---

### **Project Overview**

Rice crops are highly vulnerable to leaf diseases that significantly reduce agricultural productivity and crop quality. This project builds an AI-powered rice leaf disease detection system using deep learning and computer vision techniques to automatically identify and classify rice leaf diseases from image data.

The system leverages convolutional neural networks (CNNs), image preprocessing pipelines, and deep learning-based classification models to analyze rice leaf images and predict disease categories with high accuracy. The project aims to support precision agriculture and assist farmers in early disease diagnosis for better crop management.

---

### **Key Features**

* **AI-Based Disease Detection:** Detects and classifies rice leaf diseases using deep learning models.
* **Image Preprocessing Pipeline:** Performs resizing, normalization, and augmentation for robust model training.
* **Deep Learning Classification:** Uses CNN-based architectures for multiclass disease prediction.
* **Automated Feature Learning:** Extracts disease patterns directly from rice leaf images.
* **Model Evaluation & Analytics:** Tracks model accuracy, loss, precision, recall, and confusion matrices.
* **Interactive Streamlit Web App:** Deployed application for uploading images and getting real-time predictions.
* **Transfer Learning Architectures:** Supports advanced models like MobileNetV2 and EfficientNetB0 for high accuracy.
* **Model Explainability (Grad-CAM):** Visualizes the regions of interest in the leaf that triggered the disease prediction.

---

### **Dataset**

* **Source:** Rice Leaf Disease Image Dataset ("Rice Leaf Disease Image Samples", Mendeley Data, V1, doi: 10.17632/fwcj7stb8r.1)
* **Coverage:** Multi-class disease leaf image samples
* **Data Type:** Labeled RGB rice leaf images

#### **Disease Categories**

* Bacterial Leaf Blight
* Brown Spot
* Leaf Smut
* Healthy Rice Leaves

#### **Key Features Analyzed**

* Leaf texture irregularities
* Color variations and discoloration regions
* Spot and lesion characteristics
* Disease spread and density patterns
* Image pixel intensity distributions

---

### **Project Structure**

```bash
RiceLeafDetection/
│
├── app/                          # Streamlit application files
│   └── app.py                    # Main Streamlit dashboard
├── data/                         # Project datasets
├── docs/                         # Documentation and visualizations
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── src/                          # Core Python logic and scripts
├── requirements.txt              # Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Image Preprocessing & Augmentation**

* Loads rice leaf image datasets
* Resizes images into model-compatible dimensions (e.g. 224x224)
* Normalizes pixel values for stable training
* Performs train-validation-test splitting

The system applies data augmentation techniques to improve model generalization:

| Augmentation Technique | Purpose                           |
| ---------------------- | --------------------------------- |
| Rotation               | Handles varying leaf orientations |
| Horizontal Flip        | Improves robustness               |
| Zoom                   | Captures scale variations         |
| Brightness Adjustment  | Handles lighting differences      |
| Rescaling              | Normalizes image pixel values     |

---

### **2. Deep Learning Architecture**

Uses convolutional neural networks (CNNs) and transfer learning models for automated disease classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

---

### **3. Model Explainability (Grad-CAM)**

The system integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to produce visual explanations from the CNN:
* Highlights specific disease patterns and lesion spots
* Aids agricultural experts in understanding model reasoning
* Validates classification focus points on the leaf structure

---

### **Model Performance**

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 95%+   |
| Precision | High   |
| Recall    | High   |
| F1-Score  | Strong |

---

### **Interactive Application Deployment**

The project features an interactive **Streamlit Web Application** designed with clean UI aesthetics, enabling users to upload leaf images and run real-time classification.

#### **To Launch the Platform Locally:**
```powershell
streamlit run app/app.py
```

---

### **Technology Stack**

| Category             | Tools                     |
| -------------------- | ------------------------- |
| Programming          | Python                    |
| Deep Learning        | TensorFlow, Keras         |
| Data Processing      | NumPy, Pandas             |
| Visualization        | Matplotlib, Seaborn       |
| Image Processing     | OpenCV                    |
| Notebook Environment | Jupyter Notebook          |
| Web Framework        | Streamlit                 |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/yourusername/Rice-Leaf-Disease-Diagnostic-AI-System.git

cd RiceLeafDetection
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **3. Launch Notebook**

```bash
jupyter notebook
```

Open:

```bash
rice_leaf_ai_system.ipynb
```

---

### **4. Train the Model**

```python
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)
```

---

### **Example Use Case**

A smart agriculture platform can use this system to:

1. Detect rice leaf diseases from uploaded images
2. Assist farmers with early disease identification
3. Reduce manual crop inspection effort
4. Improve crop yield through timely treatment

---

### **Future Improvements**

* Mobile app deployment for farmers
* Cloud-based agricultural monitoring dashboard
* Integration with fertilizer and treatment recommendation systems

---

### **Contributors**

* **Jegadeesh D** — Deep learning model development, image preprocessing, CNN training, evaluation, and agricultural AI analytics

---

### **License**

MIT License
