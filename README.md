# Facial-Emotion-Detection-System

---

## 🚀 Project Overview & Web Interface

This is an AI-powered **Facial Emotion Detection System** running a custom-trained **YOLO** model (`best.pt`). We have built a premium, responsive glassmorphism web interface that allows users to analyze facial expressions in real-time.

### 🌟 Key Features
- **Dual Analysis Modes**:
  - 📂 **Image Upload**: Drag-and-drop or file pick your portrait photos.
  - 📷 **Live Camera**: Capture a snapshot directly from your webcam.
- **Smart Camera Prioritization**:
  - Automatically identifies and defaults to your **integrated laptop camera** rather than any connected mobile phones or virtual feeds.
  - Generates a manual selection dropdown menu dynamically if multiple cameras are detected.
- **Vibrant Analytics Dashboard**:
  - Renders annotated faces with bounding boxes.
  - Populates real-time statistics showing emotion classifications alongside visual progress bars.

### 🏷️ Supported Emotion Categories
The model classifies detections into **8 distinct facial emotions**:
- 😠 **Angry**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😌 **Content**
- 🤢 **Disgust**
- 😐 **Neutral**
- 😲 **Surprise**

---

## 🧠 Model Training & Experimentation

Based on the [experiment.ipynb](file:///c:/Users/DELL/Desktop/face/Facial-Emotion-Detection-System/notebook/experiment.ipynb) notebook, here are the details of the machine learning training pipeline:

### 📊 Dataset & Class Distributions
- **Merged Dataset (`merged_emotions`)**: Combines a secondary 8-class facial expression dataset with a primary dataset to form a large-scale unified training set:
  - **Train split**: 8,329 images/labels
  - **Validation split**: 2,038 images/labels
  - **Test split**: 1,024 images/labels
  - **Total**: 11,391 images (467.49 MB)
- **Class Distributions (Training Set)**:
  - `happy`: 2,892 objects
  - `sad`: 2,210 objects
  - `angry`: 2,041 objects
  - `fear`: 1,800 objects
  - `surprise`: 873 objects
  - `neutral`: 866 objects
  - `disgust`: 807 objects
  - `content`: 789 objects

### ⚙️ Training Parameters (YOLOv11n)
- **Model Type**: Object Detection (`detect` task)
- **Computational Complexity**: 6.3 GFLOPs (Fused parameters: 2,582,932)
- **Input Image Size**: 640x640 pixels
- **Batch Size**: 16
- **Data Augmentation Strategy**:
  - **Horizontal Flipping (0.5)**: Randomly mirrors training images to learn orientation-invariant features.
  - **HSV Augmentation**: Hue (0.015), Saturation (0.7), and Value (0.4) scaling to adapt to lighting and skin tone changes.
  - **Random Erasing (0.4)**: Randomly masks image blocks to build robustness to partial occlusions (e.g. glasses, face masks, shadows).

### 📈 Baseline Evaluation Metrics
On initial model validation, the baseline weights yield the following performance figures:
- **mAP@0.5**: `0.838` (83.8%)
- **Precision (P)**: `0.828` (82.8%)
- **Recall (R)**: `0.737` (73.7%)
- **Class-specific mAP@0.5**:
  - `happy`: `0.941` (94.1%)
  - `sad`: `0.827` (82.7%)
  - `angry`: `0.811` (81.1%)
  - `fear`: `0.774` (77.4%)

---

## 🛠️ Quick Start & Setup

#### 1. Install Dependencies
Make sure you have python installed, then run:
```bash
pip install -r requirements.txt
```
*(If you are using `uv`, you can install them using `uv pip install -r requirements.txt`)*

#### 2. Start the Server
Launch the FastAPI backend:
```bash
python app.py
```
*(or `uv run python app.py`)*

#### 3. Run in the Browser
Open your web browser and go to:
[http://localhost:8080](http://localhost:8080)

*Note: Browsers require local environments (`localhost` / `127.0.0.1`) or secure connections (`https`) to access webcam hardware.*

---

## 📚 Machine Learning Concepts

learning 

Average Precision (AP).

AP is essentially:

Area under the Precision-Recall curve

The larger the area, the better.

What is mAP@0.5?

 average Precision (AP) is calculated for each class. The mean Average Precision (mAP) is then calculated by taking the average of the AP values across all classes.