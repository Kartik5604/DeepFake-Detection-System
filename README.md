Deepfake Detection System (Media Authenticator)

A CNN based classification system for detecting deepfake and manipulated media, built using an iterative test, train and evaluate development cycle to maximize accuracy and generalization across datasets.

===============================================================================

Overview

The Media Authenticator is a deep learning system designed to classify images and video frames as authentic or AI manipulated (deepfake). It combines a convolutional neural network architecture with a custom preprocessing and augmentation pipeline, achieving 92% classification accuracy while remaining reusable across multiple datasets.

===============================================================================

Key Features
CNN based binary classifier trained to distinguish real vs deepfake media with 92% accuracy
Modular data preprocessing and augmentation pipeline that improved model accuracy by 5% and enabled reuse across 3+ datasets without rewriting core logic
Iterative test, train and evaluate workflow for continuous model refinement
OpenCV based frame extraction for processing video inputs alongside static images
Dataset agnostic architecture that can be adapted to new deepfake datasets with minimal configuration changes

===============================================================================
Tech Stack
Category	Tools / Libraries
Language	Python
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib
System Architecture
Input Media (Image/Video)
        |
        v
 Preprocessing Pipeline (OpenCV)
   Frame extraction (for video)
   Face detection and cropping
   Resizing and normalization
   Augmentation (rotation, flip, noise, brightness)
        |
        v
   CNN Classification Model (TensorFlow)
        |
        v
   Prediction: Real or Deepfake (with confidence score)
Project Structure
deepfake detection system/

  data/
    raw/                  Original datasets
    processed/            Preprocessed and augmented data
    dataset_config.yaml   Dataset specific configuration

  preprocessing/
    frame_extractor.py    Extracts frames from video input
    face_detector.py      Detects and crops facial regions
    augmentation.py       Applies data augmentation
    pipeline.py           Orchestrates the full preprocessing flow

  model/
    cnn_model.py          CNN architecture definition
    train.py               Training script
    evaluate.py            Evaluation and metrics script
    predict.py              Inference on new media

  utils/
    data_loader.py         Dataset loading utilities
    metrics.py              Accuracy, precision, recall, F1 tracking

  saved_models/
    best_model.h5           Trained model checkpoint

  notebooks/
    experiments.ipynb       Exploratory training experiments

  requirements.txt
  README.md

===============================================================================

Methodology

1. Data Collection and Preprocessing Curated media samples from multiple deepfake datasets, extracting frames from video and standardizing images through face detection, resizing, and normalization.

2. Data Augmentation Applied transformations such as rotation, horizontal flip, brightness and contrast shifts, and noise injection to increase dataset diversity and reduce overfitting. This contributed a 5% accuracy improvement.

3. Model Design Built a CNN architecture with convolutional, pooling, and dense layers, tuned through iterative experimentation.

4. Training and Evaluation Followed an iterative test, train and evaluate cycle, monitoring accuracy, precision, recall, and F1 score across validation splits to guide architecture and hyperparameter adjustments.

5. Generalization Testing Validated pipeline reusability by applying the same preprocessing and augmentation module across 3+ distinct datasets with minimal reconfiguration.

===============================================================================

Results
Metric	Value
Classification Accuracy	92%
Accuracy Gain from Augmentation Pipeline	+5%
Datasets Supported	3+
Setup and Usage

1. Clone the repository

bash
git clone
cd deepfake-detection-system

2. Install dependencies

bash
pip install -r requirements.txt

3. Preprocess data

bash
python preprocessing/pipeline.py --dataset data/raw/dataset_name

4. Train the model

bash
python model/train.py --epochs 30 --batch_size 32

5. Evaluate the model

bash
python model/evaluate.py --model saved_models/best_model.h5

6. Run inference on new media

bash
python model/predict.py --input path/to/media_file
requirements.txt
tensorflow>=2.12.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Future Improvements
Extend to real time video stream detection
Incorporate transformer based architectures (such as Vision Transformers) for improved robustness
Add explainability using Grad CAM to visualize regions influencing predictions
Deploy as a REST API or web interface for accessible media verification

===============================================================================

Author
Kartik Gite >> Junior Associate, AI/ML Engineer, ESDS Software Solutions Ltd. B.Tech, Artificial Intelligence and Data Science
