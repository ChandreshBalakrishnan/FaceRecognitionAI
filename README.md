# Body Language Recognition using Mediapipe and Machine Learning

This project uses Mediapipe for detecting pose, face, and hand landmarks, combined with machine learning models to classify body language in real-time.

---

## Features

- Real-time detection of face, pose, and hand landmarks using Mediapipe.
- Landmark coordinate recording into a CSV file for training machine learning models.
- Training and evaluation of classification models using Logistic Regression, Ridge Classifier, Random Forest, and Gradient Boosting.
- Real-time predictions displayed with confidence scores on a webcam feed.

---

## Requirements

Ensure you have the following dependencies installed:
- **Python 3.7 or higher**
- Mediapipe
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Pickle

Install dependencies using pip:
```bash
pip install mediapipe opencv-python numpy pandas scikit-learn
