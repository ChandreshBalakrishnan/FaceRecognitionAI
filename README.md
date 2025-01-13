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
```
## Training the Model
Record data for different classes using the program.
Ensure the coords.csv file has sufficient data for each class.
Split the data into training and testing sets.

## Conclusion
This project provides a robust framework for recognizing body language using Mediapipe and machine learning. By combining landmark detection with predictive modeling, it offers a practical tool for real-time applications such as gesture recognition, health monitoring, and interactive interfaces.

Future improvements could involve fine-tuning models with larger datasets, integrating advanced deep learning techniques, or expanding the scope to include more complex gestures and actions. With its modular design, this project is a strong foundation for further exploration in human-computer interaction and computer vision.

