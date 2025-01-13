Body Language Recognition using Mediapipe and Machine Learning
This project leverages Mediapipe for pose, face, and hand landmark detection combined with machine learning models to classify body language. The application processes webcam input, extracts landmarks, trains classification models, and predicts body language classes in real-time.

Features
Real-time detection of facial, pose, and hand landmarks using Mediapipe.
Data recording of landmark coordinates to a CSV file for training.
Machine learning classification using Logistic Regression, Ridge Classifier, Random Forest, and Gradient Boosting models.
Real-time prediction and display of body language class and probability.
Requirements
Ensure you have the following libraries installed:

Python 3.7 or higher
OpenCV
Mediapipe
NumPy
Pandas
Scikit-learn
Pickle
You can install these dependencies using:

bash
Copy code
pip install mediapipe opencv-python numpy pandas scikit-learn
Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/body-language-recognition.git
cd body-language-recognition
Create a file named coords.csv in the project directory:

bash
Copy code
touch coords.csv
Add the following headers to coords.csv:

csv
Copy code
class,x1,y1,z1,v1,x2,y2,z2,v2,...,xn,yn,zn,vn
Replace n with the total number of landmark coordinates (computed from the Mediapipe model).

Usage
1. Data Collection
Run the script to collect data for a specific class:

bash
Copy code
python collect_data.py
Replace class_name in the script with the desired body language class (e.g., "Happy", "Sad").

2. Model Training
Once data collection is complete:

Load the collected data from coords.csv.
Split the data into training and testing sets.
Train the machine learning models.
Save the trained model using Pickle.
3. Real-Time Prediction
Run the real-time prediction script:

bash
Copy code
python predict_real_time.py
The webcam feed will display:

Detected landmarks.
Predicted body language class and confidence scores.
Project Structure
collect_data.py: Script for collecting landmark data.
train_model.py: Script for training machine learning models.
predict_real_time.py: Real-time body language classification.
coords.csv: CSV file to store collected landmark data.
body_language.pkl: Pickled machine learning model.
Example
Launch the webcam feed for real-time detection and prediction:

bash
Copy code
python predict_real_time.py
View the detected body language class and confidence probabilities on the webcam feed.

Notes
Ensure your coords.csv file is correctly formatted before starting the model training process.
Modify the Mediapipe confidence thresholds as needed in the script.
Use the fit_models dictionary to experiment with different machine learning models.
Acknowledgments
Mediapipe for the powerful computer vision tools.
OpenCV for real-time video processing.
License
This project is licensed under the MIT License. Feel free to use and modify as needed.
