# Nationality and Emotion Prediction Model 🌍😊

This project uses YOLOv8 and CNN (trained on multiple datasets) to predict the nationality, emotion, age, and dress color of a person from an uploaded image. The model tailors its predictions based on the nationality of the individual, and applies various conditions for different nationalities, such as:

Predicting age and dress color for Indian nationality.

Predicting age and emotion for United States nationality.

Predicting emotion and dress color for African nationality.

Only predicting nationality and emotion for other nationalities.

## Features ✨

Predicts nationality based on the person's facial features (e.g., Indian, American, African, etc.). 🌏

Detects emotion (happy, sad, angry, etc.) from the face. 😄😢

Based on nationality, predicts additional attributes:

Age and dress color for Indians 🇮🇳.

Age and emotion for United States 🇺🇸.

Emotion and dress color for Africans 🇿🇦.

Nationality and emotion for other nationalities.

Rejects predictions for ages below 10 and above 60.

## Technologies Used 🔧

YOLOv8 (PyTorch): For facial detection and segmentation.

CNN (Keras): For nationality, emotion, age, and dress color prediction.

OpenCV: For image and video processing.

## Installation ⚙️

### Clone the repository:

git clone https://github.com/Hariarul/VisionAI-Advanced-Image-Video-Analysis

### Install dependencies:

pip install -r requirements.txt

Download the pre-trained models:

YOLOv8 (for facial detection).

CNN models for age, emotion, dress color, and nationality prediction.

### Run the application:

streamlit run Nationality.py

## How It Works 🎬

Image Upload: Upload an image of a person, and the model predicts:

Nationality (Indian, United States, African, etc.).

Emotion (happy, sad, angry, etc.).

Additional predictions based on the nationality:

Indians: Predicts age and dress color.

United States: Predicts age and emotion.

Africans: Predicts emotion and dress color.

Other Nationalities: Only predicts nationality and emotion.

Age Restriction: The model only predicts age if the person is between 10-60 years old. Ages below 10 and above 60 are rejected.

Results Display: Displays predicted attributes like nationality, emotion, age, and dress color on-screen.

## Example Output 🧑‍🤝‍🧑

### Input Image: A person from India.

### Displayed Output:

Nationality: Indian

Emotion: Happy 😊

Age: 25 years

### Input Image: A person from the United States.

### Displayed Output:

Nationality: United States 

Emotion: Sad 😢

Age: 30 years

### Input Image: A person from Africa.

### Displayed Output:

Nationality: African 🇿🇦

Emotion: Angry 😡

Input Image: A person from any other nationality.

## Results 📊
Accuracy: The model achieves an accuracy of 70% for nationality and emotion prediction, and 75% for age and dress color prediction across various nationalities.
Real-time Processing: The model can process 5-10 frames per second in real-time video streams.

### Example Display:
### Indian Nationality:

Nationality: Indian

Emotion: Happy 😊

Age: 25 years

### United States Nationality:

Nationality: United States 

Emotion: Sad 😢

Age: 30 years

### African Nationality:

Nationality: African 

Emotion: Angry 😡

