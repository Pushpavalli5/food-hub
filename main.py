import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

class FoodRecognitionTracker:
    def __init__(self):
        # Load pre-trained model (using MobileNetV2 for lightweight recognition)
        self.model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=True
        )
        
        # Predefined food categories dictionary
        self.food_calories = {
            'pizza': 285,  # per 100g
            'salad': 50,
            'burger': 250,
            'apple': 52,
            'chicken': 165,
            'rice': 130,
            'fish': 206,
            'cake': 340
        }

    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        img = Image.open(image)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array

    def predict_food(self, image):
        """Predict food item and estimate calories"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Predict using ImageNet classes
        predictions = self.model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())
        
        # Get top prediction
        top_prediction = decoded_predictions[0][0]
        food_name = top_prediction[1].lower()
        confidence = top_prediction[2]
        
        # Estimate calories (use predefined or default)
        calories = self.food_calories.get(food_name, 100)
        
        return {
            'food_name': food_name,
            'confidence': float(confidence),
            'estimated_calories': calories
        }

def main():
    # Streamlit UI
    st.title("🍽️ Simple Food Recognition Tracker")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a food image", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Tracker instance
    tracker = FoodRecognitionTracker()
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Predict food
        try:
            prediction = tracker.predict_food(uploaded_file)
            
            # Display results
            st.subheader("Recognition Results")
            st.write(f"🍲 Food Item: {prediction['food_name'].capitalize()}")
            st.write(f"🎯 Confidence: {prediction['confidence']*100:.2f}%")
            st.write(f"🔥 Estimated Calories: {prediction['estimated_calories']} cal")
            
            # Progress visualization
            st.progress(prediction['confidence'])
        
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
