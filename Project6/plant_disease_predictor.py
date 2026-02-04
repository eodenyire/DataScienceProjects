"""
Plant Disease Predictor Module for identifying plant diseases from leaf images.

This module provides functionality to build, train, and use a deep learning model
for plant disease classification using transfer learning with MobileNetV2.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')


class PlantDiseasePredictor:
    """
    A class for predicting plant diseases from leaf images using deep learning.
    
    This class provides methods to build, train, and use a CNN model based on
    MobileNetV2 for plant disease classification.
    """
    
    # Plant disease classes
    DISEASE_CLASSES = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    # Disease information and recommendations
    DISEASE_INFO = {
        'Apple___Apple_scab': {
            'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit.',
            'treatment': 'Apply fungicides, remove infected leaves, and ensure good air circulation.'
        },
        'Apple___Black_rot': {
            'description': 'Fungal disease causing leaf spots, fruit rot, and cankers.',
            'treatment': 'Prune infected branches, apply fungicides, and remove mummified fruit.'
        },
        'Apple___Cedar_apple_rust': {
            'description': 'Fungal disease causing orange spots on leaves.',
            'treatment': 'Remove nearby cedar trees, apply fungicides in spring, and plant resistant varieties.'
        },
        'Apple___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Blueberry___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Cherry_(including_sour)___Powdery_mildew': {
            'description': 'Fungal disease causing white powdery coating on leaves.',
            'treatment': 'Apply sulfur-based fungicides, improve air circulation, and avoid overhead watering.'
        },
        'Cherry_(including_sour)___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'description': 'Fungal disease causing gray rectangular lesions on leaves.',
            'treatment': 'Use resistant varieties, rotate crops, and apply fungicides if severe.'
        },
        'Corn_(maize)___Common_rust_': {
            'description': 'Fungal disease causing reddish-brown pustules on leaves.',
            'treatment': 'Plant resistant varieties and apply fungicides if infection is severe.'
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'description': 'Fungal disease causing long gray-green lesions on leaves.',
            'treatment': 'Use resistant hybrids, practice crop rotation, and manage crop residue.'
        },
        'Corn_(maize)___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Grape___Black_rot': {
            'description': 'Fungal disease causing leaf spots and fruit mummification.',
            'treatment': 'Remove infected fruit, apply fungicides, and ensure good air circulation.'
        },
        'Grape___Esca_(Black_Measles)': {
            'description': 'Fungal disease causing leaf discoloration and fruit spotting.',
            'treatment': 'Prune infected vines, avoid wounding, and apply appropriate fungicides.'
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'description': 'Fungal disease causing brown spots on leaves.',
            'treatment': 'Apply fungicides, remove infected leaves, and improve vineyard management.'
        },
        'Grape___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'description': 'Bacterial disease causing yellowing, misshapen fruit, and tree decline.',
            'treatment': 'Remove infected trees, control psyllid vectors, and use disease-free nursery stock.'
        },
        'Peach___Bacterial_spot': {
            'description': 'Bacterial disease causing dark spots on leaves and fruit.',
            'treatment': 'Apply copper-based bactericides, prune for air circulation, and use resistant varieties.'
        },
        'Peach___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Pepper,_bell___Bacterial_spot': {
            'description': 'Bacterial disease causing dark spots on leaves and fruit.',
            'treatment': 'Use disease-free seeds, apply copper sprays, and practice crop rotation.'
        },
        'Pepper,_bell___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Potato___Early_blight': {
            'description': 'Fungal disease causing dark spots with concentric rings on leaves.',
            'treatment': 'Apply fungicides, remove infected leaves, and practice crop rotation.'
        },
        'Potato___Late_blight': {
            'description': 'Oomycete disease causing water-soaked lesions and white mold.',
            'treatment': 'Apply fungicides preventively, destroy infected plants, and use resistant varieties.'
        },
        'Potato___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Raspberry___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Soybean___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Squash___Powdery_mildew': {
            'description': 'Fungal disease causing white powdery coating on leaves.',
            'treatment': 'Apply fungicides, improve air circulation, and remove infected leaves.'
        },
        'Strawberry___Leaf_scorch': {
            'description': 'Fungal disease causing purple spots that turn brown on leaves.',
            'treatment': 'Remove infected leaves, apply fungicides, and ensure good air circulation.'
        },
        'Strawberry___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Tomato___Bacterial_spot': {
            'description': 'Bacterial disease causing dark spots on leaves and fruit.',
            'treatment': 'Use disease-free seeds, apply copper sprays, and avoid overhead watering.'
        },
        'Tomato___Early_blight': {
            'description': 'Fungal disease causing brown spots with concentric rings.',
            'treatment': 'Apply fungicides, practice crop rotation, and remove lower leaves.'
        },
        'Tomato___Late_blight': {
            'description': 'Oomycete disease causing water-soaked spots and plant collapse.',
            'treatment': 'Apply fungicides preventively, destroy infected plants, and improve air circulation.'
        },
        'Tomato___Leaf_Mold': {
            'description': 'Fungal disease causing yellow spots on upper leaf surfaces.',
            'treatment': 'Improve ventilation, reduce humidity, and apply fungicides.'
        },
        'Tomato___Septoria_leaf_spot': {
            'description': 'Fungal disease causing small circular spots with dark borders.',
            'treatment': 'Remove infected leaves, apply fungicides, and avoid overhead watering.'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'description': 'Pest causing stippling and yellowing of leaves.',
            'treatment': 'Apply miticides, use predatory mites, and maintain plant health.'
        },
        'Tomato___Target_Spot': {
            'description': 'Fungal disease causing concentric ring patterns on leaves.',
            'treatment': 'Apply fungicides, practice crop rotation, and remove infected debris.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'description': 'Viral disease causing leaf yellowing and curling.',
            'treatment': 'Control whitefly vectors, remove infected plants, and use resistant varieties.'
        },
        'Tomato___Tomato_mosaic_virus': {
            'description': 'Viral disease causing mottled leaves and stunted growth.',
            'treatment': 'Remove infected plants, disinfect tools, and use disease-free seeds.'
        },
        'Tomato___healthy': {
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'treatment': 'Continue regular care and monitoring.'
        }
    }
    
    def __init__(self, model_path=None, img_size=(224, 224)):
        """
        Initialize the PlantDiseasePredictor.
        
        Args:
            model_path (str, optional): Path to pre-trained model file
            img_size (tuple): Target image size for model input
        """
        self.model = None
        self.img_size = img_size
        self.num_classes = len(self.DISEASE_CLASSES)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self):
        """
        Build the CNN model using transfer learning with MobileNetV2.
        
        Returns:
            keras.Model: Compiled model ready for training
        """
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Build the model
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.img_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Expand dimensions to create batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image_path, top_k=3):
        """
        Predict plant disease from an image.
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return
        
        Returns:
            dict: Prediction results with disease names, confidences, and info
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please build or load a model first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            disease_name = self.DISEASE_CLASSES[idx]
            confidence = float(predictions[idx] * 100)
            
            # Get disease information
            disease_info = self.DISEASE_INFO.get(disease_name, {
                'description': 'No information available.',
                'treatment': 'Consult with a plant pathologist.'
            })
            
            results.append({
                'disease': disease_name,
                'confidence': confidence,
                'description': disease_info['description'],
                'treatment': disease_info['treatment']
            })
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please build or load a model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_disease_info(self, disease_name):
        """
        Get information about a specific disease.
        
        Args:
            disease_name (str): Name of the disease
        
        Returns:
            dict: Disease information including description and treatment
        """
        return self.DISEASE_INFO.get(disease_name, {
            'description': 'No information available.',
            'treatment': 'Consult with a plant pathologist.'
        })


def create_demo_model():
    """
    Create a demo model for testing purposes.
    This function builds a model without training it.
    
    Returns:
        PlantDiseasePredictor: Predictor with built model
    """
    predictor = PlantDiseasePredictor()
    predictor.build_model()
    print("Demo model created successfully!")
    print(f"Model has {predictor.num_classes} output classes")
    return predictor


if __name__ == "__main__":
    # Create and display model summary
    predictor = create_demo_model()
    if predictor.model:
        print("\nModel Summary:")
        predictor.model.summary()
