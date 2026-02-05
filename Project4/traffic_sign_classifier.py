"""
Traffic Sign Classification Model
This module implements a deep learning model for classifying traffic signs
using transfer learning with MobileNetV2.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2


class TrafficSignClassifier:
    """
    A traffic sign classifier using transfer learning with MobileNetV2.
    """
    
    def __init__(self, num_classes=43, img_size=(224, 224)):
        """
        Initialize the classifier.
        
        Args:
            num_classes (int): Number of traffic sign classes (default: 43 for German Traffic Signs)
            img_size (tuple): Input image size (height, width)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.class_names = self._get_class_names()
        
    def _get_class_names(self):
        """
        Get the names of traffic sign classes.
        
        Returns:
            list: List of class names
        """
        # Common traffic sign classes (German Traffic Sign Recognition Benchmark)
        class_names = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
            'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
            'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
            'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
            'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
            'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
            'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
            'Keep left', 'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ]
        return class_names
    
    def build_model(self, use_pretrained=True):
        """
        Build the model using transfer learning with MobileNetV2.
        
        Args:
            use_pretrained (bool): Whether to use pretrained weights
        """
        try:
            # Load pre-trained MobileNetV2 without top layers
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet' if use_pretrained else None
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Create new model on top
            inputs = keras.Input(shape=(*self.img_size, 3))
            
            # Pre-process input
            x = preprocess_input(inputs)
            
            # Base model
            x = base_model(x, training=False)
            
            # Add custom layers
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            self.model = keras.Model(inputs, outputs)
            
        except Exception as e:
            # Fallback: Create a simpler CNN model without pretrained weights
            print(f"⚠ Could not load pretrained weights: {e}")
            print("Creating a simpler CNN model without pretrained weights...")
            
            inputs = keras.Input(shape=(*self.img_size, 3))
            
            # Simple CNN architecture optimized for traffic signs
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        # Load image
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image_path):
        """
        Predict the traffic sign class for an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results with class name, confidence, and top 3 predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.class_names[i],
                'confidence': float(predictions[0][i])
            }
            for i in top_3_indices
        ]
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'top_3': top_3_predictions
        }
    
    def train_model(self, train_data_dir, val_data_dir=None, epochs=20, batch_size=32):
        """
        Train the model on traffic sign data.
        
        Args:
            train_data_dir (str): Directory containing training data
            val_data_dir (str): Directory containing validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Traffic signs shouldn't be flipped
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Load validation data if provided
        validation_generator = None
        if val_data_dir:
            validation_generator = val_datagen.flow_from_directory(
                val_data_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical'
            )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_data_dir else 'loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if val_data_dir else 'loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ]
        )
        
        return history
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or build a model first.")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")


def create_sample_model(save_path='models/traffic_sign_model.h5'):
    """
    Create and save a sample model for demonstration purposes.
    
    Args:
        save_path (str): Path to save the model
    """
    print("Creating sample traffic sign classification model...")
    
    # Create classifier
    classifier = TrafficSignClassifier()
    
    # Build model without pretrained weights (for offline demo)
    classifier.build_model(use_pretrained=False)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    classifier.save_model(save_path)
    
    print("Sample model created successfully!")
    return classifier


if __name__ == "__main__":
    # Create a sample model for demonstration
    create_sample_model()
