"""
Dog Breed Classification Model
This script demonstrates how to build and train a dog breed classification model
using transfer learning with a pre-trained MobileNetV2 model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class DogBreedClassifier:
    """
    A dog breed classifier using transfer learning with MobileNetV2.
    """
    
    def __init__(self, num_classes=10, img_size=(224, 224)):
        """
        Initialize the classifier.
        
        Args:
            num_classes (int): Number of dog breed classes
            img_size (tuple): Input image size (height, width)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.class_names = []
        
    def build_model(self, use_pretrained=True):
        """
        Build the model using transfer learning with MobileNetV2.
        
        Args:
            use_pretrained (bool): Whether to use pretrained weights. Set to False for offline mode.
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
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            self.model = keras.Model(inputs, outputs)
            
        except Exception as e:
            # Fallback: Create a simpler CNN model without pretrained weights
            print(f"⚠ Could not load pretrained weights: {e}")
            print("Creating a simpler CNN model without pretrained weights...")
            
            inputs = keras.Input(shape=(*self.img_size, 3))
            
            # Simple CNN architecture
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
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
    
    def create_sample_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators for training and validation.
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Save class names
        self.class_names = list(train_generator.class_indices.keys())
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=10):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            
        Returns:
            History object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        return history
    
    def save_model(self, filepath='models/dog_breed_model.h5'):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save class names
        class_names_path = filepath.replace('.h5', '_classes.txt')
        with open(class_names_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"Class names saved to {class_names_path}")
    
    def load_model(self, filepath='models/dog_breed_model.h5'):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load class names
        class_names_path = filepath.replace('.h5', '_classes.txt')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Class names loaded from {class_names_path}")
    
    def predict(self, img_array):
        """
        Predict the breed of a dog from an image array.
        
        Args:
            img_array (numpy.ndarray): Image array of shape (height, width, 3)
            
        Returns:
            tuple: (predicted_class, confidence, all_predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        img_array = tf.image.resize(img_array, self.img_size)
        img_array = tf.expand_dims(img_array, 0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = self.class_names[predicted_class_idx] if self.class_names else f"Class {predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (self.class_names[idx] if self.class_names else f"Class {idx}", predictions[0][idx])
            for idx in top_3_idx
        ]
        
        return predicted_class, float(confidence), top_3_predictions


def create_sample_model():
    """
    Create and save a sample model for demonstration purposes.
    This creates a model with 10 common dog breeds.
    """
    # Common dog breeds for the model
    breeds = [
        'beagle', 'boxer', 'bulldog', 'chihuahua', 'golden_retriever',
        'german_shepherd', 'labrador', 'poodle', 'rottweiler', 'yorkshire_terrier'
    ]
    
    print("Creating sample dog breed classification model...")
    classifier = DogBreedClassifier(num_classes=len(breeds))
    classifier.class_names = breeds
    
    # Build the model
    model = classifier.build_model()
    print(f"\nModel architecture:")
    model.summary()
    
    # Save the model (untrained, for demonstration)
    classifier.save_model('models/dog_breed_model.h5')
    
    print("\n✓ Sample model created successfully!")
    print(f"✓ Model supports {len(breeds)} dog breeds:")
    for i, breed in enumerate(breeds, 1):
        print(f"  {i}. {breed.replace('_', ' ').title()}")
    
    return classifier


if __name__ == '__main__':
    """
    This main section demonstrates how to use the DogBreedClassifier.
    In a real scenario, you would:
    1. Prepare your dataset in train/val directories
    2. Create data generators
    3. Train the model
    4. Save the model
    
    For this demo, we create a sample pre-built model structure.
    """
    print("=" * 70)
    print("Dog Breed Classification Model Builder")
    print("=" * 70)
    
    # Create sample model
    classifier = create_sample_model()
    
    print("\n" + "=" * 70)
    print("Model Training Instructions (for real datasets):")
    print("=" * 70)
    print("""
To train with your own dataset:

1. Organize your data:
   image_data/
   ├── train/
   │   ├── beagle/
   │   ├── boxer/
   │   └── ...
   └── val/
       ├── beagle/
       ├── boxer/
       └── ...

2. Use the following code:
   
   classifier = DogBreedClassifier(num_classes=10)
   classifier.build_model()
   
   train_gen, val_gen = classifier.create_sample_data_generators(
       'image_data/train',
       'image_data/val'
   )
   
   history = classifier.train(train_gen, val_gen, epochs=20)
   classifier.save_model('models/dog_breed_model.h5')

3. The trained model will be saved and ready for use in the Flask app.
    """)
