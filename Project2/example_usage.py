"""
Example Usage of Dog Breed Classification Model
This script demonstrates how to use the DogBreedClassifier for predictions.
"""

import os
import numpy as np
from PIL import Image
from dog_breed_model import DogBreedClassifier, create_sample_model


def example_create_and_save_model():
    """
    Example: Create and save a model.
    """
    print("\n" + "="*70)
    print("Example 1: Creating and Saving a Model")
    print("="*70)
    
    # Create a sample model
    classifier = create_sample_model()
    
    print("\n✓ Model created and saved successfully!")
    print(f"✓ Model file: models/dog_breed_model.h5")
    print(f"✓ Classes file: models/dog_breed_model_classes.txt")


def example_load_and_predict():
    """
    Example: Load a model and make predictions.
    """
    print("\n" + "="*70)
    print("Example 2: Loading Model and Making Predictions")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('models/dog_breed_model.h5'):
        print("\n⚠ Model not found. Creating sample model first...")
        create_sample_model()
    
    # Initialize and load model
    classifier = DogBreedClassifier(num_classes=10)
    classifier.load_model('models/dog_breed_model.h5')
    
    print("\n✓ Model loaded successfully!")
    print(f"✓ Number of classes: {len(classifier.class_names)}")
    print(f"✓ Supported breeds: {', '.join(classifier.class_names)}")
    
    # Create a sample image for demonstration
    # In real usage, you would load an actual dog image
    print("\nNote: For actual predictions, provide a real dog image.")
    print("Example code for loading and predicting:")
    print("""
    # Load your dog image
    from PIL import Image
    import numpy as np
    
    img = Image.open('path/to/your/dog_image.jpg')
    img_array = np.array(img)
    
    # Make prediction
    breed, confidence, top_3 = classifier.predict(img_array)
    
    print(f"Predicted Breed: {breed}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\\nTop 3 Predictions:")
    for i, (breed_name, conf) in enumerate(top_3, 1):
        print(f"{i}. {breed_name}: {conf * 100:.2f}%")
    """)


def example_model_architecture():
    """
    Example: Display model architecture.
    """
    print("\n" + "="*70)
    print("Example 3: Model Architecture")
    print("="*70)
    
    classifier = DogBreedClassifier(num_classes=10)
    model = classifier.build_model()
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel Details:")
    print(f"✓ Total Parameters: {model.count_params():,}")
    print(f"✓ Input Shape: {model.input_shape}")
    print(f"✓ Output Shape: {model.output_shape}")


def example_training_workflow():
    """
    Example: Show training workflow (conceptual).
    """
    print("\n" + "="*70)
    print("Example 4: Training Workflow (Conceptual)")
    print("="*70)
    
    print("""
To train the model with your own dataset:

1. Prepare Your Dataset:
   ----------------------
   Organize images in this structure:
   
   image_data/
   ├── train/
   │   ├── beagle/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── boxer/
   │   │   └── ...
   │   └── ... (other breeds)
   └── val/
       ├── beagle/
       ├── boxer/
       └── ... (other breeds)

2. Initialize and Build Model:
   ---------------------------
   from dog_breed_model import DogBreedClassifier
   
   classifier = DogBreedClassifier(num_classes=10)
   classifier.build_model()

3. Create Data Generators:
   -----------------------
   train_gen, val_gen = classifier.create_sample_data_generators(
       train_dir='image_data/train',
       val_dir='image_data/val',
       batch_size=32
   )

4. Train the Model:
   ----------------
   history = classifier.train(
       train_generator=train_gen,
       val_generator=val_gen,
       epochs=20
   )

5. Save the Model:
   ---------------
   classifier.save_model('models/dog_breed_model.h5')

6. Evaluate:
   ---------
   # Training history is returned with metrics
   print(f"Final accuracy: {history.history['accuracy'][-1]:.2f}")
   print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.2f}")
    """)


def example_web_app_integration():
    """
    Example: Integration with Flask web app.
    """
    print("\n" + "="*70)
    print("Example 5: Web App Integration")
    print("="*70)
    
    print("""
The model is automatically loaded in the Flask app (app.py).

To run the web application:

1. Ensure model exists:
   python dog_breed_model.py

2. Start the Flask app:
   python app.py

3. Open browser:
   http://localhost:5000

4. Upload dog image:
   - Click "Choose Image"
   - Select a dog photo
   - Click "Predict Breed"

5. View results:
   - Predicted breed
   - Confidence score
   - Top 3 predictions

The Flask app handles:
- File upload validation
- Image preprocessing
- Model prediction
- Result display
    """)


def example_batch_prediction():
    """
    Example: Batch prediction on multiple images.
    """
    print("\n" + "="*70)
    print("Example 6: Batch Predictions")
    print("="*70)
    
    print("""
To predict multiple images at once:

import os
from dog_breed_model import DogBreedClassifier
from PIL import Image
import numpy as np

# Load model
classifier = DogBreedClassifier(num_classes=10)
classifier.load_model('models/dog_breed_model.h5')

# Directory with images
image_dir = 'path/to/images'
image_files = [f for f in os.listdir(image_dir) 
               if f.endswith(('.jpg', '.png', '.jpeg'))]

# Predict each image
results = []
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    img = Image.open(img_path)
    img_array = np.array(img)
    
    breed, confidence, top_3 = classifier.predict(img_array)
    results.append({
        'file': img_file,
        'breed': breed,
        'confidence': confidence
    })
    
    print(f"{img_file}: {breed} ({confidence*100:.1f}%)")

# Save results to CSV
import csv
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'breed', 'confidence'])
    writer.writeheader()
    writer.writerows(results)
    """)


def main():
    """
    Run all examples.
    """
    print("\n" + "="*70)
    print("DOG BREED CLASSIFICATION - EXAMPLE USAGE")
    print("="*70)
    
    print("\nThis script demonstrates various ways to use the Dog Breed Classifier.")
    print("\nChoose an example to run:")
    print("1. Create and save a model")
    print("2. Load model and make predictions")
    print("3. Display model architecture")
    print("4. Show training workflow")
    print("5. Web app integration guide")
    print("6. Batch predictions guide")
    print("0. Run all examples")
    
    try:
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            example_create_and_save_model()
        elif choice == '2':
            example_load_and_predict()
        elif choice == '3':
            example_model_architecture()
        elif choice == '4':
            example_training_workflow()
        elif choice == '5':
            example_web_app_integration()
        elif choice == '6':
            example_batch_prediction()
        elif choice == '0':
            example_create_and_save_model()
            example_load_and_predict()
            example_model_architecture()
            example_training_workflow()
            example_web_app_integration()
            example_batch_prediction()
        else:
            print("\nInvalid choice. Please run again and select 0-6.")
            return
        
        print("\n" + "="*70)
        print("Example completed successfully!")
        print("="*70)
        print("\nFor more information, see:")
        print("- README.md for detailed documentation")
        print("- app.py for Flask web application")
        print("- dog_breed_model.py for model implementation")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")


if __name__ == '__main__':
    main()
