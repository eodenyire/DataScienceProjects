"""
Example Usage Script for Traffic Sign Classifier

This script demonstrates how to use the TrafficSignClassifier class
for various tasks including model creation, training, and prediction.
"""

import os
from traffic_sign_classifier import TrafficSignClassifier, create_sample_model


def example_1_create_and_save_model():
    """
    Example 1: Create a new model and save it.
    """
    print("=" * 60)
    print("Example 1: Creating and Saving a Model")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TrafficSignClassifier(num_classes=43, img_size=(224, 224))
    
    # Build model (without pretrained weights for this example)
    print("\n📦 Building model...")
    classifier.build_model(use_pretrained=False)
    
    # Display model summary
    print("\n📊 Model Summary:")
    classifier.model.summary()
    
    # Save the model
    model_path = 'models/traffic_sign_model.h5'
    os.makedirs('models', exist_ok=True)
    classifier.save_model(model_path)
    
    print(f"\n✅ Model created and saved to {model_path}")


def example_2_load_and_predict():
    """
    Example 2: Load a saved model and make predictions.
    """
    print("\n" + "=" * 60)
    print("Example 2: Loading Model and Making Predictions")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TrafficSignClassifier()
    
    # Check if model exists
    model_path = 'models/traffic_sign_model.h5'
    if not os.path.exists(model_path):
        print("\n⚠️  No saved model found. Creating a sample model first...")
        create_sample_model(model_path)
    
    # Load the model
    print(f"\n📂 Loading model from {model_path}...")
    classifier.load_model(model_path)
    
    # Example prediction (you would replace this with an actual image path)
    image_path = 'path/to/traffic_sign.jpg'
    
    if os.path.exists(image_path):
        print(f"\n🔍 Predicting traffic sign in {image_path}...")
        result = classifier.predict(image_path)
        
        print("\n📊 Prediction Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_3'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
    else:
        print(f"\n⚠️  Image file not found: {image_path}")
        print("Please provide a valid traffic sign image path to test predictions.")


def example_3_train_model():
    """
    Example 3: Train a model on custom dataset.
    
    Note: This requires a properly structured dataset with subdirectories
    for each class containing training images.
    """
    print("\n" + "=" * 60)
    print("Example 3: Training a Model on Custom Dataset")
    print("=" * 60)
    
    # Check if training data exists
    train_data_dir = 'data/train'
    val_data_dir = 'data/validation'
    
    if not os.path.exists(train_data_dir):
        print(f"\n⚠️  Training data directory not found: {train_data_dir}")
        print("\nTo train a model, you need to:")
        print("1. Download the GTSRB dataset")
        print("2. Organize it into subdirectories (one per class)")
        print("3. Split into train/ and validation/ directories")
        print("\nDataset structure should look like:")
        print("data/")
        print("├── train/")
        print("│   ├── 0/  (class 0 images)")
        print("│   ├── 1/  (class 1 images)")
        print("│   └── ...")
        print("└── validation/")
        print("    ├── 0/")
        print("    ├── 1/")
        print("    └── ...")
        return
    
    # Initialize classifier
    classifier = TrafficSignClassifier(num_classes=43, img_size=(224, 224))
    
    # Build model with transfer learning
    print("\n📦 Building model with transfer learning...")
    classifier.build_model(use_pretrained=True)
    
    # Train the model
    print("\n🎓 Starting training...")
    print(f"Training data: {train_data_dir}")
    print(f"Validation data: {val_data_dir}")
    
    history = classifier.train_model(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        epochs=20,
        batch_size=32
    )
    
    # Save trained model
    model_path = 'models/trained_traffic_sign_model.h5'
    classifier.save_model(model_path)
    
    print(f"\n✅ Training complete! Model saved to {model_path}")


def example_4_class_names():
    """
    Example 4: Display all supported traffic sign classes.
    """
    print("\n" + "=" * 60)
    print("Example 4: Supported Traffic Sign Classes")
    print("=" * 60)
    
    classifier = TrafficSignClassifier()
    
    print(f"\n📋 Total Classes: {classifier.num_classes}")
    print("\nClass List:")
    print("-" * 60)
    
    for class_id, class_name in enumerate(classifier.class_names):
        print(f"{class_id:2d}. {class_name}")
    
    print("-" * 60)


def example_5_model_architecture():
    """
    Example 5: Display model architecture details.
    """
    print("\n" + "=" * 60)
    print("Example 5: Model Architecture Details")
    print("=" * 60)
    
    classifier = TrafficSignClassifier()
    classifier.build_model(use_pretrained=False)
    
    print("\n📐 Model Architecture:")
    print(f"Input Shape: {classifier.img_size} x 3 (RGB)")
    print(f"Output Classes: {classifier.num_classes}")
    print(f"\nBase Model: MobileNetV2 (Transfer Learning)")
    print("Custom Layers:")
    print("  - GlobalAveragePooling2D")
    print("  - Dropout(0.3)")
    print("  - Dense(256, activation='relu')")
    print("  - Dropout(0.3)")
    print("  - Dense(128, activation='relu')")
    print("  - Dropout(0.2)")
    print("  - Dense(43, activation='softmax')")
    
    print("\n📊 Model Summary:")
    classifier.model.summary()
    
    # Count parameters
    trainable_params = sum([layer.count_params() for layer in classifier.model.trainable_weights])
    non_trainable_params = sum([layer.count_params() for layer in classifier.model.non_trainable_weights])
    
    print(f"\n📈 Parameters:")
    print(f"Trainable: {trainable_params:,}")
    print(f"Non-trainable: {non_trainable_params:,}")
    print(f"Total: {trainable_params + non_trainable_params:,}")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 60)
    print("Traffic Sign Classifier - Example Usage")
    print("=" * 60)
    print("\nThis script demonstrates various ways to use the classifier:")
    print("1. Create and save a model")
    print("2. Load a model and make predictions")
    print("3. Train a model on custom dataset")
    print("4. View supported traffic sign classes")
    print("5. Display model architecture details")
    
    # Run examples
    try:
        # Example 1: Create model
        example_1_create_and_save_model()
        
        # Example 4: Show classes
        example_4_class_names()
        
        # Example 5: Architecture
        example_5_model_architecture()
        
        # Example 2: Load and predict
        example_2_load_and_predict()
        
        # Example 3: Training (if data available)
        # example_3_train_model()  # Uncomment to run training example
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
