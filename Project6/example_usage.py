"""
Example usage of the Plant Disease Predictor module.

This script demonstrates how to use the PlantDiseasePredictor class
for building models and making predictions.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

from plant_disease_predictor import PlantDiseasePredictor, create_demo_model
import os


def example_1_build_model():
    """
    Example 1: Build a new model
    """
    print("=" * 60)
    print("Example 1: Building a Plant Disease Prediction Model")
    print("=" * 60)
    
    # Create predictor instance
    predictor = PlantDiseasePredictor()
    
    # Build the model
    model = predictor.build_model()
    
    print(f"\n✅ Model built successfully!")
    print(f"Number of disease classes: {predictor.num_classes}")
    print(f"Input image size: {predictor.img_size}")
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    return predictor


def example_2_predict_with_demo_model():
    """
    Example 2: Make predictions with a demo model
    Note: This requires an actual image file to work
    """
    print("\n" + "=" * 60)
    print("Example 2: Making Predictions (Demo)")
    print("=" * 60)
    
    # Create demo model
    predictor = create_demo_model()
    
    # Note: In a real scenario, you would have a trained model
    # For this example, we'll just show the structure
    print("\n📋 To make a prediction, you would use:")
    print("""
    # Load an image
    image_path = "path/to/plant_leaf.jpg"
    
    # Make prediction
    results = predictor.predict(image_path, top_k=3)
    
    # Access results
    top_prediction = results['top_prediction']
    print(f"Disease: {top_prediction['disease']}")
    print(f"Confidence: {top_prediction['confidence']:.2f}%")
    print(f"Description: {top_prediction['description']}")
    print(f"Treatment: {top_prediction['treatment']}")
    """)
    
    return predictor


def example_3_disease_classes():
    """
    Example 3: List all supported disease classes
    """
    print("\n" + "=" * 60)
    print("Example 3: Supported Disease Classes")
    print("=" * 60)
    
    predictor = PlantDiseasePredictor()
    
    print(f"\nTotal classes: {len(predictor.DISEASE_CLASSES)}\n")
    
    # Group by plant type
    plants = {}
    for disease in predictor.DISEASE_CLASSES:
        plant = disease.split('___')[0]
        if plant not in plants:
            plants[plant] = []
        plants[plant].append(disease)
    
    for plant, diseases in sorted(plants.items()):
        print(f"\n🌱 {plant}:")
        for disease in diseases:
            condition = disease.split('___')[1]
            print(f"   - {condition}")


def example_4_disease_information():
    """
    Example 4: Access disease information
    """
    print("\n" + "=" * 60)
    print("Example 4: Disease Information Lookup")
    print("=" * 60)
    
    predictor = PlantDiseasePredictor()
    
    # Example: Get information about Tomato Early Blight
    disease_name = 'Tomato___Early_blight'
    info = predictor.get_disease_info(disease_name)
    
    print(f"\n📋 Disease: {disease_name}")
    print(f"\nDescription:")
    print(f"   {info['description']}")
    print(f"\nTreatment:")
    print(f"   {info['treatment']}")
    
    # Another example: Healthy plant
    print("\n" + "-" * 60)
    disease_name = 'Tomato___healthy'
    info = predictor.get_disease_info(disease_name)
    
    print(f"\n📋 Disease: {disease_name}")
    print(f"\nDescription:")
    print(f"   {info['description']}")
    print(f"\nTreatment:")
    print(f"   {info['treatment']}")


def example_5_save_and_load_model():
    """
    Example 5: Save and load model
    """
    print("\n" + "=" * 60)
    print("Example 5: Saving and Loading Models")
    print("=" * 60)
    
    # Build a model
    print("\n1. Building a new model...")
    predictor = PlantDiseasePredictor()
    predictor.build_model()
    
    # Save the model
    model_path = "example_model.h5"
    print(f"\n2. Saving model to {model_path}...")
    try:
        predictor.save_model(model_path)
        print("   ✅ Model saved successfully!")
    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
    
    # Load the model
    print(f"\n3. Loading model from {model_path}...")
    try:
        new_predictor = PlantDiseasePredictor(model_path=model_path)
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\n4. Cleaned up example model file")


def example_6_preprocessing():
    """
    Example 6: Image preprocessing demonstration
    """
    print("\n" + "=" * 60)
    print("Example 6: Image Preprocessing")
    print("=" * 60)
    
    predictor = PlantDiseasePredictor()
    predictor.build_model()
    
    print("\n📸 Image Preprocessing Pipeline:")
    print("   1. Load image using PIL")
    print("   2. Convert to RGB color space")
    print(f"   3. Resize to {predictor.img_size}")
    print("   4. Convert to numpy array")
    print("   5. Expand dimensions for batch processing")
    print("   6. Apply MobileNetV2 preprocessing")
    print("   7. Ready for model inference!")
    
    print("\n💡 To preprocess an image:")
    print("""
    # Example code
    image_path = "plant_leaf.jpg"
    preprocessed = predictor.preprocess_image(image_path)
    print(f"Preprocessed shape: {preprocessed.shape}")
    # Output: (1, 224, 224, 3)
    """)


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "PLANT DISEASE PREDICTOR - EXAMPLES")
    print("=" * 80)
    print("\nAuthor: Emmanuel Odenyire")
    print("Email: eodenyire@gmail.com")
    print("\n" + "=" * 80)
    
    try:
        # Run examples
        example_1_build_model()
        example_2_predict_with_demo_model()
        example_3_disease_classes()
        example_4_disease_information()
        example_5_save_and_load_model()
        example_6_preprocessing()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
        print("\n💡 Next Steps:")
        print("   1. Prepare training data (PlantVillage dataset recommended)")
        print("   2. Train the model with actual data")
        print("   3. Save the trained model")
        print("   4. Use the Streamlit app for predictions")
        print("\n   Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
