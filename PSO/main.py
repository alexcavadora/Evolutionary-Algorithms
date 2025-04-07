import os
from PIL import Image
import numpy as np
from image_replicator import ImageReplicator

def preprocess_image(input_image_path):
    """Preprocess the image: load, convert to grayscale, and save."""
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, input_image_path)
    
    try:
        # Read image
        image = Image.open(image_path)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Save the grayscale image
        output_path = os.path.join(script_dir, "image_bw.png")
        image.save(output_path)
        
        print(f"Preprocessed image saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def main():
    # Input image path
    INPUT_IMAGE = "imperial-japanese-rising-sun-flag-uhd-4k-wallpaper.jpg"
    
    # Preprocess the image
    preprocessed_image = preprocess_image(INPUT_IMAGE)
    
    if preprocessed_image:
        # Create the replicator with optimized parameters
        replicator = ImageReplicator(
            image_path=preprocessed_image,
            output_path="output_images",
            max_strokes=10000,  # Increase for better quality
            section_size=15,
            scale_factor=0.2  # Scale down for faster processing
        )
        
        # Start the replication process
        print("Starting image replication...")
        replicator.replicate_image()
        print("Replication complete!")

if __name__ == "__main__":
    main()