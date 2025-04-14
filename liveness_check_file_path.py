from deepface import DeepFace
from PIL import Image
import io
import os
import sys

def check_liveness(image_path):
    """
    Perform liveness detection on the given image file.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Convert the image to RGB mode if it's not already in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save the image temporarily (DeepFace requires a file path)
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path, format="JPEG")

        # Perform liveness detection
        result = DeepFace.extract_faces(temp_image_path, anti_spoofing=True)
        is_real = result[0]['is_real'] if result else False

        # Delete the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {"is_real": is_real}
    except Exception as e:
        # Ensure the temporary file is deleted even if an error occurs
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python liveness_check.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    result = check_liveness(image_path)
    print(result)