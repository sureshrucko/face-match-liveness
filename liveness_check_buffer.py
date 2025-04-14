from deepface import DeepFace
from PIL import Image
import io
import os
import sys

def check_liveness(image_buffer):
    """
    Perform liveness detection on the given image buffer.
    """
    try:
        # Convert the image buffer to a PIL Image
        image = Image.open(io.BytesIO(image_buffer))
        
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
    # Read the image buffer from stdin
    image_buffer = sys.stdin.buffer.read()
    result = check_liveness(image_buffer)
    print(result)