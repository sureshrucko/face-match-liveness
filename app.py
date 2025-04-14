from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import io
import os

app = Flask(__name__)

@app.route('/liveness', methods=['POST'])
def check_liveness():
    """
    API endpoint to check liveness of a face in an image.
    Accepts an image file as a buffer in the POST request.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image_file = request.files['image']

    try:
        # Convert the image buffer to a PIL Image
        image = Image.open(io.BytesIO(image_file.read()))
        
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
        print(result)
        return jsonify({"is_real": is_real})
    except Exception as e:
        # The temporary file is deleted even if an error occurs
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)