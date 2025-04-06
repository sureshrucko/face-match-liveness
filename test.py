from deepface import DeepFace
"""
This script uses the DeepFace library to perform face verification and liveness detection.
Functions:
- DeepFace.verify: Compares two face images to determine if they belong to the same person.
- DeepFace.extract_faces: Extracts faces from an image and performs anti-spoofing (liveness detection).
Workflow:
1. Face Verification:
    - Compares two images ("sachin1.jpeg" and "sachin2.jpeg") to check if they belong to the same person.
    - Prints the result of the verification.
2. Liveness Detection:
    - Analyzes the image ("rohit2.jpeg") to determine if the face is real or spoofed.
    - Prints the result of the liveness check.
Dependencies:
- DeepFace library: Ensure it is installed and properly configured.
- Input images: Ensure the specified image files exist in the working directory.
Note:
- The `anti_spoofing` parameter in `DeepFace.extract_faces` enables liveness detection.
"""
# Compare two faces
result = DeepFace.verify("sachin1.jpeg", "sachin2.jpeg")

print("Are they same person?", result["verified"])

# check liveness of image
result1 = DeepFace.extract_faces("rohit2.jpeg", anti_spoofing=True)
print("liveness check", result1[0]['is_real'])