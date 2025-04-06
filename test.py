from deepface import DeepFace
# Compare two faces
result = DeepFace.verify("sachin1.jpeg", "sachin2.jpeg")

print("Are they same person?", result["verified"])

# check liveness of image
result1 = DeepFace.extract_faces("rohit2.jpeg", anti_spoofing=True)
print("liveness check", result1[0]['is_real'])