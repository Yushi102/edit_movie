"""
Test MediaPipe FaceMesh initialization
"""
import mediapipe as mp
import os

print("Testing MediaPipe FaceMesh initialization...")

# Get MediaPipe path
mp_path = os.path.dirname(mp.__file__)
print(f"MediaPipe path: {mp_path}")

# Check if model file exists
model_path = os.path.join(mp_path, 'modules', 'face_landmark', 'face_landmark_front_cpu.binarypb')
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Model path: {model_path}")

# Try to initialize FaceMesh
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    print("\n✅ FaceMesh initialized successfully!")
    face_mesh.close()
except Exception as e:
    print(f"\n❌ FaceMesh initialization failed: {e}")
    print(f"\nError type: {type(e).__name__}")
    
    # Try with different settings
    print("\nTrying with minimal settings...")
    try:
        face_mesh = mp_face_mesh.FaceMesh()
        print("✅ FaceMesh initialized with default settings!")
        face_mesh.close()
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
