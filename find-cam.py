from cv2_enumerate_cameras import enumerate_cameras

print("Scanning for cameras...")
for camera in enumerate_cameras():
    print(f"Index: {camera.index} | Name: {camera.name}")