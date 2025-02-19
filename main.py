import cv2
from ultralytics import YOLO

# Load model hasil training
model = YOLO("best.pt")

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi plat nomor pada frame
    results = model(frame)

    # Gambar bounding box pada frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bbox
            conf = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Kelas deteksi
            
            # Gambar kotak di sekitar plat nomor
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil di layar
    cv2.imshow("License Plate Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
