import cv2
import os
import time
from ultralytics import YOLO
import easyocr

# Inisialisasi EasyOCR Reader
reader = easyocr.Reader(['en'])

# Load model hasil training
model = YOLO("best.pt")  # Sesuaikan dengan modelmu

# Buat folder untuk menyimpan gambar plat
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi plat nomor pada frame
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bbox
            conf = box.conf[0].item()  # Confidence score
            
            # Hitung rasio lebar vs tinggi
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height  # Rasio lebar terhadap tinggi

            # Filter berdasarkan ukuran rasio (hindari area bukan plat)
            if aspect_ratio < 2.0:
                continue  

            # Gambar bounding box pada plat nomor yang valid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop bagian plat nomor
            plate_region = frame[y1:y2, x1:x2]
            
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

            # Simpan gambar plat nomor ke lokal
            timestamp = int(time.time())  # Gunakan timestamp untuk unik
            plate_filename = f"{output_folder}/plate_{timestamp}.jpg"
            cv2.imwrite(plate_filename, plate_region)
            print(f"Plat nomor tersimpan: {plate_filename}")

            # Gunakan EasyOCR untuk membaca plat nomor dari file yang disimpan
            plate_text = reader.readtext(plate_filename, detail=0)
            plate_text = " ".join(plate_text) if plate_text else "Tidak terbaca"

            # Simpan hasil OCR ke file teks
            text_filename = f"{output_folder}/plate_numbers.txt"
            with open(text_filename, "a") as text_file:
                text_file.write(f"{plate_text}\n")

            print(f"Plat nomor terbaca: {plate_text}")

            # Tampilkan teks di layar
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan hasil di layar
    cv2.imshow("License Plate Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
