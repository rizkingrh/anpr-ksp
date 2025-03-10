import cv2
import os
import time
import re
import numpy as np
import mysql.connector
import requests
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load model hasil training
model = YOLO("best.pt")

# Buat folder untuk menyimpan gambar plat
output_folder = "plates"
API_URL = "http://192.168.30.71:8000/api/store"
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Waktu terakhir OCR dijalankan
last_ocr_time = time.time()

def send_to_api(label_box, image_path):
    data = {
        "numberplate": str(label_box),  # Data plat nomor
    }

    files = {
        "image": open(image_path, "rb")
    }

    response = requests.post(API_URL, data=data, files=files)

    if response.status_code == 201:
        print("Data berhasil dikirim ke API!")
    else:
        print("Gagal mengirim data!")

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
            
            # Gambar bounding box pada plat nomor yang valid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Crop bagian plat nomor
            crop_height = int((y2 - y1) * 0.7)  # Ambil 70% bagian atas
            crop_width_offset = int((x2 - x1) * 0.03)  # Ambil 10% dari lebar untuk kanan & kiri
            plate_region = frame[y1:y1 + crop_height, x1 + crop_width_offset:x2 - crop_width_offset]
            
            # Menjadikan black and white
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            label_box = ocr.ocr(plate_region, cls=True)
            
            if label_box and label_box[0]:
                label_box = " ".join(res[1][0] for res in label_box[0])
                label_box = re.sub(r'[^A-Za-z0-9]+', '', label_box).upper()

            # Simpan gambar plat nomor ke lokal setiap 5 detik
            current_time = time.time()
            if current_time - last_ocr_time >= 5:
                timestamp = int(current_time)
                plate_filename = f"{output_folder}/plate.jpg"
                cv2.imwrite(plate_filename, plate_region)
                
                send_to_api(label_box, plate_filename)
                os.remove(plate_filename)

                # Update waktu terakhir OCR
                last_ocr_time = current_time

            # Tampilkan teks di layar
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, str(label_box), (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan teks "Plate Detection" di kiri atas layar
    cv2.putText(frame, "Number Plate:", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan hasil di layar
    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
