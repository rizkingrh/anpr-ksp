import cv2
import os
import time
import re
import numpy as np
import mysql.connector
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load model hasil training
model = YOLO("best.pt")

# Buat folder untuk menyimpan gambar plat
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Waktu terakhir OCR dijalankan
last_ocr_time = time.time()

def connect_to_db():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        cursor = connection.cursor()
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS anpr_ksp")
        print("Database 'anpr_ksp' checked/created.")

        # Connect to the newly created or existing database
        connection.database = "anpr_ksp"

        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS histories (
            id INT AUTO_INCREMENT PRIMARY KEY,
            numberplate TEXT,
            time TIME,
            date DATE
        )
        """
        cursor.execute(create_table_query)
        print("Table 'histories' checked/created.")

        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        raise

def save_to_database(numberplate, time, date):
        """Save data to the MySQL database."""
        try:
            cursor = db_connection.cursor()
            query = """
                INSERT INTO histories (numberplate, time, date)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (numberplate, time, date))
            db_connection.commit()
            print(f"Data saved to database: {numberplate}, {time}, {date}")
        except mysql.connector.Error as err:
            print(f"Error saving to database: {err}")
            raise

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi plat nomor pada frame
    results = model(frame)
    db_connection = connect_to_db()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bbox
            conf = box.conf[0].item()  # Confidence score
            
            # Gambar bounding box pada plat nomor yang valid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Crop bagian plat nomor (ambil 2/3 bagian atas untuk menghindari tahun kendaraan)
            crop_height = int((y2 - y1) * 0.7)  # Ambil 70% bagian atas
            crop_width_offset = int((x2 - x1) * 0.03)  # Ambil 10% dari lebar untuk kanan & kiri
            plate_region = frame[y1:y1 + crop_height, x1 + crop_width_offset:x2 - crop_width_offset]
            
            # Menjadikan black and white
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            label_box = ocr.ocr(plate_region, cls=True)
            
            if label_box and label_box[0]:
                label_box = " ".join(res[1][0] for res in label_box[0])
                label_box = re.sub(r'[^A-Za-z0-9]+', '', label_box).upper()
                
            print(label_box)

            # Simpan gambar plat nomor ke lokal setiap 5 detik
            current_time = time.time()
            if current_time - last_ocr_time >= 5:
                timestamp = int(current_time)
                plate_filename = f"{output_folder}/plate_{timestamp}_{label_box}.jpg"
                cv2.imwrite(plate_filename, plate_region)
                print(f"Plat nomor tersimpan: {plate_filename}")

                # OCR
                plate_text = ocr.ocr(plate_filename, cls=True)
                if plate_text and plate_text[0]:
                    plate_text = " ".join([res[1][0] for res in plate_text[0]])
                    plate_text = re.sub(r'[^A-Za-z0-9]+', '', plate_text).upper()

                # Simpan hasil OCR ke file teks
                text_filename = f"{output_folder}/plate_numbers.txt"
                with open(text_filename, "a") as text_file:
                    text_file.write(f"{plate_text}\n")
                
                current_time_db = datetime.now()
                label_box_db = str(label_box)
                save_to_database(
                    label_box_db,
                    current_time_db.strftime("%H:%M:%S"),
                    current_time_db.strftime("%Y-%m-%d")
                )

                print(f"Plat nomor terbaca: {plate_text}")

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

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
