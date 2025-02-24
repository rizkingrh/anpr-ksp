import cv2
import os
import time
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from fast_plate_ocr import ONNXPlateRecognizer

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load model hasil training
model = YOLO("best.pt")  # Sesuaikan dengan modelmu

# Buat folder untuk menyimpan gambar plat
output_folder = "plates"
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Waktu terakhir OCR dijalankan
last_ocr_time = time.time()
label_box = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi plat nomor pada frame
    results = model(frame)

    # Tampilkan teks "Plate Detection" di kiri atas layar
    cv2.putText(frame, "Number Plate:", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            label_box = ocr.ocr(plate_region, cls=True)
            # label_box = label_box[0]
            # label_box = re.sub(r'[^A-Za-z0-9 ]+', '', label_box).upper()
            # plate_regex = r'^[A-Za-z]{1,2}\d{1,4}[A-Za-z]{0,3}$'
            if label_box and label_box[0]:
                label_box = " ".join([res[1][0] for res in label_box[0]])
                label_box = re.sub(r'[^A-Za-z0-9]+', '', label_box).upper()
            print(label_box)
            
            # match = re.match(plate_regex, label_box)
            # if match:
            #     print(f"Plat nomor valid: {label_box}")
            # else:
            #     print("Plat nomor tidak valid")
            # label_box = " ".join(label_box) if label_box else "Tidak terbaca"
            # label_box = re.sub(r'[^A-Za-z0-9 ]+', '', label_box).upper()

            # Simpan gambar plat nomor ke lokal setiap 5 detik
            current_time = time.time()
            if current_time - last_ocr_time >= 5:
                timestamp = int(current_time)
                plate_filename = f"{output_folder}/plate_{timestamp}.jpg"
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

                print(f"Plat nomor terbaca: {plate_text}")

                # Update waktu terakhir OCR
                last_ocr_time = current_time

            # Tampilkan teks di layar
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, str(label_box), (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan hasil di layar
    cv2.imshow("License Plate Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
