import cv2

url = "rtsp://ksp_explore:initial2023@115.85.94.50:550"
url2 = "rtsp://admin:Gateksp2024@192.168.2.106"
cap = cv2.VideoCapture(url2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()