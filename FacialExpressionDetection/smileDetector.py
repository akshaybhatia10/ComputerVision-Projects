# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')


capture = cv2.VideoCapture(0)

while True:
    response, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 2)
        cropped_gray = gray[y:y+h, x:x+w]
        cropped_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(cropped_gray, 1.6, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(cropped_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    cv2.imshow('Smile Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()