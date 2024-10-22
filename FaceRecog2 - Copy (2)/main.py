import os
import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Initialize Firebase Admin
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "",
    'storageBucket': ""
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Set the window to full screen
cv2.namedWindow("Webcam Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load the encoding file
print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
id = -1
scan_active = True  # Flag to control the scanning state
studentInfo = None  # Variable to store student information
unknown_text = "We do not know you"  # Message for unknown faces

while True:
    success, img = cap.read()
    
    # Check if the frame was captured successfully
    if not success:
        print("Failed to capture image from webcam. Please check your connection.")
        break  # Exit the loop if the webcam fails to capture an image
    
    # Resize and convert image for face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if scan_active and faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                id = studentIds[matchIndex]
                scan_active = False  # Stop scanning after a face is recognized

                # Get the Data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                # Prepare to display student info
                if studentInfo:
                    info_text = f"Student ID: {id}"
                    name_text = f"Name: {studentInfo['name']}"
                else:
                    info_text = f"Student ID: {id}, Name: Not Found"
                    name_text = ""

            else:
                # If no match is found, set the unknown message
                scan_active = False
                info_text = unknown_text
                name_text = ""

                # Optionally, you can add a small delay before showing the unknown message
                cv2.waitKey(1000)

                # Optionally, break out of the loop if you don't want to scan further
                break

        # Start countdown before displaying the information if recognized
        if studentInfo or unknown_text:
            countdown = 10  # Countdown time in seconds
            while countdown > 0:
                # Clear previous text
                success, img = cap.read()  # Capture a new frame
                if not success:
                    print("Failed to capture image from webcam during countdown.")
                    break  # Exit countdown if the camera fails

                img = cv2.resize(img, (640, 480))  # Resize if needed

                # Display the countdown
                cv2.putText(img, f"Displaying Info in: {countdown}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Webcam Feed", img)
                cv2.waitKey(1000)  # Wait for 1 second
                countdown -= 1

    # Overlay student information if scanning is inactive
    if not scan_active:
        # Clear previous texts by redrawing the webcam image
        success, img = cap.read()  # Capture a new frame
        if success:
            img = cv2.resize(img, (640, 480))  # Resize if needed
            cv2.putText(img, info_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # Smaller font size
            cv2.putText(img, name_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Smaller font size

            # Show information for 3 seconds
            cv2.imshow("Webcam Feed", img)
            cv2.waitKey(3000)  # Show information for 3 seconds

        # Reset for new scan
        scan_active = True
        id = -1
        studentInfo = None  # Reset studentInfo for the next scan

    # If scanning is still active, display the webcam feed
    if scan_active:
        cv2.imshow("Webcam Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
