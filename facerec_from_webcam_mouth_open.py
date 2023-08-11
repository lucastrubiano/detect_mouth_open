"""
Detect a face in webcam video and check if mouth is open.
"""
import face_recognition
import cv2
from mouth_open_algorithm import get_lip_height, get_mouth_height


def is_mouth_open(face_landmarks):
    top_lip = face_landmarks["top_lip"]
    bottom_lip = face_landmarks["bottom_lip"]

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    # print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f'
    #       % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))

    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(frame)

    if face_locations:
        top, right, bottom, left = face_locations[0]

        # Obtain a new frame with the face centered
        face_image = frame[top:bottom, left:right]

        face_landmarks_list = face_recognition.face_landmarks(face_image)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_landmarks in zip(
            face_locations, face_landmarks_list
        ):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Display text for mouth open
            if is_mouth_open(face_landmarks):
                # text = 'Mouth is open'
                print("Mouth is open")

            # cv2.putText(frame, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
