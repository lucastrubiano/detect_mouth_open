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

# Define the codec and create VideoWriter object to save video to local
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # codec
# cv2.VideoWriter( filename, fourcc, fps, frameSize )
# out = cv2.VideoWriter("output.avi", fourcc, 7, (640, 480))

# Load a sample picture and learn how to recognize it.
# peter_image = face_recognition.load_image_file(
#     "peter.jpg"
# )  # replace peter.jpg with your own image !!
# peter_face_encoding = face_recognition.face_encodings(peter_image)[0]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face enqcodings in the frame of video
    face_locations = [face_recognition.face_locations(frame)[0]]
    print("face_locations: ")
    print(len(face_locations))

    top, right, bottom, left = face_locations[0]

    # Obtain a new frame with the face centered
    face_image = frame[top:bottom, left:right]
    
    face_encodings = face_recognition.face_encodings(
        face_image, None, num_jitters=1, model="small"
    )
    print("face_encodings: ")
    print(len(face_encodings))
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    print("face_landmarks_list: ")
    print(len(face_landmarks_list))

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Lucas", (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)


        # Display text for mouth open / close
        ret_mouth_open = is_mouth_open(face_landmarks)
        if ret_mouth_open is True:
            text = 'Mouth is open'
            print('Mouth is open')
        else:
            text = 'Open your mouth'
            print('Open your mouth')
        cv2.putText(frame, text, (left, top - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)
    # out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()