import cv2
import face_recognition
from imutils.video import FPS
import imutils

cap = cv2.VideoCapture("newGirl.mp4")
fps = FPS().start()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize variables
face_locations = []
frame_counter = 0

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    frame_counter += 1
    # This condition prevents from infinite looping
    # incase video ends.
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    rgb_frame = imutils.resize(frame, width=450)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    print(frame_counter)
    # Display the resulting image
    cv2.imshow('Video Face Detection', frame)

    # display a piece of text to the frame (so we can benchmark
    # fairly against the fast method)
    cv2.putText(frame, "Slow Method", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # show the frame and update the FPS counter
    fps.update()

    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
