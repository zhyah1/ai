import os
import cv2
import dlib


def create_person_directory(person_name):
    directory = os.path.join("images", person_name.lower().replace(" ", "_"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def detect_face(frame):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return len(faces) > 0


def capture_images(directory):
    cap = cv2.VideoCapture(0)

    count = 0
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Captured Image", frame)

        if detect_face(frame):
            filename = os.path.join(directory, f"image{count+1}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"Taking Image {count+1}\r" )
        else:
            print("waiting For Person.......\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    person_name = input("Enter the person's name: ")
    directory = create_person_directory(person_name)
    capture_images(directory)


if __name__ == "__main__":
    main()
