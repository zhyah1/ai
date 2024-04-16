from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os
from collections import Counter



####LOADINGSCREEN
from tqdm import tqdm
import pyttsx3


#TEXT TO SPEECH
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()


# person_names = []



class MpDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    def detect(self, image, bgr=False):
        if bgr:
            image = image[:, :, ::-1]
        image_rows, image_cols, _ = image.shape
        detections = self.detector.process(image).detections
        if not detections:
            return False, None, None, None
        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
        if (not start_point) or (not end_point):
            return False, None, None, None
        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]], start_point[0], start_point[1]


def generate_embedding(cropped_image, bgr=False):
    if bgr:
        cropped_image = cropped_image[:, :, ::-1]
    height, width, _ = cropped_image.shape
    return face_recognition.face_encodings(cropped_image, known_face_locations=[(0, width, height, 0)])[0]


def load_known_faces():
    #path
    face_embeddings_path = "Face_recognition/MIX/face_embeddings.npz"


    known_face_data = np.load(face_embeddings_path)
    known_face_embeddings = known_face_data['embeddings']
    known_face_names = known_face_data['names']
    return known_face_embeddings, known_face_names


def identify_faces(known_face_embeddings, known_face_names, image):
    detector = MpDetector()
    face_detection_status, face_crop, start_x, start_y = detector.detect(image, True)
    if face_detection_status:
        current_face_embedding = generate_embedding(np.array(face_crop))
        face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]
        if min_distance < 0.5:
            # Face recognized as a known person
            return True, known_face_names[min_distance_index], start_x, start_y
        else:
            # Unknown face
            # print ("Unknown")
            unk = "Unknown"
            return True, unk, None, None
    else:
        # No face detected
        print("Noface" , end = "r")
        return False, None, None, None

#COUNT PART
def find_largest_repeating(names):
    counts = Counter(names)
    print(counts)
    max_name, max_count = counts.most_common(1)[0]
    print(max_name,max_count)
    
    if max_name == 'Unknown' and max_count == 20:
        print ("\nUNKNOWN PERSON Welcome\n")
        # text_to_speech("UNKNOWN PERSON")
        return(None)
    # Check if the highest count is greater than 15
    if max_count > 15 and max_name != 'Unknown':
        accuracyrate = max_count * 5
        print(f"\n\nPerson Identified as : '{max_name}' With Accuracy {accuracyrate} %.\n")
        
        # text_to_speech(f"Person Identified as :{max_name} With Accuracy {accuracyrate} Percentage")
        return (max_name)
        # exit()
    else:
        print("\n\nPerson Unidentified-----Please Come Closer :\n")
        text_to_speech("Person Unidentified-----Please Come Closer :")
        return ("interrupt")



#Multiple face Check
# Function to check multiple people looking into the camera

def detect_faces(frame):
    # Load the pre-trained Haar Cascade Classifier for face detection
    harpath = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + harpath)

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return len(faces)






def main1():
    known_face_embeddings, known_face_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    prev_x, prev_y = None, None
    person_names = []
    i = 0 
    while i < 20:
        ret, frame = cap.read()
        if not ret:
            break
        # num_persons = detect_faces(frame)
        # print(num_persons, " no of persons")
        face_recognition_status, person_name, start_x, start_y = identify_faces(
            known_face_embeddings, known_face_names, frame)
        
        if face_recognition_status:
            # Known person recognized
            #print("Person identified:", person_name )
            person_names.append(person_name)
            i = i + 1
            # print(i)
            # You can perform further actions here, such as displaying the person's name, etc.

        prev_x, prev_y = start_x, start_y

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"Recognizing....: {i*5} %", end="\r")
    num_persons = detect_faces(frame)
    print(num_persons, " no of persons")
    if num_persons > 1:
        # print("multiple")
        cap.release()
        cv2.destroyAllWindows()
        
        return("multiple")
        
    person = find_largest_repeating(person_names)
    if person == "interrupt":
            cap.release()
            cv2.destroyAllWindows()
            return main1()
        
    cap.release()
    cv2.destroyAllWindows()
    return person


if __name__ == "__main__":
    main1()
# print(person_names)
print("\nFingers Are Crossed....🤞")
