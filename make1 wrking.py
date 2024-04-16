from typing import List
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os


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
            return False, []
        locations = detections[0].location_data.relative_bounding_box
        start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin, locations.ymin, image_cols, image_rows)
        end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
        if (not start_point) or (not end_point):
            return False, []
        return True, image[start_point[1]:end_point[1], start_point[0]:end_point[0]]


def generate_embedding(cropped_image, bgr=False):
    if bgr:
        cropped_image = cropped_image[:, :, ::-1]
    height, width, _ = cropped_image.shape
    return face_recognition.face_encodings(cropped_image, known_face_locations=[(0, width, height, 0)])[0]


def capture_known_faces() -> List[dict]:
    known_faces = []
    detector = MpDetector()
    i = 0
    for person_dir in os.listdir('images'):
        person_path = os.path.join('images', person_dir)
        if not os.path.isdir(person_path):
            continue
        print(f'Loading images for person {i}')
        person_data = {'name': person_dir, 'embeddings': []}
        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = cv2.imread(image_path)
            face_detection_status, face_crop = detector.detect(image, True)
            if face_detection_status:
                cv2.imshow(f"Person {i+1}", face_crop)
                emb = generate_embedding(np.array(face_crop))
                person_data['embeddings'].append(emb)
        if person_data['embeddings']:
            known_faces.append(person_data)
            print(f'Done loading images for person {i}')
            i += 1
    cv2.destroyAllWindows()
    return known_faces


def main():
    known_faces_data = capture_known_faces()
    known_face_embeddings = []
    known_face_names = []
    for person_data in known_faces_data:
        name = person_data['name']
        embeddings = np.array(person_data['embeddings'])
        known_face_embeddings.append(embeddings)
        known_face_names.extend([name] * len(embeddings))
    known_face_embeddings = np.concatenate(known_face_embeddings)
    np.savez("face_embeddings.npz", embeddings=known_face_embeddings, names=known_face_names)


if __name__ == "__main__":
    main()

print("\nSUCCESSFUL 🤣")
