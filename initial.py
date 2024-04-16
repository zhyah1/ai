import cv2
import numpy as np
import face_recognition
import mediapipe as mp

class MpDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.buffered_detections = None
        self.buffer_frame_count = 0
        self.detection_interval = 5  # Detect faces every 5 frames

    def detect(self, image, bgr=False):
        if bgr:
            image = image[:, :, ::-1]
        image_rows, image_cols, _ = image.shape
        
        # Only perform face detection every few frames
        self.buffer_frame_count += 1
        if self.buffer_frame_count % self.detection_interval == 0:
            detections = self.detector.process(image).detections
            if detections:
                self.buffered_detections = detections
        
        face_crops, start_points = [], []
        if self.buffered_detections:
            for detection in self.buffered_detections:
                locations = detection.location_data.relative_bounding_box
                start_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    locations.xmin, locations.ymin, image_cols, image_rows)
                end_point = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    locations.xmin + locations.width, locations.ymin + locations.height, image_cols, image_rows)
                if start_point and end_point:
                    face_crops.append(image[start_point[1]:end_point[1], start_point[0]:end_point[0]])
                    start_points.append((start_point[0], start_point[1]))

        return len(face_crops) > 0, face_crops, start_points


def generate_embedding(cropped_image, bgr=False):
    if bgr:
        cropped_image = cropped_image[:, :, ::-1]
    height, width, _ = cropped_image.shape
    return face_recognition.face_encodings(cropped_image, known_face_locations=[(0, width, height, 0)])[0]


def load_known_faces():
    face_embeddings_path = "face_embeddings.npz"

    known_face_data = np.load(face_embeddings_path)
    known_face_embeddings = known_face_data['embeddings']
    known_face_names = known_face_data['names']
    return known_face_embeddings, known_face_names


def identify_faces(known_face_embeddings, known_face_names, face_crops):
    recognized_faces = []

    for face_crop in face_crops:
        current_face_embedding = generate_embedding(np.array(face_crop))
        face_distances = face_recognition.face_distance(known_face_embeddings, current_face_embedding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]

        if min_distance < 0.5:
            recognized_faces.append((known_face_names[min_distance_index]))
        else:
            recognized_faces.append(('Unknown'))

    return recognized_faces


def display_faces(recognized_faces, frame):
    for person_name in recognized_faces:
        print(person_name)

def main():
    known_face_embeddings, known_face_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    detector = MpDetector()
    face_crops = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_detection_status, new_face_crops, start_points = detector.detect(frame, True)
        if face_detection_status:
            face_crops = new_face_crops  # Update the detected faces buffer

        if len(face_crops) > 0:
            recognized_faces = identify_faces(known_face_embeddings, known_face_names, face_crops)
            display_faces(recognized_faces, frame)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
