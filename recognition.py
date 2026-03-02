import face_recognition
import cv2
import numpy as np
import glob
import pickle

f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)
f.close()
f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)
f.close()

print("Reference dictionary loaded: ", ref_dictt)

known_face_encodings = []
known_face_ids = []
for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_ids += [ref_id]

def recognise(frame, face_locations, face_ids):
    for (top_s, right, bottom, left), ref_id in zip(face_locations, face_ids):
        top_s *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        display_name = ref_dictt[ref_id] if ref_id in ref_dictt else "Unknown"
        cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def detect_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_ids = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        matched_id = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            matched_id = known_face_ids[best_match_index]
        print("face detected with id: ", matched_id)
        face_ids.append(matched_id)
    return face_locations, face_ids

mode = input("Use webcam (w) or image file (f)? ").strip().lower()

if mode == 'f':
    img_path = input("Enter image path: ").strip()
    frame = cv2.imread(img_path)
    face_locations, face_ids = detect_faces(frame)
    recognise(frame, face_locations, face_ids)
    cv2.imshow('Video', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_ids = []
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        print("frame captured")
        if process_this_frame:
            face_locations, face_ids = detect_faces(frame)
        process_this_frame = not process_this_frame
        recognise(frame, face_locations, face_ids)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
