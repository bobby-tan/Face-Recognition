import sys
import cv2
import face_recognition
import pickle

name=input("Enter the name of the person: ")
ref_id = input("enter id")

try:
    f = open("ref_name.pkl", "rb")
    ref_dictt = pickle.load(f)
    f.close()
except:
    ref_dictt = {}

ref_dictt[ref_id] = name

f=open("ref_name.pkl", "wb")
pickle.dump(ref_dictt, f)
f.close()

try:
    f = open("ref_emb.pkl", "rb")
    embed_dict = pickle.load(f)
    f.close()
except:
    embed_dict = {}

mode = input("Use webcam (w) or image file (f)? ").strip().lower()

def process_frame(frame, i):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if face_locations != []:
        full_face_locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in face_locations]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_frame, full_face_locations)[0]
        if ref_id in embed_dict:
            embed_dict[ref_id]+=[face_encoding]
        else:
            embed_dict[ref_id]=[face_encoding]
        print(f"Snap {i+1}/5 saved.")
        return True
    else:
        print("No face detected — try again.")
        return False

if mode == 'f':
    i = 0
    while i < 5:
        img_path = input(f"Enter image path for snap {i+1}/5: ").strip()
        frame = cv2.imread(img_path)
        if frame is None:
            print("Could not load image, try again.")
            continue
        if process_frame(frame, i):
            i += 1
else:
    state = {'snap': False, 'quit': False, 'frame_shape': None}

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not state['frame_shape']:
            return
        h, w = state['frame_shape']
        if 10 <= x <= 110 and h - 60 <= y <= h - 10:
            print("Quit clicked")
            state['quit'] = True
        if w - 110 <= x <= w - 10 and h - 60 <= y <= h - 10:
            print("Snap clicked")
            state['snap'] = True

    def draw_buttons(frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, h - 60), (110, h - 10), (0, 0, 200), -1)
        cv2.putText(frame, "Quit", (30, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame, (w - 110, h - 60), (w - 10, h - 10), (0, 200, 0), -1)
        cv2.putText(frame, "Snap", (w - 90, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.namedWindow("Capturing")
    cv2.setMouseCallback("Capturing", mouse_callback)

    for i in range(5):
        webcam = cv2.VideoCapture(0)
        state['snap'] = False
        state['quit'] = False
        while True:
            check, frame = webcam.read()
            state['frame_shape'] = frame.shape[:2]

            display = frame.copy()
            draw_buttons(display)
            cv2.imshow("Capturing", display)
            cv2.waitKey(20)

            if state['snap']:
                if process_frame(frame, i):
                    webcam.release()
                    cv2.waitKey(1)
                    cv2.destroyAllWindows()
                    cv2.namedWindow("Capturing")
                    cv2.setMouseCallback("Capturing", mouse_callback)
                    break
                state['snap'] = False

            elif state['quit']:
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

f=open("ref_embed.pkl","wb")
pickle.dump(embed_dict,f)
f.close()
