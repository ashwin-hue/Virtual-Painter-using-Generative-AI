import cv2
import numpy as np
import mediapipe as mp
import os
import time
import PIL
import google.generativeai as genai
import PIL.Image
import pyautogui
from PIL import Image
API_KEY = # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Hand Tracking Module
class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) != 0:
            # Thumb
            if self.lmList[4][1] > self.lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other fingers
            for tipId in [8, 12, 16, 20]:
                if self.lmList[tipId][2] < self.lmList[tipId - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


# Virtual Painter
def virtual_painter():
    # Parameters
    brush_thickness = 15
    eraser_thickness = 50
    folder_path = r"D:\PY PRO\AI_Virtual_Painter_2.0-main\Header"

    # Load header images
    overlay_list = []
    for im in os.listdir(folder_path):
        img_path = os.path.join(folder_path, im)
        img = cv2.imread(img_path)
        if img is not None:
            overlay_list.append(cv2.resize(img, (1280, 125)))
    if not overlay_list:
        raise ValueError("No valid images found in the Header folder!")

    header = overlay_list[0]
    draw_color = (255, 0, 255)  # Default color (purple)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera. Please check your camera setup.")

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    xp, yp = 0, 0
    img_canvas = np.zeros((720, 1280, 3), np.uint8)

    last_print_time = time.time() - 30  # Initialize to allow detection immediately
    count = 0
    count_reset_time = None

    while True:
        # Step 1: Capture frame
        success, img = cap.read()
        if not success:
            print("Failed to capture video frame.")
            break
        img = cv2.flip(img, 1)

        # Step 2: Find hand landmarks
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)

        if lm_list:
            # Tip of the index and middle fingers
            x1, y1 = lm_list[8][1:]  # Index finger
            x2, y2 = lm_list[12][1:]  # Middle finger

            # Step 3: Check which fingers are up
            fingers = detector.fingersUp()

            # Check for "5" condition
            current_time = time.time()
            if all(f == 1 for f in fingers) and (current_time - last_print_time >= 30):
                count += 1
                print(f"5 detected! Count: {count}")
                last_print_time = current_time
                count_reset_time = current_time

            # Reset count after 5 seconds
            if count_reset_time and (current_time - count_reset_time >= 15):
                print("Count reset to 0")
                count = 0
                count_reset_time = None
            if count == 1:
                image = pyautogui.screenshot()
                image.save("image1.png")
                opened_image = Image.open("image1.png")
                model = genai.GenerativeModel(model_name="gemini-1.5-pro")
                response = model.generate_content([opened_image, "analyise the image and give me short discription, i there is any math euqation solve and give me the answers"])
                print(response.text)

            # Step 4: Selection mode - Two fingers up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 125:  # Check for header selection
                    if 250 < x1 < 450:
                        header = overlay_list[0]
                        draw_color = (255, 0, 255)  # Purple
                    elif 550 < x1 < 750:
                        header = overlay_list[1]
                        draw_color = (255, 0, 0)  # Blue
                    elif 800 < x1 < 950:
                        header = overlay_list[2]
                        draw_color = (0, 255, 0)  # Green
                    elif 1050 < x1 < 1200:
                        header = overlay_list[3]
                        draw_color = (0, 0, 0)  # Eraser

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

            # Step 5: Drawing mode - Index finger up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Draw on the canvas
                if draw_color == (0, 0, 0):  # Eraser
                    cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

                xp, yp = x1, y1

        # Combine the canvas and the current frame
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        # Add the header to the frame
        img[0:125, 0:1280] = header

        cv2.imshow("Virtual Painter", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the virtual painter
if __name__ == "__main__":
    virtual_painter()
