from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import time
import HandTracking as ht
import pyautogui

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_hand_tracking')
def start_hand_tracking():
    pTime = 0
    width = 640
    height = 480
    frameR = 100
    smoothening = 8
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    drag = False  # To track if drag mode is active

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = ht.handDetector(maxHands=1)
    screen_width, screen_height = pyautogui.size()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)

            # Move mouse pointer
            if fingers[0] == 1 and fingers[4] == 0:  # Thumb up, pinky down
                x1, y1 = lmlist[4][1:]  # Thumb tip
                x_screen = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                y_screen = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                curr_x = prev_x + (x_screen - prev_x) / smoothening
                curr_y = prev_y + (y_screen - prev_y) / smoothening

                pyautogui.moveTo(screen_width - curr_x, curr_y)
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

            # Left-click gesture (Thumb + Index finger close)
            if fingers[0] == 1 and fingers[1] == 1:
                length, img, lineInfo = detector.findDistance(4, 8, img)

                if length < 20:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.leftClick()

            # Right-click gesture (Index + Middle finger close)
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)

                if length < 20:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.rightClick()

            # Scroll up gesture (Only Index finger up)
            if fingers == [0, 1, 0, 0, 0]:  # Only Index finger is up
                pyautogui.scroll(20)  # Scroll up
                cv2.putText(img, "Scrolling Up", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # Scroll down gesture (Index and Pinky fingers up)
            if fingers == [0, 1, 0, 0, 1]:  # Index and Pinky fingers are up
                pyautogui.scroll(-20)  # Scroll down
                cv2.putText(img, "Scrolling Down", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            # Drag-and-Drop gesture (All fingers closed, i.e., palm closed)
            if fingers == [0, 0, 0, 0, 0]:  # All fingers are down
                if not drag:  # Start dragging if not already dragging
                    drag = True
                    pyautogui.mouseDown()
                cv2.putText(img, "Dragging", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
            else:
                if drag:  # Stop dragging if drag mode was active
                    drag = False
                    pyautogui.mouseUp()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        # Exit loop when 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "Hand tracking stopped"})

if __name__ == "__main__":
    app.run(debug=True)
