from cvzone.HandTrackingModule import HandDetector #type: ignore
import cv2  #type: ignore
import numpy as np
import pyautogui
import time

# Parameters
width, height = 900, 720
gestureThreshold = 300
key_mapping = {
    "open_hand": "right",
    "fist": "left",
    "v_sign": "up",
    "ok_sign": "down"
}

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Get image frame
    success, img = cap.read()

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 1, 1, 1, 1]:  # open hand
                print("Next")
                pyautogui.press('right')
                time.sleep(3)
            elif fingers == [0, 0, 0, 0, 0]:  # fist
                print("Previous")
                pyautogui.press('left')
                time.sleep(3)
            elif fingers == [0, 1, 1, 0, 0]:  # v sign
                print("Zoomin")
                pyautogui.press('up')
                time.sleep(3)
            elif fingers == [1, 0, 0, 0, 0]:  # ok sign
                print("ZoomOut")
                pyautogui.press('down')
                time.sleep(3)

    cv2.imshow("Image", img)
 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
