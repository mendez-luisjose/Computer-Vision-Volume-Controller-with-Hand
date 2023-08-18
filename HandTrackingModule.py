import cv2
import mediapipe as mp
import time
import math

class handDetector() :
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5) :
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True) :
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if (self.results.multi_hand_landmarks) :
            for handLms in self.results.multi_hand_landmarks :
                if draw :
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNumber=0, draw=True) :
        xList = []
        yList = []
        bbox = []
        lmList = []

        if (self.results.multi_hand_landmarks) :
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark) :
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                lmList.append([id, cx, cy])

                if (draw) :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)      

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)  

            bbox = xmin, ymin, xmax, ymax

            if draw :
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (255, 0, 0), 2)

        return lmList, bbox
    
    def findDistance(self, p1, p2, img, lmList, draw=True) :
        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]

        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw :
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
    
    def fingersUp(self, lmList):
        fingers = []
        # Thumb
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main() :
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True :
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if (len(lmList) != 0) :
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 255), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__" :
    main()