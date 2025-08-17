import cv2
import mediapipe as mp
import time
class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.Hands
        self.hands=self.mpHands.Hands(self,mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
    def FindHands(self,img,draw=True):
        imgRgb=cv2.cvtColor(cv2.COLOR_BGR2RGB)
        self.resultss=self.hands.process(imgRgb)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        return img
    def findposition(self,img,handNO=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNO]
            for id,lm in enumerate(myhand.landmarks):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmlist
    def main(self):
        ptime=0
        ctime=0
        cap=cv2.VideoCapture(1)
        detector=HandDetector()
        while True:
            success,img=cap.read()
            img=detector.FindHands(img)
            lmlist=detector.findposition(img)
            if len(lmlist)!=0:
                print(lmlist[4])
            ctime=time.time()
            fps=1/(ctime-ptime)
            ptime=ctime
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.imshow("Image:",img)
            cv2.waitKey(1)

    if __name__=='--main__':
        main()