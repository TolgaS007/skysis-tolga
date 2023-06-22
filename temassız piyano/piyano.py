import cv2  
import mediapipe as mp
import pygame

pygame.init()
pygame.mixer.init()

sounds = ["wav/c1.wav","wav/d1.wav","wav/e1.wav","wav/f1.wav","wav/g1.wav","wav/a1.wav","wav/b1.wav","wav/c2.wav",]

colors = [(255,0,0),(255,128,0),(255,255,0),(0,255,0),(0,255,255), (0,0,255),(128,0,255),(255,0,255)]

positions = [(0,50),(75,50),(150,50),(225,50),(300,50),(375,50),(450,50),(525,50)]
width = 75
height = 100

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
   
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
            for id,lm in enumerate(handlms.landmark):
                h,w,c= img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                for i in range(8):
                    x,y = positions[i]
                    color = colors[i]
                    cv2.rectangle(img, (x,y),(x+width, y+height),color,2)
                    if(x<cx and cx< (x+width) and y< cy and cy<(y+height)):
                        sound = pygame.mixer.Sound(sounds[i])
                        sound.play()
    img = cv2.flip(img,1)
    
    cv2.imshow("image", img)
    cv2.waitKey(1)