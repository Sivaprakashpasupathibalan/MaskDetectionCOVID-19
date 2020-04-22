import cv2
import os

DIR = r'F:\IMAGE\WITHOUT_MASKS'

# Load the cascade for front face and nose
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
num = 0   #count variable for images
# To capture the video from camera
cap = cv2.VideoCapture(0)

while True:
    #video is returned in frames as images
    #ret, mask_img = cap.read()
    mask_img = cv2.imread('F:\IMAGE\sample1.jpg')
    # Convert to grayscale
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    #Detect the faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1,4)
    for (x,y,w,h) in faces:
        mask_img = cv2.rectangle(mask_img,(x,y),(x+w,y+h),(255,0,0),2)      # Drawing the rectangle on the face
        Nose_detect = gray[y:y+h, x:x+w]                                    #checks for the nose in the face
        Nose_colour = mask_img[y:y+h, x:x+w]              
        noses = nose_cascade.detectMultiScale(Nose_detect,1.1,1)                  #checks the noses and draws the rectangle
        for (ex,ey,ew,eh) in noses:
            cv2.rectangle(Nose_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            crop_img = mask_img[y:y+h, x:x+w]
            os.chdir(DIR)
            global num
            num = num + 1
            cv2.imwrite("mask"+str(num)+".jpg", crop_img) 
            cv2.imshow("CROP",crop_img)
    #prints the image in window
    cv2.imshow("OUTPUT",mask_img)
    #code to close the application and destroy the windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
