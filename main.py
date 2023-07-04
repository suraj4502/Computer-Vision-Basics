import cv2
import numpy as np


## TO show a image
# img = cv2.imread('Resources/me.png')

# cv2.imshow("me",img)
# cv2.waitKey(0)


## To show a video

# cap = cv2.VideoCapture(r'C:\Users\91775\Desktop\Computer Vision\Resources\demo.mp4')

# if not cap.isOpened():
#     print("Error opening video file")

# while cap.isOpened():
    
#     ret, frame = cap.read()

#     if ret:
#         cv2.imshow('Video', frame)

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()
    
    
    
## Taking Camera Input

# cap = cv2.VideoCapture(0)

# cap.set(3,640)  # setting width
# cap.set(4,480)  # setting Height
# cap.set(10,100) # setting contrast

# while True:
#     succes, img =  cap.read()
#     cv2.imshow("video",img)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
    
## Performing operations on Image:

# img = cv2.imread('Resources/mee.png')
# kernal = np.ones((5,5),np.uint8)

# Grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur_Img = cv2.GaussianBlur(Grey_img,(7,7),0)
# edge_detection = cv2.Canny(img,150,200)
# dilation = cv2.dilate(edge_detection,kernel=kernal,iterations=1)
# erosion = cv2.erode(dilation,kernal, 1)


# cv2.imshow("grey",Grey_img)
# cv2.imshow("blur",Blur_Img)
# cv2.imshow("edges",edge_detection)
# cv2.imshow("Dilation",dilation)
# cv2.imshow("erosion",erosion)
# cv2.waitKey(0)
# cv2.waitKey(0)


## CROPPING AND RESIZING
# img = cv2.imread('Resources/mee.png')
# # print(img.shape)  # (480,359,3) (height,width,colorChannels)

# resized_image = cv2.resize(img,(240,180))  # width, height

# cropped_image = img[0:300,0:250]           # height , width

# cv2.imshow("Original",img)
# cv2.imshow("Resized",resized_image)
# cv2.imshow("Cropped",cropped_image)
# cv2.waitKey(0)


## Shapes and text on image

# img = np.zeros((512,512,3),np.uint8)

# # print(img)
# # img[:] =255,0,0
# # img[100:200,200:300]= 0,255,0
# # img[300:400,100:200]= 0,0,255
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,255),3)
# cv2.rectangle(img,(50,100),(200,400),(255,255,0),2)# width , Height
# cv2.circle(img,(300,200),30,(255,50,0),5)
# cv2.imshow("ZEROS",img)
# cv2.waitKey(0)


## Image Stacking

# img = cv2.imread('Resources/mee.png')
# vertical = np.vstack((img,img))
# horizontal = np.hstack((img,img))

# cv2.imshow("Vertical",vertical)
# cv2.imshow("Horizontal",horizontal)
# cv2.waitKey(0)


## FACE Detection

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
img = cv2.imread('Resources/mee.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(img_gray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow("Result",img)
cv2.waitKey(0)