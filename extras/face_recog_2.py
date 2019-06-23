import numpy as np
import cv2


cam = cv2.VideoCapture(0) # number represents which cam to use. by default webcam is 0
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #create a haar-cascade object for face detection. haarcascade has features that it extracts. we are not extracting the features rn tho, just using the algorithm.  

data = [] #placeholder for storing data
ix = 0 #current frame number

while True: #we write an infinite loop now
    ret, frame = cam.read()
    if ret == True: #ret is a bool value to check if the cam is working or not
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert color to gray scale. open cv works on gray
        faces = face_cas.detectMultiScale(gray, 1.3, 5) #apply the haar-cascade to detect faces in the current frame, 1.3 and 5 are fine tunig parameters for haar cascade object
        # fro each face object we get, we have the corner coords x,y and width and height of face
        for(x,y,w,h) in faces:
            face_component = frame[y:y+h, x:x+w, :] #third colon means all rgb
            fc = cv2.resize(face_component, (50,50)) #resize the face img to 50x50x3
            
            if ix%10 == 0 and len(data) < 20: # capture imgs after ever 10 frames, only if total no of imgs is less than 20
                data.append(fc)
                #draw a rectangle for visualisation around the face in the img
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
          ix += 1 #increment current frame number
          cv2.imshow('frame', frame) #display frame
          if cv2.waitKey(1) == 27 or len(data) >= 20: #after ecery 1 ms it takes i/p. 27 is id for 'esc' key, other condition is if captures more than 20 imgs
            break
    else:
        print "error" #if cam is not working
cv2.destroyAllWindows()
data = np.asarray(data) #convert data in numpy format

print data.shape

np.save('face_01',data) #save data as a numpy matrix in encoded format
                
                 
                           
                
        
    else:
        print "error"