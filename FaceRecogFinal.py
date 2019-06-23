import numpy as np 
import cv2

#start cam and haar cascade
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #create a haar-cascade object for face detection. haarcascade has features that it extracts. we are not extracting the features rn tho, just using the algorithm.  
font = cv2.FONT_HERSHEY_SIMPLEX

f_01 = np.load('face_01.npy').reshape((20, 50*50*3)) #linearising into one matrix
f_02 = np.load('face_02.npy').reshape((12, 50*50*3))
f_03 = np.load('face_03.npy').reshape((20, 50*50*3))

print (f_01.shape, f_02.shape, f_03.shape)

names = {
    0: 'Pratyush',
    1: 'Ankita',
    2: 'Anju', 
}

labels = np.zeros((52,1))
labels[:20, : ] = 0.0
labels[20:32, :] = 1.0
labels[32:, :] = 2.0

data = np.concatenate([f_01, f_02, f_03])
print(data.shape,labels.shape)

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets,k=5): 
    m = train.shape[0]
    dist = []
    
    for ix in range(m): 
        dist.append(distance(x, train[ix]))
      
    dist = np.asarray(dist)
    # Lets us pick up top K values
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts = True)
    
    return counts[0][np.argmax(counts[1])]



while True:
   #get each frame
    ret, frame = cam.read()
 # convert to grayscale and get faces
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        # for each frame
        for(x,y,w,h) in faces:
            face_component = frame[y:y+h, x:x+w, :]
            fc = cv2.resize(face_component, (50,50))

            # after processing the image and rescaling
            # convert to linear vector using flatten()
            # and pass to knn along with data

            lab = knn(fc.flatten(), data, labels) #flatten converts a matrix to linear vector
            text = names[int(lab)] # convert this label to int and get corresponding name
            cv2.putText(frame, text, (x,y), font, 1, (255,255,0), 2) #onts width, colour etc

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break            
    else:
        print("error")