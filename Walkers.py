import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier('PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    if(frame is not None):
           #Convert Each Frame into Grayscale
           grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           # Pass frame to our body classifier
           bodies = body_classifier.detectMultiScale(frame, 1.9, 3)
    
          # Extract bounding boxes for any bodies identified
           for (x, y, w, h) in bodies:
               
                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 34, 20), 3)

           if cv2.waitKey(20) == 32: #32 is the Space Key
               break


 
cap.release()
cv2.destroyAllWindows()
