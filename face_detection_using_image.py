import cv2
import cv2.data
import os

# Importing the haarcascades folder from the OpenCV library for face detection using a pre-trained model.
# The os.listdir() function converts all the available folders in cv2 into a list format.
print(os.listdir(cv2.data.haarcascades))
filename = "haarcascade_frontalface_default.xml"

# creating a path for our souce file using the specific haarcascade file from the print statement above.
# After finding the path, we append our source file to that path and store it in the 'path' variable.
print(cv2.data.haarcascades)
path = cv2.data.haarcascades + filename

# Reading the image using OpenCV
image = cv2.imread("photo.jpg")

# Using the CascadeClassifier to detect faces in the image using the source path.
# After detecting the face, it scales the image by 30% for 5 times to check if the face remains consistent.
# If a face is not detected in any of the 5 attempts, it will discard that detection.
# The print statement returns the coordinates of the detected faces form the detectMultiScale().
model = cv2.CascadeClassifier(path)
faces = model.detectMultiScale(image, 1.3, 5)
print(faces)

# For each detected face, we draw a box using the coordinates returned by detectMultiScale().
# We use the rectangle() function to draw the box on the image.
for face in faces:
    x1 = face[0]
    y1 = face[1]
    x2 = x1 + face[2]
    y2 = y1 + face[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 0], 3)

# After line 32, we have successfully created a box around the face and now display the image using imshow().
cv2.imshow("me", image)

# Freezing the window until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
