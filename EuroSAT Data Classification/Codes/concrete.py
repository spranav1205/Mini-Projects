import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

path = r"" #Path to your satellite image that includes highways

img = cv2.imread(path)
image = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
'''
lowerBound = np.array([105, 0, 0])
upperBound = np.array([[150, 110, 150]])

mask = cv2.inRange(hsv,lowerBound, upperBound)
result = cv2.bitwise_and(image, image, mask= mask)

cv2.imwrite('road_mask.png', result)
'''

def nothing(x):
    pass

cv2.namedWindow('Trackbars')

# Create trackbars for adjusting HSV ranges
cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, nothing)

while True:
    # Get the current positions of the trackbars
    h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')

    # Define the lower and upper bounds for the HSV range
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Create a mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    result = cv2.bitwise_and(image, image, mask= mask)

    # Display the mask
    cv2.imshow('Mask', result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()