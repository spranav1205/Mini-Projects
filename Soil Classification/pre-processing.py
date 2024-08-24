import numpy as np
import pandas as pd
import os
import cv2 as cv2

#Number of bins of hues and saturation
bins1 = 10
bins2 = 10

df = pd.DataFrame(columns=np.arange(bins1 * bins2))
df["Soil type"] = None

#Each row in the feature array represents a particular hue or color.
def hsv_hist(image: np.ndarray, bins1: int, bins2 : int):
    (h,w,z) = np.shape(image)

    feature_array = np.zeros((bins1, bins2), dtype=np.uint32)

    for i in range (h//3,2*h//3):
        for j in range (w//3,2*w//3):
           pixel = image[i][j]
           hue = (pixel[0]*bins1)//360
           saturation = (pixel[1]*bins2)//256

           feature_array[hue][saturation] += 1

    feature_array = feature_array.flatten()
    return feature_array

directory = r"" #Path to your directory

#Place holder 
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    for filename_2 in os.listdir(filepath):
        if filename_2.endswith((".jpg", ".jpeg", ".png")):  
            image_path = os.path.join(filepath, filename_2)
            img = cv2.imread(image_path)
            image =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            feature = hsv_hist(image, bins1, bins2)
            new_row = np.append(feature,filename)
            df.loc[len(df)] = new_row


print(df.describe)
df.to_csv('soil_1.csv') #Convert the histograms to a CSV file




           
