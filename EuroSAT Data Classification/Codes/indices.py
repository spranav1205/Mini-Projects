import numpy as np
import rasterio
from rasterio.errors import RasterioError
import cv2 as cv2
import pandas as pd
import os

bins1 = 5 #Hue
bins2 = 5 #Saturation

def normalize_to_uint8(band):
    band_min, band_max = np.min(band), np.max(band)
    band_normalized = 255 * (band - band_min) / (band_max - band_min)
    return band_normalized.astype(np.uint8)

#Vegetation
def ndvi (nir: np.ndarray, RED: np.ndarray):

    if nir.shape != RED.shape:
        raise ValueError("Both arrays must have the same shape")
    (h,w) = np.shape(nir)
    NDVI = np.empty(shape = (h,w), dtype = float)
    delta = np.full((h,w), np.finfo(float).tiny)

    NDVI = (nir - RED) / (nir + RED + delta)
    
    return NDVI

#Enhanced Vegetation
def evi (nir: np.ndarray, RED: np.ndarray, BLUE: np.ndarray, L = int(1), C1 = int(6), C2 = int(7.5)):
    
    if nir.shape != RED.shape or nir.shape != BLUE.shape:
        raise ValueError("Both arrays must have the same shape")
    (h,w) = np.shape(nir)
    EVI = np.empty(shape = (h,w), dtype = float)
    delta = np.full((h,w), np.finfo(float).tiny)

    red = normalize_to_uint8(RED)
    blue = normalize_to_uint8(BLUE)

    L_array = np.full((h,w), L)
    EVI = (nir - red) / (nir + C1*red - C2 * blue + L_array + delta)
    
    return EVI

#Water
def ndwi (nir: np.ndarray, GREEN: np.ndarray):

    if nir.shape != GREEN.shape:
        raise ValueError("Both arrays must have the same shape")
    (h,w) = np.shape(nir)
    NDWI = np.empty(shape = (h,w), dtype = float)
    delta = np.full((h,w), np.finfo(float).tiny)

    NDWI = (GREEN - nir) / (nir + GREEN + delta)
    
    return NDWI

#Turbidity of water
def ndti (RED: np.ndarray, GREEN: np.ndarray):

    if RED.shape != GREEN.shape:
        raise ValueError("Both arrays must have the same shape")
    (h,w) = np.shape(RED)
    NDTI = np.empty(shape = (h,w), dtype = float)
    delta = np.full((h,w), np.finfo(float).tiny)

    NDTI = (RED - GREEN) / (RED + GREEN + delta)
    
    return NDTI

#Canopy
def fcd (RED: np.ndarray, GREEN: np.ndarray, BLUE: np.ndarray):

    if BLUE.shape != RED.shape or GREEN.shape != BLUE.shape:
        raise ValueError("Both arrays must have the same shape")

    (h,w) = np.shape(RED)

    AVI = np.empty(shape = (h,w), dtype = float)
    BI = np.empty(shape = (h,w), dtype = float)
    SI = np.empty(shape = (h,w), dtype = float)   

    delta = np.full((h,w), np.finfo(float).tiny)
    one = np.full((h,w),fill_value=1)
    max = np.full((h,w),fill_value=256)

    red = normalize_to_uint8(RED)
    green = normalize_to_uint8(GREEN)
    blue = normalize_to_uint8(BLUE)

    AVI = np.cbrt(np.multiply((red + one).astype(np.float64), 
                          np.multiply((max - green).astype(np.float64), 
                                      (red - green).astype(np.float64))))
    BI = (RED + BLUE - GREEN)/(RED + GREEN + BLUE + delta)
    SI = np.sqrt(np.multiply(np.clip(max - blue, 0, None), np.clip(max - green, 0, None)))

    return AVI, BI, SI

#Concrete
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

#For image mask
def generate_mask(image: np.ndarray, min: float, max: float):
    (h,w) = np.shape(image)
    mask = np.zeros(shape=(h,w))

    for i in range(0,h):
        for j in range(0,w):
            if (image[i][j] >= min and image[i][j] < max):
                  mask[i][j] = 1

    return mask           

#For feature (area of mask)
def scan(image: np.ndarray, min: float, max: float):
    (h,w) = np.shape(image)
    mask = np.zeros(shape=(h,w))

    for i in range(0,h):
        for j in range(0,w):
            if (image[i][j] >= min and image[i][j] < max):
                  mask[i][j] = 1

    return np.sum(mask)           

def concrete(hsv,image):
    lowerBound = np.array([105, 0, 0])
    upperBound = np.array([[150, 110, 150]])

    mask = cv2.inRange(hsv,lowerBound, upperBound)
    result = cv2.bitwise_and(image, image, mask= mask)

    return np.sum(mask)/255

#Extract features
def extract_features(file_path):
    
    try:
        with rasterio.open(file_path) as dataset:
            # Read the bands
            RED = dataset.read(4)   # Band 4: Red
            GREEN = dataset.read(3) # Band 3: Green
            BLUE = dataset.read(2)  # Band 2: Blue
            nir = dataset.read(8)   # Band 8: Near-Infrared (NIR)

            NDVI = ndvi(nir,RED)
            EVI = evi(nir,RED,BLUE)
            NDWI = ndwi(nir,GREEN)
            NDTI = ndti(RED,GREEN)
            AVI, BI, SI = fcd(RED, GREEN, BLUE)

            features = []

            # NDVI (Normalized Difference Vegetation Index)
            features.append(scan(NDVI, -1, 0))    # Non-vegetated areas
            features.append(scan(NDVI, 0, 0.2))   # Sparse vegetation or soil
            features.append(scan(NDVI, 0.2, 0.5)) # Moderate vegetation
            features.append(scan(NDVI, 0.5, 0.7)) # Dense vegetation
            features.append(scan(NDVI, 0.7, 1))   # Very dense, healthy vegetation

            # EVI (Enhanced Vegetation Index)
            features.append(scan(EVI, -1, 0))    # Non-vegetated areas
            features.append(scan(EVI, 0, 0.2))   # Sparse vegetation
            features.append(scan(EVI, 0.2, 0.5)) # Moderate vegetation
            features.append(scan(EVI, 0.5, 0.8)) # Dense vegetation
            features.append(scan(EVI, 0.8, 1))   # Very dense, healthy vegetation

            # NDWI (Normalized Difference Water Index) [Green and NIR]
            features.append(scan(NDWI, -1, -0.2)) # Non-water surfaces
            features.append(scan(NDWI, -0.2, 0))  # Barely any water content
            features.append(scan(NDWI, 0, 0.2))   # Low water content
            features.append(scan(NDWI, 0.2, 1))   # High water content

            # NDTI (Normalized Difference Turbidity Index) THEORY UNCLEAR
            features.append(scan(NDTI, -1, -0.4)) # Non-tilled areas
            features.append(scan(NDTI, -0.4, 0))  # Slightly tilled land
            features.append(scan(NDTI, 0, 0.4))   # Moderately tilled land
            features.append(scan(NDTI, 0.4, 1))   # Heavily tilled land

            # AVI (Advanced Vegetation Index)
            features.append(scan(AVI, 0, 50))    # Non-vegetated areas
            features.append(scan(AVI, 50, 100))  # Sparse vegetation
            features.append(scan(AVI, 100, 150)) # Moderate vegetation
            features.append(scan(AVI, 150, 200)) # Dense vegetation
            features.append(scan(AVI, 200, 255)) # Very dense, healthy vegetation

            # BI (Bare Soil Index)
            features.append(scan(BI, 0, 0.2))   # High vegetation cover
            features.append(scan(BI, 0.2, 0.4)) # Low vegetation cover
            features.append(scan(BI, 0.4, 0.6)) # Mixed vegetation and bare soil
            features.append(scan(BI, 0.6, 0.8)) # Sparse vegetation, mostly bare soil
            features.append(scan(BI, 0.8, 1))   # Bare soil

            # SI (Shadow Index)
            features.append(scan(SI, 0, 0.2))   # No shadow
            features.append(scan(SI, 0.2, 0.4)) # Light shadow
            features.append(scan(SI, 0.4, 0.6)) # Moderate shadow
            features.append(scan(SI, 0.6, 0.8)) # Strong shadow
            features.append(scan(SI, 0.8, 1))   # Very strong shadow

            #Concrete
            red = normalize_to_uint8(RED)
            green = normalize_to_uint8(GREEN)
            blue = normalize_to_uint8(BLUE)
            
            '''
            bgr = np.stack([blue,green,red], axis = -1)
            image =  cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = hsv_hist(bgr, bins1, bins2)
            features = np.append(features, hist)
            '''

            bgr = np.stack([blue,green,red], axis = -1)
            image =  cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            concrete_index = concrete(image,bgr)

            features.append(concrete_index)

    except RasterioError as e:
        # Handle errors specific to rasterio
        print(f"Rasterio error: {e}")
    except FileNotFoundError as e:
        # Handle file not found errors
        print(f"File not found error: {e}")
    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {e}")

    return features

''''
df = pd.DataFrame(columns=np.arange((33 + bins1 * bins2)))
df["Classification"] = None
'''

directory = r"" #Path to your satellite images

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    #Construct a Data Frame
    df = pd.DataFrame(columns=np.arange((34))) # 33 + concrete + bins1 * bins2 (if required)
    df["Classification"] = None

    for filename_2 in os.listdir(filepath):
        if filename_2.endswith((".tif")):  # Check if file is an image
            tif_path = os.path.join(filepath, filename_2)
            feature = extract_features(tif_path)
            new_row = np.append(feature,filename)
            df.loc[len(df)] = new_row
    
    #Push the Data Frame into a CSV file
    print(df)
    name = filename + ".csv"
    df.to_csv(name)

