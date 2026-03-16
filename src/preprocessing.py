# src/preprocessing.py
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def compute_ndvi_from_bgr(img_bgr):
    img = img_bgr.astype(float)
    b, g, r = cv2.split(img)
    
    nir_proxy = g  
    red = r        
    
    denom = (nir_proxy + red + 1e-6)
    
    # 🔥 STANDARD FORMULA: Higher values now equal vegetation/plants
    ndvi = (nir_proxy - red) / denom 
    
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi

def threshold_mask_from_ndvi(ndvi, thresh=0.05):
    # To get soil, we want values where NDVI is LOW
    # mask = (ndvi <= thresh).astype('uint8') * 255
    # 🔥 PLANTS = WHITE: We want values where NDVI is GREATER than the threshold    
    mask = (ndvi > thresh).astype('uint8') * 255
    return mask

def detect_weeds_by_row_crops(img_bgr, row_spacing=50, row_tolerance=15):
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255
    
    h, w = veg_mask.shape
    density = np.sum(veg_mask > 0, axis=0) / h  
    
    smoothed = gaussian_filter1d(density, sigma=10)
    peaks, _ = find_peaks(smoothed, distance=row_spacing//2, prominence=0.1)
    
    crop_row_mask = np.zeros_like(veg_mask)
    for peak in peaks:
        x_start = max(0, peak - row_tolerance)
        x_end = min(w, peak + row_tolerance)
        crop_row_mask[:, x_start:x_end] = 255
    
    # Soil = everything that is NOT vegetation
    soil_mask = cv2.bitwise_not(veg_mask)
    return soil_mask

def detect_weeds_by_size_color(img_bgr, min_area=50, max_area=5000):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    crop_lower = np.array([40, 80, 60])   
    crop_upper = np.array([75, 255, 255])
    crop_mask = cv2.inRange(hsv, crop_lower, crop_upper)
    
    weed_lower = np.array([20, 30, 30])   
    weed_upper = np.array([90, 255, 255])
    potential_weed = cv2.inRange(hsv, weed_lower, weed_upper)
    
    weed_mask = cv2.bitwise_and(potential_weed, cv2.bitwise_not(crop_mask))
    
    contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(weed_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
    return filtered_mask

def detect_weeds_texture_based(img_bgr, threshold=0.6):
    """
    Detect weeds using texture analysis.
    Threshold controls the variance percentile.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255
    
    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = sqr_mean - mean**2
    
    # Convert 0.0-1.0 slider to a 1-99 percentile
    percentile_val = max(1, min(99, int((1.0 - threshold) * 100))) 
    
    # Safety check to prevent crashing on empty masks
    if not np.any(veg_mask > 0):
        return np.zeros_like(gray, dtype=np.uint8)
        
    texture_thresh = np.percentile(variance[veg_mask > 0], percentile_val)
    high_variance_mask = (variance > texture_thresh).astype('uint8') * 255
    
    weed_mask = cv2.bitwise_and(veg_mask, high_variance_mask)
    return weed_mask

def compute_color_based_mask(img_bgr, lower_green, upper_green):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def resize_keep_aspect(img, max_side=512):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return new

def combined_weed_heatmap(img):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=float)

    color_mask = detect_weeds_by_size_color(img)
    heat += color_mask / 255.0  

    row_mask = detect_weeds_by_row_crops(img)
    heat += row_mask / 255.0

    texture_mask = detect_weeds_texture_based(img)
    heat += texture_mask / 255.0

    heat = np.clip(heat / 3.0, 0, 1)
    return heat