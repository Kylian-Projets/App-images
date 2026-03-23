import PIL
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st


def histogram(_img,gray_scale):
    img_array = np.array(_img.convert("L"))

    fig,ax = plt.subplots()
    ax.hist(img_array.ravel(),bins=256,range=(0,256),color="green")
    ax.set_title("Intensités (niveaux de gris)")
    ax.set_xlabel("Luminosité (0-255)")
    ax.set_ylabel("Nombre de pixels")

    return fig

def otsu(_img):

    img_array = np.array(_img.convert("L"))
    threshold,image_bin = cv2.threshold(img_array,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = Image.fromarray(image_bin)
    return (threshold, image_bin)

def morpho(_img_bin,kernel_size, iterations):

    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img = np.array(_img_bin)

    #erode
    img_erode = cv2.erode(img,kernel,iterations)

    #dilation
    img_dilate = cv2.dilate(img,kernel, iterations)

    #opening
    img_open = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

    #closing
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel)

    return (img_erode,img_dilate,img_open,img_close)



def full_processing_pipeline(_image_pil, bg_type="light", blur_strength=5, morph_val=3, min_area=500):
    
    img_copy = _image_pil.copy()
    img_copy.thumbnail((500, 500)) 

    img_area = img_copy.size[0] * img_copy.size[1]    
    img_rgb = np.array(img_copy.convert('RGB'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Blur and binarization 
    k_blur = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
    img_blurred = cv2.GaussianBlur(img_gray, (k_blur, k_blur), 0)
    mode = cv2.THRESH_BINARY_INV if bg_type == "light" else cv2.THRESH_BINARY
    #val_otsu, img_bin = cv2.threshold(img_blurred, 0, 255, mode + cv2.THRESH_OTSU)
    if bg_type == "light":
        img_bin = cv2.adaptiveThreshold(
        img_blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,
        11, 2 )
    else:
        img_bin = cv2.adaptiveThreshold(
        img_blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        11, 2)

    # Morpho
    kernel = np.ones((morph_val, morph_val), np.uint8)
    img_morpho = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # Bounding Boxes
    img_final_render = img_rgb.copy()
    contours, _ = cv2.findContours(img_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    real_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area < 0.8*img_area: 
            # Calcul du rectangle (x, y, largeur, hauteur)
            x, y, w, h = cv2.boundingRect(cnt)
            # Dessin du rectangle : (image, début, fin, couleur BGR, épaisseur)
            cv2.rectangle(img_final_render, (x, y), (x + w, y + h), (0, 255, 0), 2)
            real_count += 1

    return {
        "gris": Image.fromarray(img_gray),
        "bin": Image.fromarray(img_bin),
        "morpho": Image.fromarray(img_morpho),
        "final": Image.fromarray(img_final_render),
        "count": real_count,
        #"otsu_val": val_otsu
    }