import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess(image: np.ndarray, save_result: bool = True, output_path: str = "output_image.png", debug: bool = False):

    if debug:
        print('Preprocessing image...')

    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image # Transform for OpenCV format
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    contrast = enhance_contrast(gray) # Not used
    equalized = histogram_equalization(gray)
    denoised = denoise(gray)
    thresh = binarize(gray)

    if debug:
        plt.figure(figsize=(12, 10))
        plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Deskewed'), plt.axis('off')
        plt.subplot(232), plt.imshow(contrast, cmap='gray'), plt.title('CLAHE'), plt.axis('off')
        plt.subplot(233), plt.imshow(equalized, cmap='gray'), plt.title('Equalized'), plt.axis('off')
        plt.subplot(234), plt.imshow(denoised, cmap='gray'), plt.title('Denoised'), plt.axis('off')
        plt.subplot(235), plt.imshow(thresh, cmap='gray'), plt.title('Combined Threshold'), plt.axis('off')
        plt.tight_layout()
        plt.savefig("debug_preprocess.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    if save_result:
        cv2.imwrite(output_path, thresh)
        print(f"Processed image saved to {output_path}")
    
    return thresh
    
    # final = test(image)
    
    # return final
    
def histogram_equalization(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.equalizeHist(gray)   
    
def enhance_contrast(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cv2.equalizeHist(gray)


def binarize(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return otsu

    
    # Combined threshold
    # _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     image, 
    #     255, 
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY_INV,
    #     11, 
    #     2
    # )

    # combined = cv2.bitwise_or(otsu, adaptive_thresh)
    
    # return combined

def denoise(image: np.ndarray):
    # return cv2.fastNlMeansDenoising(image, h=5)
    return cv2.bilateralFilter(image, 9, 75, 75)
