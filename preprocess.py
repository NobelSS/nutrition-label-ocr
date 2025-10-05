import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess(image: np.ndarray, save_result: bool = True, output_path: str = "output_image.png", debug: bool = False):

    if debug:
        print('Preprocessing image...')

    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image # Transform for OpenCV format
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = binarize(gray)

    if debug:

        contrast = enhance_contrast(gray) # Not used
        equalized = histogram_equalization(gray)
        denoised = denoise(equalized)
        adapt_thresh = adaptive_threshold(denoised)
        morphological_img = morphological(adapt_thresh, kernel_size=2)
        plt.figure(figsize=(12, 10))
        plt.subplot(241), plt.imshow(image, cmap='gray'), plt.title('Deskewed'), plt.axis('off')
        plt.subplot(242), plt.imshow(contrast, cmap='gray'), plt.title('CLAHE'), plt.axis('off')
        plt.subplot(243), plt.imshow(equalized, cmap='gray'), plt.title('Equalized'), plt.axis('off')
        plt.subplot(244), plt.imshow(denoised, cmap='gray'), plt.title('Denoised'), plt.axis('off')
        plt.subplot(245), plt.imshow(thresh, cmap='gray'), plt.title('Otsu'), plt.axis('off')
        plt.subplot(246), plt.imshow(adapt_thresh, cmap='gray'), plt.title('Adaptive Threshold'), plt.axis('off')
        plt.subplot(247), plt.imshow(morphological_img, cmap='gray'), plt.title('Morphological'), plt.axis('off')
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

def adaptive_threshold(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    return cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        25, 
        10
    )

def morphological(image: np.ndarray, kernel_size: int = 2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

