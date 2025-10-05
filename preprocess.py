import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

def preprocess(image: np.ndarray, save_result: bool = True, output_path: str = "output_image.png", debug: bool = False):

    if debug:
        print('Preprocessing image...')

    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image # Transform for OpenCV format
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = binarize(gray)
    
    sharpened_kernel = sharpen_kernel(gray)
    sharpened_mask = sharpen_mask(gray)
        
    thresh_sharpened_mask = binarize(sharpened_mask)
    thresh_sharpened_kernel = binarize(sharpened_kernel)
    
    sauvola = threshold_sauvola(gray, window_size=25)
    binary_sauvola = (gray > sauvola).astype(np.uint8) * 255
    binary_sauvola = cv2.bitwise_not(binary_sauvola)
    if debug:

        contrast = enhance_contrast(gray) # Not used
        equalized = histogram_equalization(gray)
        denoised = denoise(equalized)
        adapt_thresh = adaptive_threshold(denoised)
        morphological_img = morphological(thresh, kernel_size=2)
    
        # sharpened_kernel = sharpen_kernel(gray)
        # sharpened_mask = sharpen_mask(gray)
        
        # thresh_sharpened_mask = binarize(sharpened_mask)
        # thresh_sharpened_kernel = binarize(sharpened_kernel)
        
        # sauvola = threshold_sauvola(gray, window_size=25)
        # binary_sauvola = (gray > sauvola).astype(np.uint8) * 255
        # binary_sauvola = cv2.bitwise_not(binary_sauvola)
        
        plt.figure(figsize=(16, 10))
        plt.subplot(351), plt.imshow(image, cmap='gray'), plt.title('Deskewed'), plt.axis('off')
        plt.subplot(352), plt.imshow(contrast, cmap='gray'), plt.title('CLAHE'), plt.axis('off')
        plt.subplot(353), plt.imshow(equalized, cmap='gray'), plt.title('Equalized'), plt.axis('off')
        plt.subplot(354), plt.imshow(denoised, cmap='gray'), plt.title('Denoised'), plt.axis('off')
        plt.subplot(355), plt.imshow(thresh, cmap='gray'), plt.title('Otsu'), plt.axis('off')
        plt.subplot(356), plt.imshow(adapt_thresh, cmap='gray'), plt.title('Adaptive Threshold'), plt.axis('off')
        plt.subplot(357), plt.imshow(morphological_img, cmap='gray'), plt.title('Morphological'), plt.axis('off')
        plt.subplot(358), plt.imshow(sharpened_kernel, cmap='gray'), plt.title('Sharpen Kernel'), plt.axis('off')
        plt.subplot(359), plt.imshow(sharpened_mask, cmap='gray'), plt.title('Sharpen Mask'), plt.axis('off')
        plt.subplot(3,5,10), plt.imshow(thresh_sharpened_kernel, cmap='gray'), plt.title('Thresh + Sharpen Kernel'), plt.axis('off')
        plt.subplot(3,5,11), plt.imshow(thresh_sharpened_mask, cmap='gray'), plt.title('Thresh + Sharpen Mask'), plt.axis('off')
        plt.subplot(3,5,12), plt.imshow(binary_sauvola, cmap='gray'), plt.title('Sauvola'), plt.axis('off')
        plt.tight_layout()
        plt.savefig("debug_preprocess.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    if save_result:
        cv2.imwrite(output_path, thresh)
        print(f"Processed image saved to {output_path}")
    
    return thresh_sharpened_mask
    
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

def sharpen_kernel(image: np.ndarray):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sharpen_mask(image: np.ndarray):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened