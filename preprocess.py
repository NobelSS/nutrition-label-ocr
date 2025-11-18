import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

def preprocess(image: np.ndarray, save_result: bool = True, save_path: str = "output_image.png", debug: bool = False):

    if debug:
        print('Preprocessing image...')

    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image # Transform for OpenCV format
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # denoised = denoise(gray)
    # thresh = binarize(denoised)
    thresh = binarize(gray)
    
    sharpened_kernel = sharpen_kernel(gray)
    sharpened_mask = sharpen_mask(gray)
        
    thresh_sharpened_mask = binarize(sharpened_mask)
    thresh_sharpened_kernel = binarize(sharpened_kernel)
    
    sauvola_threshold = threshold_sauvola(gray, window_size=25)
    binary_sauvola = (gray > sauvola_threshold).astype(np.uint8) * 255
    binary_sauvola = cv2.bitwise_not(binary_sauvola)
    
    # if is_text_dark(thresh):
    #     binary_sauvola = cv2.bitwise_not(binary_sauvola)
    #     thresh_sharpened_mask = cv2.bitwise_not(thresh_sharpened_mask)
    #     thresh_sharpened_kernel = cv2.bitwise_not(thresh_sharpened_kernel)

    
    if debug or save_result:
        
        contrast = enhance_contrast(gray)
        # equalized = histogram_equalization(gray)
        # denoised = denoise(equalized)
        # adapt_thresh = adaptive_threshold(denoised)
        # morphological_img = morphological(thresh, kernel_size=2)

        plot_preprocess_results(
            image=image,
            # contrast=contrast,
            # equalized=equalized,
            # denoised=denoised,
            thresh=thresh,
            # adapt_thresh=adapt_thresh,
            # morphological_img=morphological_img,
            sharpened_kernel=sharpened_kernel,
            sharpened_mask=sharpened_mask,
            thresh_sharpened_kernel=thresh_sharpened_kernel,
            thresh_sharpened_mask=thresh_sharpened_mask,
            binary_sauvola=binary_sauvola,
            show=debug,
            save_path=save_path if save_result else None
        )
    
    return sharpened_mask
    
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
    # return cv2.bilateralFilter(image, 9, 75, 75)
    return cv2.GaussianBlur(image, (3, 3), 0)

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

def plot_preprocess_results(
    image=None,
    contrast=None,
    equalized=None,
    denoised=None,
    thresh=None,
    adapt_thresh=None,
    morphological_img=None,
    sharpened_kernel=None,
    sharpened_mask=None,
    thresh_sharpened_kernel=None,
    thresh_sharpened_mask=None,
    binary_sauvola=None,
    show=False,
    save_path=None
):
    """Display or save a safe multi-step preprocessing visualization."""
    steps = [
        ("Deskewed", image),
        ("CLAHE", contrast),
        ("Equalized", equalized),
        ("Denoised", denoised),
        ("Otsu", thresh),
        ("Adaptive Threshold", adapt_thresh),
        ("Morphological", morphological_img),
        ("Sharpen Kernel", sharpened_kernel),
        ("Sharpen Mask", sharpened_mask),
        ("Thresh + Sharpen Kernel", thresh_sharpened_kernel),
        ("Thresh + Sharpen Mask", thresh_sharpened_mask),
        ("Sauvola", binary_sauvola),
    ]

    # Use explicit figure to avoid thread conflicts
    fig = plt.figure(figsize=(16, 10))
    for i, (title, img) in enumerate(steps, start=1):
        ax = fig.add_subplot(3, 5, i)
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(np.zeros((50, 50)), cmap='gray')  # placeholder
            ax.text(25, 25, 'None', ha='center', va='center', color='red', fontsize=8)
        ax.set_title(title)
        ax.axis('off')

    fig.tight_layout()

    # Save or show depending on flags
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[DEBUG] Saved preprocess plot → {save_path}")
    elif show:
        plt.show()
        plt.close(fig)
        
        
import cv2
import numpy as np

def is_text_dark(img: np.ndarray, debug: bool = False) -> bool:
    """
    Determine if text is darker than the background.
    Robust to dark borders, colored backgrounds, and low contrast.

    Args:
        img: Grayscale image (uint8)
        debug: Print stats for analysis

    Returns:
        True if text is dark-on-light, False if text is light-on-dark.
    """
    if len(img.shape) != 2:
        raise ValueError("Input must be a grayscale image.")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Otsu threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.mean(binary == 255)  # fraction of white pixels
    mean_original = np.mean(img)
    mean_binary = np.mean(binary)

    # Heuristic: if most of the image is white after thresholding → text is dark
    text_dark = white_ratio > 0.5

    if debug:
        print(f"[is_text_dark] white_ratio={white_ratio:.3f}, mean_original={mean_original:.2f}, mean_binary={mean_binary:.2f}, text_dark={text_dark}")

    return text_dark
