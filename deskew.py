import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

def deskew(image: np.ndarray, show_result: bool = True):
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(image)

    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def compute_projection_profile(img):
        return np.sum(img, axis=1)

    def find_best_rotation(binary_img, delta=0.5, limit=10):
        print("Finding the best rotation angle...")
        
        best_score = -np.inf
        best_angle = 0
        for angle in np.arange(-limit, limit + delta, delta):
            rotated = rotate(binary_img, angle, resize=False)
            proj = compute_projection_profile(rotated)
            score = np.std(proj) # Lower value means pixels are evenly spread (skewed), higher value means the text is aligned
            if score > best_score:
                best_score = score
                best_angle = angle
        return best_angle

    angle = find_best_rotation(binary)

    deskewed = rotate(image, angle, resize=True)

    # Display results
    if show_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Skewed Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(deskewed, cmap='gray')
        plt.title(f"Deskewed Image (Angle: {angle:.2f}°)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    return deskewed
