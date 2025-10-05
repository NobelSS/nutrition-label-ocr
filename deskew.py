import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

def deskew(image: np.ndarray, show_result: bool = False, debug: bool = False):
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # inverted = cv2.bitwise_not(image)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def compute_projection_profile(img):
        binary_bool = (img == 0).astype(np.uint8)
        return np.sum(binary_bool, axis=1)

    def find_best_rotation(binary_img, delta=0.5, limit=10, debug=False):
                
        best_score = -np.inf
        best_angle = 0
        for angle in np.arange(-limit, limit + delta, delta):
            rotated = rotate(binary_img, angle, resize=False)
            proj = compute_projection_profile(rotated)
            score = np.std(proj) # Lower value means pixels are evenly spread (skewed), higher value means the text is aligned
            if debug:
                print(f"Angle: {angle:.2f}°, Score: {score:.4f}")
                plt.imshow(rotated, cmap='gray')
                plt.title(f"Angle: {angle:.2f}°, Score: {score:.4f}")
                plt.axis('off')
                plt.show()

            if score > best_score:
                best_score = score
                best_angle = angle
        return best_angle

    angle = find_best_rotation(binary, debug=debug)
    if abs(angle) < 1:
        angle = 0

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
        plt.savefig("debug_deskew.png", dpi=300, bbox_inches='tight')
        plt.show()
        

    
    return deskewed
