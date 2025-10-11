import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from preprocess import is_text_dark

def deskew(image: np.ndarray, show_result: bool = False, debug: bool = False):
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    text_is_dark = is_text_dark(binary)

    def compute_projection_profile(img, text_dark=True):
        """
        Compute projection profile based on whether text is dark or light
        """
        if text_dark:
            # Text is black (0), count black pixels
            binary_bool = (img == 0).astype(np.uint8)
        else:
            # Text is white (255), count white pixels
            binary_bool = (img == 255).astype(np.uint8)
        return np.sum(binary_bool, axis=1)

    def find_best_rotation(binary_img, text_dark=True, delta=0.5, limit=10, debug=False):
        best_score = -np.inf
        best_angle = 0
        scores = []
        angles = np.arange(-limit, limit + delta, delta)

        for angle in angles:
            rotated = rotate(binary_img, angle, resize=False, preserve_range=True).astype(np.uint8)
            # Rebinarize after rotation to avoid gray values
            _, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
            
            proj = compute_projection_profile(rotated, text_dark)
            score = np.std(proj)
            scores.append(score)

            if text_dark:
                if score < best_score:
                    best_score = score
                    best_angle = angle
            else:
                if score > best_score:
                    best_score = score
                    best_angle = angle
                
        if debug:
            plt.figure(figsize=(10, 5))
            plt.plot(angles, scores)
            plt.xlabel("Angle (°)")
            plt.ylabel("Projection StdDev")
            plt.title(f"Deskew Score Curve (Text: {'Dark' if text_dark else 'Light'})")
            plt.grid(True, alpha=0.3)
            plt.axvline(x=best_angle, color='r', linestyle='--', label=f'Best angle: {best_angle:.2f}°')
            plt.legend()
            plt.show()
        
        return best_angle

    angle = find_best_rotation(binary, text_dark=text_is_dark, debug=debug)
    if abs(angle) < 1:
        angle = 0

    deskewed = rotate(image, angle, resize=True, preserve_range=True).astype(np.uint8)

    # Display results
    if show_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Skewed Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(deskewed, cmap='gray')
        plt.title(f"Deskewed Image (Angle: {angle:.2f}°)\nText: {'Dark' if text_is_dark else 'Light'}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("debug_deskew.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return deskewed