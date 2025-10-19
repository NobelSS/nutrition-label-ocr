import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import cv2

from preprocess import denoise, enhance_contrast

def compare_preprocessing_variants(image: np.ndarray, output_path: str = "preprocessing_comparison.png"):
    print("Generating preprocessing comparison...")

    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # gray = enhance_contrast(gray)

    def otsu(img): return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    def adaptive(img): return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
    def combined(img): return cv2.bitwise_or(otsu(img), adaptive(img))

    variants = [
        ("No Blur + Combined", combined(gray)),
        ("No Blur + Otsu", otsu(gray)),
        ("No Blur + Adaptive", adaptive(gray)),
        ("Bilateral + Combined", combined(denoise(gray))),
        ("Bilateral + Otsu", otsu(denoise(gray))),
        ("Bilateral + Adaptive", adaptive(denoise(gray)))
    ]

    # Use explicit figure to avoid thread conflicts
    fig = plt.figure(figsize=(15, 10))
    for i, (title, img) in enumerate(variants, 1):
        ax = fig.add_subplot(2, 3, i)
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison saved to {output_path}")