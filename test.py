import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def rectify_label_merge_sections(image, show_steps=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. Preprocess to suppress text ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Morphological close to connect borders
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel_close)

    # Morphological open to remove thin text edges
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # --- 2. Edge detection ---
    edges = cv2.Canny(opened, 50, 150)

    # --- 3. Find contours ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ No contours found.")
        return image

    # --- 4. Merge all contours ---
    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)

    # Approximate to quadrilateral
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) != 4:
        print("⚠️ Did not find a quadrilateral. Using minAreaRect fallback.")
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)          # shape (4,2), float
        approx = np.array(box, dtype=np.int32)
        approx = approx.reshape(-1,1,2)    # reshape to contour format


    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    # --- 5. Perspective transform ---
    h, w = image.shape[:2]
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    # --- 6. Visualization ---
    if show_steps:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [approx], -1, (0,255,0), 3)

        plt.figure(figsize=(18,10))
        plt.subplot(2,3,1); plt.title("Gray"); plt.imshow(gray, cmap="gray"); plt.axis("off")
        plt.subplot(2,3,2); plt.title("Closed (morph)"); plt.imshow(closed, cmap="gray"); plt.axis("off")
        plt.subplot(2,3,3); plt.title("Opened (less text)"); plt.imshow(opened, cmap="gray"); plt.axis("off")
        plt.subplot(2,3,4); plt.title("Canny Edges"); plt.imshow(edges, cmap="gray"); plt.axis("off")
        plt.subplot(2,3,5); plt.title("Merged Contour"); plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.subplot(2,3,6); plt.title("Warped / Rectified"); plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return warped

# --- Run ---
image = cv2.imread("dataset/labeled/image_77.jpg")
rectified = rectify_label_merge_sections(image, show_steps=True)
