import numpy as np
import cv2
import matplotlib.pyplot as plt

def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def local_contrast(image, block_size=16):
    h, w = image.shape
    contrasts = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size > 0:
                contrasts.append(block.std())
    return np.mean(contrasts)

def perspective_correction(image: np.ndarray, show_result: bool = False, debug: bool = False):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
            
    print("Finding Contours and performing Perspective Correction...")
    
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 1)
    contrast = clahe
    
    # if local_contrast(equalized) > local_contrast(clahe):
    #     blur = cv2.GaussianBlur(equalized, (3, 3), 1)
    #     contrast = equalized
    #     print('Using Histogram Equalization.')

    sobel_x = cv2.Sobel(contrast, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(contrast, cv2.CV_64F, 0, 1, ksize=5)

    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = cv2.convertScaleAbs(gradient)
    
    canny = cv2.Canny(contrast, 100, 200)
    
    _, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(closed, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_display, contours, -1, (255, 0, 0), 2)

    min_contour_area_threshold = 5000
    epsilon_factor = 0.02

    main_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area >= min_contour_area_threshold:
            max_area = area
            main_contour = contour
    
    corrected_image = image # Default: original image
    if main_contour is not None:
  
        if debug:
    
            cv2.drawContours(contour_display, [main_contour], -1, (0, 0, 255), 3)
            
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 4, 1)
            plt.title("All Contours (blue) with Largest Contour Highlighted (red)")
            plt.imshow(cv2.cvtColor(contour_display, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(2, 4, 2)
            plt.title("Sobel")
            plt.imshow(gradient, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 3)
            plt.title("Canny")
            plt.imshow(canny, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 4)
            plt.title("Blur")
            plt.imshow(blur, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 5)
            plt.title("Equalized")
            plt.imshow(equalized, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 6)
            plt.title("CLAHE")
            plt.imshow(clahe, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 7)
            plt.title("Closed")
            plt.imshow(closed, cmap='gray')
            plt.axis("off")
            
            plt.subplot(2, 4, 8)
            plt.title("Dilation")
            plt.imshow(dilation, cmap='gray')
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
          
        peri = cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon_factor * peri, True)

        if len(approx) == 4:
            
            height, width = image.shape[:2]
            image_area = height * width

            x, y, w, h = cv2.boundingRect(approx)
            approx_area = w * h

            if approx_area < 0.5 * image_area:
                print("Warning: Approximated quadrilateral is too small relative to the image.")
                return corrected_image # Return original image
                

            pts = order_points(approx.reshape(4, 2))        
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(pts, dst)
            
            corrected_image = cv2.warpPerspective(image, M, (width, height))

            if len(corrected_image.shape) == 2:
                corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
            
            display_image_with_contour = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(display_image_with_contour, [main_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(display_image_with_contour, [approx], -1, (255, 0, 0), 3) 
        else:
            print("Warning: Did not find a 4-cornered object for perspective correction.")
            display_image_with_contour = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(display_image_with_contour, [main_contour], -1, (0, 255, 0), 2)
    else:
        print("Warning: No main contour found for perspective correction.")
        display_image_with_contour = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)


    if show_result:
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.title("Original Image with Contours")
        plt.imshow(cv2.cvtColor(display_image_with_contour, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Perspective Corrected")
        if corrected_image is not None:
            
            if len(corrected_image.shape) == 3 and corrected_image.shape[2] == 3:
                 plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
            else:
                 plt.imshow(corrected_image, cmap='gray')
        else:
            plt.text(0.5, 0.5, 'Correction Failed', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.imshow(np.zeros_like(gray), cmap='gray') # Show a blank image if no correction

        plt.tight_layout()
        plt.show()
        
    return corrected_image
