import cv2
import numpy as np
import matplotlib.pyplot as plt

class NutritionLabelScanner:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.corners = None
        
    def load_image(self, image_path):
        """Load image from file path"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image
    
    def resize_image(self, image, max_width=1000):
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            return cv2.resize(image, (new_width, new_height))
        return image
    
    def enhance_contrast(self, image):
        """Enhance contrast using CLAHE"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def preprocess_for_labels(self, image, blur_kernel=3, canny_low=30, canny_high=100):
        """Preprocess specifically for nutrition labels with better edge detection"""
        # Resize for processing
        resized = self.resize_image(image)
        
        # Enhance contrast first
        enhanced = self.enhance_contrast(resized)
        
        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply slight Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (blur_kernel, blur_kernel), 0)
        
        # Apply Canny edge detection with lower thresholds for labels
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Apply morphological operations to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to make edges more prominent
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges, resized
    
    def find_label_contour(self, edges, min_area_ratio=0.1):
        """Find rectangular contour that likely represents a nutrition label"""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate minimum area based on image size
        image_area = edges.shape[0] * edges.shape[1]
        min_area = image_area * min_area_ratio
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        label_contour = None
        
        # Look for rectangular contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Approximate contour
            epsilon = 0.015 * cv2.arcLength(contour, True)  # More aggressive approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # print(f"Vertices: {len(approx)}")

            # Check if it's roughly rectangular (4-8 vertices)
            if 4 <= len(approx) <= 8:
                # For more than 4 vertices, try to find the best 4 corners
                if len(approx) > 4:
                    # Find convex hull
                    # hull = cv2.convexHull(contour)
                    # epsilon = 0.02 * cv2.arcLength(hull, True)
                    # approx = cv2.approxPolyDP(hull, epsilon, True)
                    
                    # if len(approx) >= 4:
                    #     # Take the 4 corners that form the largest quadrilateral
                    #     approx = self.get_best_quadrilateral(approx)
                    
                    rect = cv2.minAreaRect(approx)  # returns center, size, angle
                    box = cv2.boxPoints(rect)       # 4 corners
                    box = box.astype(np.int32).reshape(-1, 1, 2)
                    return box
                
                if len(approx) == 4:
                    # Check aspect ratio (labels are usually rectangular)
                    rect = cv2.boundingRect(approx)
                    aspect_ratio = rect[2] / rect[3]  # width / height
                    
                    # Labels are usually taller than wide or square-ish
                    if 0.3 <= aspect_ratio <= 3.0:
                        label_contour = approx
                        break
        
        return label_contour
    
    def get_best_quadrilateral(self, points):
        """Extract the best 4 corners from a set of points"""
        if len(points) <= 4:
            return points
            
        # Reshape points
        pts = points.reshape(-1, 2).astype(np.float32)
        
        # Find the center
        center = np.mean(pts, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        # Take every n/4 points to get roughly 4 corners
        n = len(sorted_pts)
        indices = [0, n//4, n//2, 3*n//4]
        quad_pts = sorted_pts[indices]
        
        return quad_pts.reshape(-1, 1, 2).astype(np.int32)
    
    def order_points(self, pts):
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left has smallest sum, bottom-right has largest sum
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference, bottom-left has largest difference
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def get_perspective_transform_matrix(self, corners, target_width=400, target_height=600):
        """Calculate perspective transform matrix with standard label dimensions"""
        rect = self.order_points(corners.reshape(4, 2))
        
        # Calculate the width and height of the original quadrilateral
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Use calculated dimensions or target dimensions
        if target_width and target_height:
            # Maintain aspect ratio while using target dimensions
            aspect_ratio = maxWidth / maxHeight
            if aspect_ratio > target_width / target_height:
                # Wider than target
                width = target_width
                height = int(target_width / aspect_ratio)
            else:
                # Taller than target
                height = target_height
                width = int(target_height * aspect_ratio)
        else:
            width, height = maxWidth, maxHeight
            
        width = max(width, 1)
        height = max(height, 1)
        
        # Define destination points for a perfect rectangle
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        return M, (width, height)
    
    def apply_perspective_transform(self, image, corners):
        """Apply perspective correction to straighten the label"""
        M, (width, height) = self.get_perspective_transform_matrix(corners)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped
    
    def enhance_text_readability(self, image):
        """Apply specific enhancements for text readability in labels"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to smooth while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 80, 80)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up with morphological operations
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_label(self, image_path=None, image=None, 
                    blur_kernel=3, canny_low=30, canny_high=100,
                    min_area_ratio=0.1, show_steps=False):
        """Main function to detect nutrition label in image"""
        
        # Load image
        if image_path:
            image = self.load_image(image_path)
        elif image is not None:
            self.original_image = image.copy()
        elif image is None:
            raise ValueError("Either image_path or image must be provided")
        
        # Preprocess image
        edges, resized_image = self.preprocess_for_labels(
            image, blur_kernel, canny_low, canny_high
        )
        
        # Find label contour
        label_contour = self.find_label_contour(edges, min_area_ratio)
        
        if label_contour is None:
            print("No label detected. Try adjusting parameters.")
            if show_steps:
                self.show_detection_failure(image, edges, resized_image)
            return None, None
        
        # Scale contour back to original image size
        scale_factor = image.shape[1] / resized_image.shape[1]
        
        scaled_contour = label_contour.astype(np.float32) * scale_factor
        
        self.corners = scaled_contour
        
        if show_steps:
            self.show_detection_steps(image, edges, resized_image, label_contour)
        
        return scaled_contour, image
    
    def rectify_label(self, image=None, corners=None, enhance=True, 
                     target_width=400, target_height=600):
        """Apply perspective correction and enhancement specifically for labels"""
        if image is None:
            image = self.original_image
        if corners is None:
            corners = self.corners
            
        if corners is None:
            raise ValueError("No corners detected. Run detect_label first.")
        
        # Apply perspective transform
        rectified = self.apply_perspective_transform(image, corners)
        
        # Enhance for text readability if requested
        if enhance:
            enhanced = self.enhance_text_readability(rectified)
            self.processed_image = enhanced
            return enhanced
        else:
            self.processed_image = rectified
            return rectified
    
    def show_detection_steps(self, original, edges, resized, contour):
        """Visualize the detection process"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image', fontsize=12)
        axes[0,0].axis('off')
        
        # Edge detection
        axes[0,1].imshow(edges, cmap='gray')
        axes[0,1].set_title('Edge Detection', fontsize=12)
        axes[0,1].axis('off')
        
        # Detected contour
        contour_img = resized.copy()
        if contour is not None:
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
            # Draw corner points
            for point in contour:
                cv2.circle(contour_img, tuple(point[0]), 8, (255, 0, 0), -1)
        
        axes[1,0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[1,0].set_title('Detected Label Boundary', fontsize=12)
        axes[1,0].axis('off')
        
        # Final result if available
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:
                axes[1,1].imshow(self.processed_image, cmap='gray')
            else:
                axes[1,1].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            axes[1,1].set_title('Rectified Label', fontsize=12)
        else:
            axes[1,1].text(0.5, 0.5, 'No rectified image\navailable', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Rectified Label', fontsize=12)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_detection_failure(self, original, edges, resized):
        """Show why detection might have failed"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Resized for Processing')
        axes[1].axis('off')
        
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Edge Detection (try adjusting parameters)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_result(self, output_path, image=None):
        """Save the processed image"""
        if image is None:
            image = self.processed_image
        
        if image is None:
            raise ValueError("No processed image to save")
        
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")