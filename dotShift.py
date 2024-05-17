import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def detect_dots(image):
    """Detect dots in the image using blob detection with sub-pixel accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    # Refine the coordinates using sub-pixel accuracy
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    refined_points = cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), 
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    return refined_points

def match_dots(dots_image1, dots_image2):
    """Match dots using the Hungarian method with a flexible cost matrix."""
    cost_matrix = np.array([[np.linalg.norm(dot1 - dot2) for dot2 in dots_image2] for dot1 in dots_image1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pairs_ref = [dots_image1[i] for i in row_ind]
    matched_pairs_test = [dots_image2[j] for j in col_ind]
    
    return matched_pairs_ref, matched_pairs_test

def calculate_deviation(dot1, dot2):
    """Calculate the deviation between two dots."""
    return np.linalg.norm(dot1 - dot2)

def visualize_results(image1, image2, matched_pairs_ref, matched_pairs_test):
    """Draw dots and connecting lines on the images for visualization."""
    # Resize images to have the same height
    min_height = min(image1.shape[0], image2.shape[0])
    image1_resized = cv2.resize(image1, (image1.shape[1], min_height))
    image2_resized = cv2.resize(image2, (image2.shape[1], min_height))

    for dot1, dot2 in zip(matched_pairs_ref, matched_pairs_test):
        cv2.circle(image1_resized, tuple(map(int, dot1)), 3, (0, 0, 255), -1)
        cv2.circle(image2_resized, tuple(map(int, dot2)), 3, (0, 255, 0), -1)
        cv2.line(image1_resized, tuple(map(int, dot1)), tuple(map(int, dot2)), (255, 0, 0), 1)

    combined_image = np.hstack((image1_resized, image2_resized))
    cv2.imshow("Matched Dots", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image1 = cv2.imread("/Users/nitesh/Desktop/ocr project/DORD/image/image1.jpeg")
    image2 = cv2.imread("/Users/nitesh/Desktop/ocr project/DORD/image/image2.jpeg")

    dots_image1 = detect_dots(image1)
    dots_image2 = detect_dots(image2)

    matched_pairs_ref, matched_pairs_test = match_dots(dots_image1, dots_image2)

    # Calculate the deviation between corresponding dots
    deviations = [calculate_deviation(dot1, dot2) for dot1, dot2 in zip(matched_pairs_ref, matched_pairs_test)]

    # Display the deviations
    for i, deviation in enumerate(deviations):
        print(f"Deviation for dot pair {i + 1}: {deviation:.6f} pixels")

    # Visualization
    visualize_results(image1.copy(), image2.copy(), matched_pairs_ref, matched_pairs_test)

if __name__ == "__main__":
    main()
