import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def detect_dots(image):
    """Detect dots in the image using a simple thresholding method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [np.array([cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"], 
             cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"]]) for contour in contours]  
"""return val is x,y coordinates of the dots(contours) detected"""

def match_dots(dots_image1, dots_image2):
    """Match dots using the Hungarian method."""
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
    for dot1, dot2 in zip(matched_pairs_ref, matched_pairs_test):
        cv2.circle(image1, tuple(map(int, dot1)), 3, (0, 0, 255), -1)
        cv2.circle(image2, tuple(map(int, dot2)), 3, (0, 255, 0), -1)
        cv2.line(image1, tuple(map(int, dot1)), tuple(map(int, dot2)), (255, 0, 0), 1)

    cv2.imshow("Reference Image with Matched Dots", image1)
    cv2.imshow("Test Image with Matched Dots", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image1 = cv2.imread("/Users/nitesh/Desktop/ocr project/DORD/image/image1.jpg")
    image2 = cv2.imread("/Users/nitesh/Desktop/ocr project/DORD/image/image2.jpg")


    dots_image1 = detect_dots(image1)
    dots_image2 = detect_dots(image2)

    if len(dots_image1) != len(dots_image2):
        print("The number of dots in the two images is not the same. Please check the images.")
        return

    matched_pairs_ref, matched_pairs_test = match_dots(dots_image1, dots_image2)

    # Calculate the deviation between corresponding dots
    deviations = [calculate_deviation(dot1, dot2) for dot1, dot2 in zip(matched_pairs_ref, matched_pairs_test)]

    # Display the deviations
    for i, deviation in enumerate(deviations):
        print(f"Deviation for dot pair {i + 1}: {deviation:.2f} pixels")

    # Visualization
    visualize_results(image1.copy(), image2.copy(), matched_pairs_ref, matched_pairs_test)

if __name__ == "__main__":
    main()
