import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def segment_image_by_color(image_path, n_clusters):
    """
    Segment an image based on colors using K-means clustering in HSV color space
    """
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Reshape image to a 2D array of pixels
    h, w, _ = img_hsv.shape
    pixels = img_hsv.reshape(-1, 3)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Reshape labels back to the image shape
    segmented = labels.reshape(h, w)

    # Get cluster centers
    centers_hsv = kmeans.cluster_centers_

    return img_rgb, segmented, centers_hsv


def apply_enhancement_based_on_colorblind_type(hsv_color, colorblind_type):

    h, s, v = hsv_color
    new_hsv = hsv_color.copy()

    # red-green colorblindness
    if colorblind_type == 'deuteranopia':
        if 60 <= h <= 180:  # greens and cyans
            new_hsv[0] = max(0, min(179, (h * 0.7) % 180))  # shift green toward blue
            new_hsv[1] = min(255, s * 1.3)  # increase saturation
        elif 0 <= h < 60:  # reds to yellows
            new_hsv[0] = max(0, min(179, (h * 1.3) % 180))  # stretch reds
            new_hsv[1] = min(255, s * 1.2)  # increase saturation

    # Red-green variant
    elif colorblind_type == 'protanopia':
        if 0 <= h <= 30 or 150 <= h <= 179:  # reds and magentas
            new_hsv[0] = (h + 30) % 180  # shift toward yellow/blue
            new_hsv[1] = min(255, s * 1.3)  # increase saturation
        elif 40 <= h <= 80:  # yellows and yellow-greens
            new_hsv[0] = max(30, h - 10)  # enhance yellow

    # Blue-yellow colorblindness
    elif colorblind_type == 'tritanopia':
        if 30 <= h <= 90:  # yellows and greens
            new_hsv[0] = (h - 15) % 180  # shift toward red
            new_hsv[1] = min(255, s * 1.2)  # increase saturation
        elif 180 <= h <= 270:  # blues and purples
            new_hsv[0] = (h + 15) % 180  # shift toward green
            new_hsv[1] = min(255, s * 1.2)  # increase saturation

    # Complete color blindness
    elif colorblind_type == 'monochromacy':
        new_hsv[0] = 0  # no hue
        new_hsv[1] = 0  # no saturation
        # Enhance brightness contrast
        if v < 128:
            new_hsv[2] = max(0, v * 0.7)  # darken dark colors
        else:
            new_hsv[2] = min(255, v * 1.2)  # brighten light colors

    return new_hsv


def apply_colors_to_segments(img_rgb, segmented, centers_hsv, transform_colors=False, colorblind_type=None):
    """
    Apply colors to segments - either original colors or transformed ones.
    """
    modified_img = img_rgb.copy()

    for i in range(len(centers_hsv)):

        # Creates a boolean array where True = pixels belonging to segment i
        # Example: If segment 2 represents "red areas", mask shows where all red pixels are located
        mask = (segmented == i)

        # Apply colorblind Transformation if required
        if transform_colors and colorblind_type:
            centers_hsv[i] = apply_enhancement_based_on_colorblind_type(centers_hsv[i], colorblind_type)

        # HSV -> RGB
        center_rgb = cv2.cvtColor(np.uint8([[centers_hsv[i]]]), cv2.COLOR_HSV2RGB)[0][0]
        # Replace all pixels in this segment with the transformed color
        modified_img[mask] = center_rgb

    return modified_img


def visualize_results(original_img, segmented_img, colorblind_img, colorblind_type):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(segmented_img)
    axes[1].set_title('Color Segmented')
    axes[1].axis('off')

    axes[2].imshow(colorblind_img)
    axes[2].set_title(f'Colorblind Accessible ({colorblind_type})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Insert path to an image
    image_path = ""

    # Segment image by colors
    original_img, segmented, centers_hsv = segment_image_by_color(image_path, n_clusters=8)

    segmented_img = apply_colors_to_segments(original_img, segmented, centers_hsv)

    # Transform for different types of colorblindness
    colorblind_types = ['deuteranopia', 'protanopia', 'tritanopia', 'monochromacy']
    
    for cb_type in colorblind_types:
        colorblind_img = apply_colors_to_segments(original_img, segmented, centers_hsv, transform_colors=True, colorblind_type=cb_type)
        visualize_results(original_img, segmented_img, colorblind_img, cb_type)
