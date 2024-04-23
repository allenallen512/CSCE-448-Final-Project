import cv2
import numpy as np
import math
import os
import errno

SOURCE_FOLDER = "../Images/"
OUT_FOLDER = "../Results/"

# def boundary(img, x, y, window):
#     img_height, img_width = img.shape[0], img.shape[1]
#     x_left, x_right, y_top, y_bottom = x - window[0], x + window[0], y - window[1], y + window[1]

#     if x_left < 0: x_left = 0
#     if x_right >= img_width: x_right = img_width - 1
#     if y_top < 0: y_top = 0
#     if y_bottom >= img_height: y_bottom = img_height - 1

#     return x_left, x_right, y_top, y_bottom

def boundary(img, x, y, window):
    # Ensure x, y, and window are scalars/integers
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        raise ValueError("x and y must be scalar integers")
    if not isinstance(window, tuple):
        raise ValueError("window must be a tuple of two integers")

    print(f"x: {x}, y: {y}, window: {window}")  # Debugging output

    img_height, img_width = img.shape[:2]
    x_left = max(x - window[0], 0)
    x_right = min(x + window[0] + 1, img_width)
    y_top = max(y - window[1], 0)
    y_bottom = min(y + window[1] + 1, img_height)
    
    return x_left, x_right, y_top, y_bottom


def compute_norm(matrix):
    matrix = matrix ** 2
    return np.sqrt(np.sum(matrix))


def compute_priority(img, fill_front, mask, window):
    conf = compute_confidence(fill_front, window, mask, img)

    sobel_map = cv2.Sobel(src=mask.astype(float), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_map_norm = compute_norm(sobel_map)
    sobel_map /= sobel_map_norm

    return fill_front * conf * sobel_map, conf


def invert(mask):
    return 1-mask

def compute_confidence(contours, window, mask, img):

    confidence = invert(mask).astype(np.float64)
        
    for i in range(len(contours)):
        for j in range(len(contours[0])):
            if contours[i][j] != 1:
                continue
            x_left, x_right, y_top, y_bottom = boundary(img, j, i, window)
            sumPsi = np.sum(confidence[y_top:y_bottom + 1 , x_left:x_right + 1])
            magPsi = (x_right - x_left) * (y_bottom - y_top)
            if magPsi > 0:
                confidence[i, j] = sumPsi / magPsi

    return confidence
        
def find_best_match(img, mask, patch_size, priorityCoord):
    best_ssd = float('inf')
    best_match = []
    inverted_mask = invert(mask)

    x_coord = int(priorityCoord[0])
    y_coord = int(priorityCoord[1])
    window_size = (10, 10)

    xl, xr, yt, yb = boundary(image, x_coord, y_coord, window_size)
    # xl, xr, yt, yb = boundary(img, priorityCoord[0], priorityCoord[1], patch_size)
    if xl is None:
        return None

    # target_patch = inverted_mask[yt:yb + 1, xl:xr + 1]
    target_patch = mask[yt:yb + 1, xl:xr + 1]
    if target_patch.size == 0:
        return None
    
    target_img = img[yt:yb + 1, xl:xr + 1] * target_patch

    for y in range(image.shape[0]): 
        for x in range(image.shape[1]):
            x_left, x_right, y_top, y_bottom = boundary(img, x, y, patch_size)
            maskPatch = inverted_mask[y_top:y_bottom + 1, x_left: x_right + 1]

            if np.any(maskPatch == 0):
                continue
            
            candidatePatch = image[y_top:y_bottom + 1, x_left:x_right + 1] * target_patch
    
            difference = np.linalg.norm(target_img - candidatePatch)
            if difference < best_ssd:
                best_ssd = difference
                best_match = [x_left, x_right, y_top, y_bottom]
                
    return best_match


#Original Code from Online Github - Need to Change
def compute_fill_front(mask):
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fill_front = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    fill_front = cv2.drawContours(fill_front, contours, -1, (255, 255, 255), thickness=1) / 255.
    return fill_front.astype('uint8')

def update_mask_image(image, mask, best_match, target_image, target_mask, source_mask, max_indices):
    sx, sy = best_match
    source_image = image[sx[0]: sx[1]+1, sy[0]:sy[1]+1]

    # Update the image.
    image[max_indices[0][0]: max_indices[0][1],
          max_indices[1][0]: max_indices[1][1]] = source_image * target_mask + target_image * source_mask

    # Update the mask.
    mask[max_indices[0][0]: max_indices[0][1],
          max_indices[1][0]: max_indices[1][1]] = 0 # Fill with black.

    return image, mask
                

def erase(image, mask, window=(9, 9)):
    mask = (mask / 255).round().astype(np.uint8)
    image_dims = image.shape[:2]
    confidence = (1 - mask).astype(np.float64)
    still_processing = math.inf
    
    while still_processing != 0:
        lab_image = cv2.cvtColor((image.astype(np.float32) / 256), cv2.COLOR_BGR2LAB)
        
        # Compute the fill front.
        fill_front = compute_fill_front(mask)
        
        # Compute priority for each point in the fill front.
        priority, updated_confidence = compute_priority(image, fill_front, mask, window)
        
        # Identify the point with the highest priority.
        max_priority_idx = np.unravel_index(np.argmax(priority), priority.shape)
        max_y, max_x = max_priority_idx
        
        # Define the region around the point with highest priority.
        half_window = (window[0] // 2, window[1] // 2)
        x1, x2 = max(max_x - half_window[0], 0), min(max_x + half_window[0] + 1, image_dims[1])
        y1, y2 = max(max_y - half_window[1], 0), min(max_y + half_window[1] + 1, image_dims[0])
        
        # Extract the target regions from the image and mask.
        target_image = image[y1:y2, x1:x2]
        target_image_lab = lab_image[y1:y2, x1:x2]
        target_mask = mask[y1:y2, x1:x2, np.newaxis].repeat(3, axis=2)
        
        # Find the best match to replace the target region.
        source_mask = 1 - target_mask
        best_match_region = find_best_match(image, target_image_lab, source_mask, (y1, y2, x1, x2))
        
        # Update the confidence map and the image/mask.
        front_points = np.argwhere(target_mask[:, :, 0] == 1)
        confidence[front_points[:, 0] + y1, front_points[:, 1] + x1] = confidence[max_y, max_x]
        image, mask = update_mask_image(image, mask, best_match_region, target_image, target_mask, source_mask, [x1, x2, y1, y2])
        
        still_processing = mask.sum()
        print(f"Remaining pixels to paint: {still_processing}")

    return image.astype(np.uint8)


if __name__ == '__main__':

    output_dir = os.path.join(OUT_FOLDER)
    
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
    image_name = SOURCE_FOLDER + 'target_01.jpg'
    mask_name = SOURCE_FOLDER + 'mask_01.jpg'

    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

    output = erase(image, mask, window=(22,22))
    cv2.imwrite(OUT_FOLDER + 'result_01.png', output)




