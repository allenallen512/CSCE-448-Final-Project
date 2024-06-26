import cv2
import numpy as np
import math
import os
import errno

SOURCE_FOLDER = "../Images/"
OUT_FOLDER = "../Results/"


def get_boundary(img, x, y, window):
    # Ensure x, y, and window are scalars/integers
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        raise ValueError("x and y must be scalar integers")
    if not isinstance(window, tuple):
        raise ValueError("window must be a tuple of two integers")

    # print(f"x: {x}, y: {y}, window: {window}")  # Debugging output

    img_height, img_width = img.shape[:2] #changed this line
    # print("the height and width of the image is: ", img_height, img_width)
    x_left = max(x - (window[0] // 2), 0) if (x - (window[0] // 2)) >= 0 else x
    x_right = min(x + (window[0] // 2), img_width - 1) #if (x + (window[0] // 2)) <= img_width else x
    y_top = max(y - (window[1] // 2), 0) if (y - (window[1] // 2)) >= 0 else y
    y_bottom = min(y + (window[1] // 2), img_height - 1) if (y + (window[1] // 2)) <= img_height else y 
    
    return x_left, x_right, y_top, y_bottom

# Need the scalar norm value of a matrix
def compute_norm(matrix):
    matrix = matrix ** 2
    return np.sqrt(np.sum(matrix))


def calculate_priority(img, fill_front, mask, window):
    # print("the max of the image: ", np.max(image))
    # print("the max of the mask in the priority function: ", np.max(mask))
    conf = get_confidence(fill_front, window, mask, img)
    # Using the sobel operator to detect edges then compute the norm
    sobel_map = cv2.Sobel(src=mask.astype(float), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_map_norm = compute_norm(sobel_map)
    sobel_map /= sobel_map_norm

    return fill_front * conf * sobel_map, conf

# Invert the mask for purposes of taking element-wise product of matrices
def invert(mask):
    return 1-mask

def get_confidence(contours, window, mask, img):
    # Values that are black in the mask are known (0) become 1 so their confidence is also 1
    confidence = invert(mask).astype(np.float64)
    # Loop through countours and find where they are drawn
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j] != 1:
                continue
            # Take area around contour which has the unknown pixels and compute confidence based on known pixels around it
            x_left, x_right, y_top, y_bottom = get_boundary(img, j, i, window)
            # Use formula in the paper
            sumPsi = np.sum(confidence[y_top:y_bottom + 1 , x_left:x_right + 1])
            magPsi = (x_right - x_left) * (y_bottom - y_top)
            if magPsi > 0:
                confidence[i, j] = sumPsi / magPsi
                
    return confidence

# Need the fill front as it show the divide between known and unknown pixels in the image
def get_target_areas(mask):
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fill_front = np.zeros_like(mask)
    # Gives a map of the countours followed by drawing given the mask inputted
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    fill_front = cv2.drawContours(fill_front, contours, -1, (255, 255, 255), thickness=1) / 255.
    # print("the shape of fillfront is: ", fill_front.shape)
    return fill_front.astype('uint8')

def get_best_patch(img, mask, window, priorityCoord):
    # print("the window being used is: ", window)
    best_ssd = float('inf')
    best_match = []
    inverted_mask = invert(mask)

    x_coord = int(priorityCoord[0])
    y_coord = int(priorityCoord[1])

    # print("the starting priority coordinates", f"({x_coord},{y_coord})")
    xl, xr, yt, yb = get_boundary(img, x_coord, y_coord, window)
    # print("the first boundary set: ", xl, xr, yt, yb)
    # xl, xr, yt, yb = boundary(img, priorityCoord[0], priorityCoord[1], patch_size)
    if xl is None:
        return None
    
    target_patch = inverted_mask[yt:yb + 1, xl:xr + 1]
    target_patch = np.stack(target_patch[:,:,np.newaxis]).repeat(3, axis=2) #size to 3D to match the LAB image

    #target patch is the mask on the points we are trying to fill
    # target_patch = mask[yt:yb + 1, xl:xr + 1]
    if target_patch.size == 0:
        return None
    
    target_img = img[yt:yb + 1, xl:xr + 1] * target_patch
    # print("the size of the target image is: ", target_img.shape)

    for y in range(window[0] // 2, img.shape[0] - window[0] // 2): #i made it only go to half the radius from the edge to stop edge cases
        for x in range(window[1] // 2, img.shape[1] - window[1] // 2):
            # print("checking coordinates", f"x: {x}, y: {y}")
            # print("the size of the whole image is: ", img.shape)

            x_left, x_right, y_top, y_bottom = get_boundary(img, x, y, window)
            # print(x_left, x_right, y_top, y_bottom)
            maskPatch = inverted_mask[y_top:y_bottom + 1, x_left: x_right + 1]

            if np.any(maskPatch == 0):
                continue
            # print("the shape of the current image is: ", img[y_top:y_bottom + 1, x_left:x_right + 1].shape)
            # print("the target patch shape before resizing: ", target_patch.shape)
            target_patch_resized = np.resize(target_patch, img[y_top:y_bottom + 1, x_left:x_right + 1].shape)
            # print("the target patch resized shape is: ", target_patch_resized.shape)
            
            # print(f"{x_left},{x_right},{y_top},{y_bottom}")
            candidatePatch = img[y_top:y_bottom + 1, x_left:x_right + 1] * target_patch_resized
            target_image_resized = np.resize(target_img, candidatePatch.shape)
            # print("the size of the target image is: ", target_img.shape)
            difference = np.linalg.norm(target_image_resized - candidatePatch)
            if difference < best_ssd:
                best_ssd = difference
                best_match = [x_left, x_right, y_top, y_bottom]
                
    return best_match




def update(image, mask, bestRegion, updateRegion, updateRegionIndex, targetMask, windowSize):
    '''
    need the image, mask, best matching region from find best match
    update region is the region we want to update
    updateRegionIndex are the index points. this can be combined with the ones aove
    targetMask is the mask where inside the mask is 1 and outside of 0
    source mask is just the inverse of the
    '''
    # print("the sum of all points in mask is: ", np.sum(mask))
    targetMask = np.stack(targetMask[:,:,np.newaxis]).repeat(3, axis=2) #size to 3D to match the LAB image

    invertedMask = 1 - targetMask
    lowX, highX, lowY, highY = bestRegion[0], bestRegion[1], bestRegion[2], bestRegion[3]
    # print("the shape of target mask: ", targetMask.shape)
    sourceImageCopy = image[lowY:highY+1,lowX:highX+1] #the part of the image we want to duplicate into the target image
    print("the size of the source image copy in update: ", sourceImageCopy.shape)
    print("the size of the target mask in update: ", targetMask.shape)
    if sourceImageCopy.shape != targetMask.shape:
        if sourceImageCopy.size > targetMask.size:
            sourceImageCopy = sourceImageCopy[:targetMask.shape[0], :targetMask.shape[1]]
        else:
            targetMask = targetMask[:sourceImageCopy.shape[0], :sourceImageCopy.shape[1]]
    newRegion = sourceImageCopy * targetMask #tarrget mask is just the regular mask inside of the box we want to fill
    oldRegion = invertedMask * updateRegion
    print("the shape of new region: " , newRegion.shape, " and the shape of old: ", oldRegion.shape)
    lowerXFill, upperXFill, lowerYFill, upperYFill = updateRegionIndex[0], updateRegionIndex[1], updateRegionIndex[2], updateRegionIndex[3]
    print("lowerXfill: ", lowerXFill, " upper XFill: ", upperXFill, " lower YFill: ", lowerYFill, " upper Y Fill: ", upperYFill)
    mask[lowerYFill:upperYFill, lowerXFill:upperXFill] = 0
    image[lowerYFill:upperYFill, lowerXFill:upperXFill] = newRegion + oldRegion
    # print("the sum of all points in mask after the update is:", np.sum(mask))
    

    return mask, image
                

def erase_and_fill_algorithm(image, mask, window):
    # print("the shape of the image in the erase function is: " , image.shape)
    mask = (mask / 255).round().astype(np.uint8)
    image_dims = image.shape[:2]
    confidence = (1 - mask).astype(np.float64)
    still_processing = math.inf
    
    while still_processing != 0:
        lab_image = cv2.cvtColor((image.astype(np.float32) / 256), cv2.COLOR_BGR2LAB)
        
        # Compute the fill front.
        fill_front = get_target_areas(mask)
        
        non_zero_indices = np.argwhere(fill_front > 0)
        mask_indicies = np.argwhere(mask > 0)

        if still_processing < 30:
            print("Non-zero indices in fill_front:")
            for index in non_zero_indices:
                print(index, " and the value here is: ", fill_front[tuple(index)])
                
        if still_processing < 30:
            print("Non-zero indices in mask:")
            for index in mask_indicies:
                print(index, " value of the mask is ", mask[tuple(index)])        
        
        # Compute priority for each point in the fill front.
        priority, updated_confidence = calculate_priority(image, fill_front, mask, window)
        print("the maximum priority is: ", np.max(priority))   
   
        # Identify the point with the highest priority.
        max_priority_idx = np.unravel_index(np.argmax(priority), priority.shape)
        max_y, max_x = max_priority_idx
        
        if (np.max(priority) == 0):
            max_y = tuple(non_zero_indices[0])[0]
            max_x = tuple(non_zero_indices[0])[1]
            
        print("the max y being used is: ", max_y)
        print("the max x being used is: ", max_x)
        
        # Define the region around the point with highest priority.
        half_window = (window[0] // 2, window[1] // 2)
        x1, x2 = max(max_x - half_window[0], 0), min(max_x + half_window[0]  + 1, image_dims[1]) #removed plus from from both min first values
        y1, y2 = max(max_y - half_window[1], 0), min(max_y + half_window[1] + 1, image_dims[0])
        
        # Extract the target regions from the image and mask.
        target_image = image[y1:y2, x1:x2]
        #target_image_lab = lab_image[y1:y2, x1:x2]
        target_mask = mask[y1:y2, x1:x2, np.newaxis].repeat(3, axis=2)
        target_mask_1D = mask[y1:y2, x1:x2]
        print("y1: ", y1, "the y2: ", y2, "the x1 ", x1, "the x2: ", x2)
        print("the shape of target mask 1D in erase: ", target_mask_1D.shape)
        
        # Find the best match to replace the target region.
        source_mask = 1 - target_mask_1D
        
        # print("the shape of source mask is: ", source_mask.shape)
        best_match_region = get_best_patch(lab_image, mask, window, (max_x, max_y))
        print("TRYING TO FILL: ", max_x, max_y, " USING: ", best_match_region)
        # print("the best region match is: ", best_match_region)
        update_Region_Index = [x1, x2, y1, y2]
        # Update the confidence map and the image/mask.
        mask, image = update(image, mask, best_match_region, target_image, update_Region_Index, target_mask_1D, [x1, x2, y1, y2])
#after getting the new masks is when we will update the confidence of all the points   
        print("the shape of the updated image is: ", image.shape, " and the shape of the new mask: ", mask.shape)
        
        # updating the confidence
        front_points = np.argwhere(target_mask == 1)
        updated_confidence[front_points] = updated_confidence[max_y, max_x]
        
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
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.cvtColor(image)

    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    print("the mask shape is: ",mask.shape)

    output = erase_and_fill_algorithm(image, mask, window=(10,10))
    # output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(OUT_FOLDER + 'result_01.png', output)




