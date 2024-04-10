import cv2
import numpy as np

# Patch Extraction
def get_patches_around_boundary(image, mask, patch_size):
    # Patch size has to be odd
    if patch_size % 2 == 0:
        patch_size += 1

    half_patch = patch_size // 2

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_patches = []

    for contour in contours:
        for point in contour:
            x, y = point[0]
            
            # Check if the patch around this point is fully within the image bounds
            if y - half_patch >= 0 and y + half_patch < image.shape[0] and x - half_patch >= 0 and x + half_patch < image.shape[1]:
                patch = image[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                boundary_patches.append(((x, y), patch))

    return boundary_patches



def compute_priority(patch, grad_mag):
    # Data term: Here, we use the sum of gradient magnitudes within the patch as a proxy
    data_term = np.sum(grad_mag)

    # Confidence term: For simplicity, let's assume a fixed confidence for this example
    confidence_term = 1.0  # In a real implementation, this would vary based on already inpainted regions

    # Combine the terms (many approaches are possible; this is just one)
    priority = data_term * confidence_term
    return priority

def find_best_match(image, target_patch, mask):
    best_ssd = float('inf')
    best_match = None
    patch_size = target_patch.shape[0]
    half_patch_size = patch_size // 2

    for y in range(half_patch_size, image.shape[0] - half_patch_size):
        for x in range(half_patch_size, image.shape[1] - half_patch_size):
            # Skip patches that overlap with the mask
            if np.any(mask[y - half_patch_size:y + half_patch_size + 1, x - half_patch_size:x + half_patch_size + 1]):
                continue

            candidate_patch = image[y - half_patch_size:y + half_patch_size + 1, x - half_patch_size:x + half_patch_size + 1]
            ssd = np.sum((candidate_patch - target_patch) ** 2)

            if ssd < best_ssd:
                best_ssd = ssd
                best_match = (x, y)

    return best_match


def inpaint(image, mask, patch_size):
    # Convert the mask to a binary format if it isn't already
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    while np.any(mask):
        boundary_patches = get_patches_around_boundary(image, mask, patch_size)

        highest_priority = -1
        best_patch = None
        best_patch_pos = None

        for pos, patch in boundary_patches:
            grad_patch = grad_mag[pos[1] - patch_size // 2:pos[1] + patch_size // 2 + 1,
                                  pos[0] - patch_size // 2:pos[0] + patch_size // 2 + 1]
            priority = compute_priority(patch, grad_patch)

            if priority > highest_priority:
                highest_priority = priority
                best_patch = patch
                best_patch_pos = pos

        best_match_pos = find_best_match(image, best_patch, mask)
        x, y = best_patch_pos
        match_x, match_y = best_match_pos

        # Copy the best matching patch into the target position
        image[y - patch_size // 2:y + patch_size // 2 + 1, x - patch_size // 2:x + patch_size // 2 + 1] = \
            image[match_y - patch_size // 2:match_y + patch_size // 2 + 1, match_x - patch_size // 2:match_x + patch_size // 2 + 1]

        # Update the mask
        mask[y - patch_size // 2:y + patch_size // 2 + 1, x - patch_size // 2:x + patch_size // 2 + 1] = 0

    return image



if __name__ == '__main__':
    # # Parameters
    # patch_size = 9  # Example patch size

    # # Compute the gradient magnitude of the image (for the data term)
    # grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # grad_mag = cv2.magnitude(grad_x, grad_y)

    # # Get patches around the boundary
    # boundary_patches = get_patches_around_boundary(image, mask, patch_size)

    # # Compute priority for each patch (for illustration; in practice, this ties into the inpainting loop)
    # for pos, patch in boundary_patches:
    #     # Extract the corresponding gradient magnitude patch
    #     grad_patch = grad_mag[pos[1] - patch_size // 2:pos[1] + patch_size // 2 + 1,
    #                         pos[0] - patch_size // 2:pos[0] + patch_size // 2 + 1]
        
    #     priority = compute_priority(patch, grad_patch)
    #     print(f"Patch at {pos} has priority {priority}")

    # Load the image and mask
    image = cv2.imread('path_to_your_image.jpg', 0)  # Load as grayscale for simplicity
    mask = cv2.imread('path_to_your_mask.jpg', 0)  # Assume mask is also grayscale

    # Ensure the mask is a binary image
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Define patch size
    patch_size = 9

    # Perform inpainting
    inpainted_image = inpaint(image, mask, patch_size)

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Inpainted Image", inpainted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




