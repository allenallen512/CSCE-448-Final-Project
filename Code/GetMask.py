import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import laplace


#got this function from homework3
def GetMask(image):
    ### You can add any number of points by using 
    ### mouse left click. Delete points with mouse
    ### right click and finish adding by mouse
    ### middle click.  More info:
    ### https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ginput.html

    plt.imshow(image)
    plt.axis('image')
    points = plt.ginput(-1, timeout=-1)
    plt.close()

    ### The code below is based on this answer from stackoverflow
    ### https://stackoverflow.com/a/15343106

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([points], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    return mask

def getGradient(mask):
    # copy the mask -- only used for viewing purposes
    mask_with_contour = mask.copy()
    
    #mmake into grey
    if len(mask.shape) > 2:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    # Find contours
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_with_contour =  cv2.drawContours(mask_with_contour, contours, 0, (0, 0, 255), 2) 


    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append(tuple(point[0]))
    
    #this will return the points on the edge of the omega gradient as well as a new mask with a red line around the mask area. 
    #the new mask is not really needed tbh. just for viewing
    return edge_points, mask_with_contour

def getFront(mask): 
    #will return an array the same size as mask. only the edges will be one.
    return (laplace(mask) > 0).astype('uint8');
    

if __name__ == '__main__':
    
    imageDir = '../Images/'
    resultDir = '../Results/'
    im1_name = 'target_01.jpg'
    image = plt.imread(imageDir + im1_name)
    
    masked = GetMask(image)
    index = 1
    
    plt.imsave("{}mask_{}.jpg".format(imageDir, str(index).zfill(2)), masked)
    
    mask = plt.imread(imageDir + "mask_01.jpg")
    edge_points, image_with_contour = getGradient(mask)
    print(edge_points)
    # plt.imsave("{}countor_{}.jpg".format(imageDir, str(index).zfill(2)), image_with_contour)


    # Display the image with the red line
    cv2.imshow('Image with Contour', image_with_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 