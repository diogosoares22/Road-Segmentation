import cv2
import math

IMAGE_NUMBER = 100
DATAPATH = "../data/"

sat_image_names = [(DATAPATH + "training/images/satImage_{0:03}.png").format(i) for i in range(1, IMAGE_NUMBER + 1)]
gd_image_names = [(DATAPATH + "training/groundtruth/satImage_{0:03}.png").format(i) for i in range(1, IMAGE_NUMBER + 1)]


def load_images(filenames):
    """ 
    Loads images for specific filenames
    @param filenames: Respective Filenames
    """
    return [cv2.imread(image_name) for image_name in filenames]


def rotate_images(degree, images):
    """ 
    Rotate images with a specific degree
    @param degree: Degree for rotation
    @param images: Respective images
    """
    # All images have the same size
    height, width = images[0].shape[:2]
    center = (width / 2, height / 2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=degree, scale=1)
    return list(map(lambda x: cv2.warpAffine(src=x, M=rotate_matrix, dsize=(width, height)), images))


def get_nearest_multiple(num, k):
    """
    Returns nearest k multiple to num
    @param num: Maximum number
    @param k: Divisor
    """
    return (num // k) * k


def crop_rotated_image(img, w_initial, h_initial, angle, k):
    """
    Crops rotated image to have maximum length
    @param img: Image to crop
    @param w_initial: Initial width 
    @param h_initial: Initial height
    @param angle: Rotation angle
    @param k: Possible patch size
    """
    center = (w_initial / 2, h_initial / 2)
    w_raw, h_raw = rotatedRectWithMaxArea(w_initial, h_initial, angle * (math.pi / 180))
    w, h = get_nearest_multiple(w_raw, k), get_nearest_multiple(h_raw, k)
    return img[int(center[1] - h // 2):int(center[1] + h // 2), int(center[0] - w // 2):int(center[0] + w // 2)]


def mirror_images(images):
    """
    Mirror images
    @param images: Respective images
    """
    return [cv2.flip(image, 0) for image in images]


def save_images(images, filenames):
    """ 
    Saves images for specific filenames
    @param images: Respective images
    @param filenames: Respective Filenames
    """
    for i in range(len(filenames)):
        cv2.imwrite(images[i], filenames[i])


# This code was copied from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    @param w : Rectangle width
    @param h : Rectangle height
    @param angle : Rotated angle 
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr
