import numpy as np
import cv2
import argparse
import sys


def set_points(event, x, y, flags, param):
    """
    This function is callback for processing interactions
    with the opencv gui
    """
    global in_points, image_display
    if len(in_points) < 4:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            in_points.append(np.array([x, y, 1], dtype=np.float32))


def draw_points(image_display, in_points):
    """
    This function renders points and lines in the input image
    for visual feedback
    """
    for point in in_points:
        x, y, _ = point
        cv2.circle(image_display, (int(x),int(y)), 2, (0,255,0), -1)

    num_pts = len(in_points)
    if num_pts == 4:
        for idx in range(num_pts):
            pt1 = in_points[idx % num_pts]
            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = in_points[(idx+1) % num_pts]
            pt2 = int(pt2[0]), int(pt2[1])
            cv2.line(image_display, pt1, pt2, (0,255,0), 1)


def sort_corner_points(points):
    """
    This function sorts the corner points in a clockwise order starting
    from the top-left corner (upper-left most point)
    """
    points.sort(key = lambda pt : pt[1])
    points[0:2] = sorted(points[0:2], key = lambda pt : pt[0])
    points[2:4] = sorted(points[2:4], key = lambda pt : pt[0], reverse = True)


def compute_projection_matrix(in_points, args):
    """
    Computes projection matrix from a list of corner points
    and assumed output rectangular points.
    """
    assert len(in_points) == 4, "You must select four (4) corner points \
                                by double clicking on each desired corner."
    h, w = get_rectangle_size(np.vstack(in_points))
    output_points = np.array([[0, 0, 1],
                              [w-1, 0, 1],
                              [w-1, h-1, 1],
                              [0, h-1, 1]], dtype=np.float32)
    P, res, rnk, s = np.linalg.lstsq(in_points, output_points, rcond=None)
    return P


def get_rectangle_size(points):
    """
    Computes the average length of opposite sides of a
    quadrilateral
    """
    w1 = np.sum((points[0]-points[1])**2)**0.5
    h1 = np.sum((points[1]-points[2])**2)**0.5
    w2 = np.sum((points[2]-points[3])**2)**0.5
    h2 = np.sum((points[3]-points[0])**2)**0.5
    return int((h1+h2)/2), int((w1+w2)/2)


def set_corner_points(points):
    global in_image
    image_backup = in_image.copy()
    image_display = in_image.copy()
    key = None
    processed = False
    while(key not in [13, 27, 113]):
        image_display = image_backup
        cv2.imshow(in_win_name, image_display)
        if len(in_points) == 4 and not processed:
            sort_corner_points(in_points)
            processed = True
        draw_points(image_display, in_points)
        key = cv2.waitKey(20) & 0xFF

    return key


if __name__ == "__main__":
    # Program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="book.jpg",
                                        help='input image path')
    parser.add_argument("-o", "--output", default="output.jpg",
                                        help='output image path')
    args = parser.parse_args()

    # Display settings
    disp_height = 600
    disp_offset = (100, 100) # horizontal, vertical
    in_win_name = "Input Image"
    out_win_name = "Output Image"

    # Read image
    in_image = cv2.imread(args.input)
    in_height, in_width, _ = in_image.shape
    left_display_width = int(disp_height*float(in_width)/float(in_height))

    # Show input and output windows
    cv2.namedWindow(in_win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(in_win_name, disp_offset[0], disp_offset[1])
    cv2.resizeWindow(in_win_name, left_display_width, disp_height)
    cv2.setMouseCallback(in_win_name, set_points)
    cv2.namedWindow(out_win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(out_win_name, disp_offset[0]+in_width+1, disp_offset[1])

    # Select corner points from input image
    in_points = []
    key = set_corner_points(in_points)
    if key in [27, 113]:
        sys.exit()

    # Compute projection matrix
    P = compute_projection_matrix(in_points, args)

    # Map input coordinates to output coordinates
    in_coordinates = [np.array([x, y, 1], dtype=np.float32) 
                        for y in range(in_height) for x in range(in_width)]
    in_coordinates = np.vstack(in_coordinates)
    out_coordinates = np.matmul(in_coordinates, P)

    # Map pixels from input image to the output image;
    #  size is based on selected corners
    out_height, out_width = get_rectangle_size(in_points)
    out_image = np.zeros((out_height, out_width, 3), np.uint8)
    for y in range(in_height):
        for x in range(in_width):
            xp, yp, zp = out_coordinates[y*in_width + x]
            xp, yp = int(xp/zp), int(yp/zp)
            if 0 <= xp < out_width and 0 <= yp < out_height:
                out_image[yp, xp, :] = in_image[y, x, :]

    # Display and save output image
    cv2.imshow(out_win_name, out_image)
    cv2.resizeWindow(out_win_name, disp_height*out_width//out_height, disp_height)
    cv2.waitKey(0)
    cv2.imwrite(args.output, out_image)

    cv2.destroyAllWindows()