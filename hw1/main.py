"""This program performs affine and projective transform on a quadrilateral
region of an input image to turn it into a rectangular output image. The 
quadrilateral region is selected by choosing four points in the input image.

Typical usage example:
  python main.py -i input.jpg -o output.jpg
"""

import argparse
import numpy as np
import cv2
import sys
import os




def set_points_callback(event, x, y, flags, in_points):
    """ Captures and processes interactions with the OpenCV window.
    Args:
      event: OpenCV event value
      x: x-coordinate of mouse on opencv image
      y: y-coordinate of mouse on opencv image
      flags: unused parameter required by OpenCV window callback
      in_points: list for storing image coordinates
    """
    if len(in_points) < 4 and event == cv2.EVENT_LBUTTONDBLCLK:
        print(f" - ({x}, {y})")
        in_points.append(np.array([x, y], np.float32))


def init_gui(in_image, in_points, args):
    """ Initializes the GUI using OpenCV windows and the input image.
    Args:
      in_image: input image to be displayed
      args: program arguments containing gui parameters
    """
    in_height, in_width, _ = in_image.shape
    left_display_width = int(args.disp_height*float(in_width)/float(in_height))
    cv2.namedWindow(args.in_win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(args.in_win_name, args.disp_offset[0], args.disp_offset[1])
    cv2.resizeWindow(args.in_win_name, left_display_width, args.disp_height)
    cv2.setMouseCallback(args.in_win_name, set_points_callback, in_points)
    cv2.namedWindow(args.out_win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(args.out_win_name, args.disp_offset[0]+left_display_width+1,
                   args.disp_offset[1])


def sort_corner_points(points):
    """ Sorts four corner points in a clockwise order starting from the top-left
    corner (upper-left most point).
    Args:
      points: list of points to be sorted
    """
    points.sort(key = lambda pt : pt[1])
    points[0:2] = sorted(points[0:2], key = lambda pt : pt[0])
    points[2:4] = sorted(points[2:4], key = lambda pt : pt[0], reverse = True)


def draw_on_gui(image_display, in_points):
    """ Draws points and lines in the display image for visual feedback.
    Args:
      image_display: image to render points and lines onto
      in_points: points to be rendered
    """
    for u, v in in_points:
        cv2.circle(image_display, (int(u),int(v)), 3, (0,255,0), -1)

    num_pts = len(in_points)
    if num_pts == 4:
        for idx in range(num_pts):
            pt1 = in_points[idx % num_pts]
            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = in_points[(idx+1) % num_pts]
            pt2 = int(pt2[0]), int(pt2[1])
            cv2.line(image_display, pt1, pt2, (0,255,0), 2)


def select_corner_points(in_image, in_points, window_name):
    """ Updates the display window for the input image and handles selection
    of the corner points for computation of the projection matrix.
    Args:
      in_points: list of corner points
      in_image: input image
      window_name: window to display
    """
    print("> Selected points:")
    KEY = {"ENTER" : 13, "ESC" : 27, "Q" : 113, "R" : 114}
    processed = False
    while True:
        image_display = in_image.copy()
        if len(in_points) == 4:
            if not processed:
                sort_corner_points(in_points)
                print("> Press ENTER to process selected region.")
                processed = True

            if key == KEY["ENTER"]:
                return
            
        draw_on_gui(image_display, in_points)
        cv2.imshow(window_name, image_display)
        key = cv2.waitKey(20) & 0xFF
        if key == KEY["ESC"] or key == KEY["Q"]:
            sys.exit()

        if key == KEY["R"]:
            in_points.clear()
            processed = False
            print("> Previous points cleared.")
            print("> Selected points:")


def get_quad_size(points):
    """ Computes the average lengths of opposite sides of a quadrilateral region.
    Args:
      points: list of corner points

    Returns:
      a tuple of the average lengths
    """
    w1 = np.sum((points[0]-points[1])**2)**0.5
    h1 = np.sum((points[1]-points[2])**2)**0.5
    w2 = np.sum((points[2]-points[3])**2)**0.5
    h2 = np.sum((points[3]-points[0])**2)**0.5
    return int((h1+h2)/2), int((w1+w2)/2)


def compute_transformation_matrix(in_points):
    """ Computes transformation matrix from a list of input corner points and
    a list assumed projected rectangular region corner points.
    Args:
      in_points: list of points

    Returns:
      3x3 affine and projective transformation matrix
    """
    in_points = [np.array([x, y, 1], np.float32) for x, y in in_points]
    h, w = get_quad_size(np.vstack(in_points))
    output_points = np.array([[0,   0,   1],
                              [w-1, 0,   1],
                              [w-1, h-1, 1],
                              [0,   h-1, 1]], dtype=np.float32)
    P, res, rnk, s = np.linalg.lstsq(in_points, output_points, rcond=None)
    return P




if __name__ == "__main__":

    # Program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="input.jpg",
                                        help="input image path")
    parser.add_argument("-o", "--output", default="output.jpg",
                                        help="output image path")
    parser.add_argument("-dh", "--disp_height", default=600,
                                        help="display height")
    parser.add_argument("-do", "--disp_offset", default=(100, 100),
                                        help="display offset")
    parser.add_argument("--in_win_name", default="Input Image",
                                        help="input window title")
    parser.add_argument("--out_win_name", default="Output Image",
                                        help="output window title")
    args = parser.parse_args()

    # Read the input image and initialize input points
    assert os.path.exists(args.input), "Input image path not found."
    in_image = cv2.imread(args.input)
    in_height, in_width, in_channel = in_image.shape
    in_points = []

    # Display the input and output windows
    init_gui(in_image, in_points, args)

    # Select corner points from input image
    print("> Instructions:\n",
          "- Select four (4) points in the image by double-clicking on them.\n",
          "- Press ESC or Q to exit the program anytime.\n",
          "- Press R to clear currently selected points.")

    select_corner_points(in_image, in_points, args.in_win_name)

    # Compute transformation matrix
    assert len(in_points) == 4, "Input points must be size 4."
    P = compute_transformation_matrix(in_points)
    print("> Transformation Matrix:\n", P)

    # Map input coordinates to output coordinates using projection matrix
    in_height, in_width, _ = in_image.shape
    in_coordinates = [np.array([x, y, 1], dtype=np.float32)
                      for y in range(in_height) for x in range(in_width)]
    in_coordinates = np.vstack(in_coordinates)
    out_coordinates = np.matmul(in_coordinates, P)

    # Map pixels from input image to the output image;
    #  output image size is based on selected corners
    out_height, out_width = get_quad_size(in_points)
    out_image = np.zeros((out_height, out_width, in_channel), np.uint8)
    for y in range(in_height):
        for x in range(in_width):
            xp, yp, zp = out_coordinates[y*in_width + x]
            xp, yp = int(xp), int(yp)
            if 0 <= xp < out_width and 0 <= yp < out_height:
                out_image[yp, xp, :] = in_image[y, x, :]

    # Save and display the output image
    cv2.imwrite(args.output, out_image)
    cv2.imshow(args.out_win_name, out_image)
    cv2.resizeWindow(args.out_win_name, args.disp_height*out_width//out_height,
                     args.disp_height)
    print("> Press any key to exit the program.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()