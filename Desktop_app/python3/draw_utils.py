import cv2
import numpy as np
from EyeGaze_Python import device_constants


def draw_line(bgr_img, start_point, end_point, color=(255, 0, 0), thickness=2):
    cv2.line(bgr_img, start_point, end_point, color, thickness)


def draw_plus(bgr_img, x, y, colors, thickness):
    h, w = bgr_img.shape[0], bgr_img.shape[1]
    if y < 0 or y > bgr_img.shape[0]:
        cv2.line(bgr_img, (int(x), 0), (int(x), h), (0,0,255), thickness)  # vertical

    elif x < 0 or x > bgr_img.shape[1]:
        cv2.line(bgr_img, (0, int(y)), (w, int(y)), (0,0,255), thickness)  # horizontal
    else:
        cv2.line(bgr_img, (0, int(y)), (w, int(y)), colors, thickness)  # horizontal
        cv2.line(bgr_img, (int(x), 0), (int(x), h), colors, thickness)  # vertical
        cv2.circle(bgr_img, (int(x), int(y)), 6, colors, cv2.FILLED)


def draw_cross(bgr_img, x, y, color=(255, 255, 255), width=1, thickness=1):
    """ Draws an "x"-shaped cross at (x,y)
    """

    x, y, w = int(x), int(y), int(width / 2)  # ensure points are ints for cv2 methods

    cv2.line(bgr_img, (x - w, y - w), (x + w, y + w), color, thickness)
    cv2.line(bgr_img, (x - w, y + w), (x + w, y - w), color, thickness)


def draw_points(bgr_img, points, color=(255, 0, 0), width=2, thickness=1, colorful_points=False):
    """ Draws an "x"-shaped cross at each point in a list
    """
    # black, blue, green, red, white, pink
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 0, 255)]
    if colorful_points:
        for i, point in enumerate(points):
            draw_cross(bgr_img, point[0], point[1], colors[i], width, thickness)
    else:
        for point in points:
            draw_cross(bgr_img, point[0], point[1], color, width, thickness)


def blank_screen(device=None, screen_height=1280, screen_width=720, scale=0.5):
    """ Creates a blank screen image for further drawing on
    """
    screen_w_px, screen_h_px = (screen_width, screen_height) if device is None else device.screen_size_px
    screen_w_px, screen_h_px = screen_w_px - device.screen_x_offset_px, screen_h_px - device.screen_y_offset_px
    img_w, img_h = screen_w_px * scale, screen_h_px * scale

    size = int(16 * scale)
    x0, dx = int(img_w / 6), int(img_w / 3)
    y0, dy = int(img_h / 6), int(img_h / 3)

    screen_img = np.zeros((int(img_h), int(img_w), 3), dtype=np.uint8)

    # drawing the 9 fixation points on the screen
    targets = [(x0 + i * dx, y0 + j * dy) for j in range(3) for i in range(3)]
    map(lambda x: cv2.circle(screen_img, (int(x[0]), int(x[1])), size, (255, 255, 255), -1), targets)

    return screen_img


def draw_center_point(bgr_img, color=(255, 255, 255), thickness=1):
    """Draw a circle at the center of the image"""
    center = (int(bgr_img.shape[1] / 2), int(bgr_img.shape[0] / 2))
    cv2.circle(bgr_img, center, 7, color, thickness)


def draw_gaze(screen_img, gaze_pts, gaze_colors, scale=0.5, return_img=False, cross_size=16, thickness=2):
    """ Draws an "x"-shaped cross on a screen for given gaze points, ignoring missing ones
    """

    width = int(cross_size * scale)

    for i, pt in enumerate(gaze_pts):
        if pt is None: continue
        draw_cross(screen_img, pt[0] * scale, pt[1] * scale, gaze_colors[i], width, thickness)
    if return_img:
        return screen_img


def draw_gaze_v2(bgr_img, gaze_pts, gaze_colors, scale=0.5, return_img=False, cross_size=20, thickness=2):
    x_gaze, y_gaze = gaze_pts[0][0], gaze_pts[0][1]
    draw_plus(bgr_img, x_gaze, y_gaze, gaze_colors, thickness)
    if return_img:
        return bgr_img


def draw_normal(img, limbus, device, arrow_len_mm=20, color=(255, 255, 255), thickness=1, scale=1):
    """ Draws an arrow pointing towards screen transformed by matrix
    """

    focal_len_x_px, focal_len_y_px, prin_point_x, prin_point_y = device.get_intrinsic_cam_params()

    long_normal = list(map(lambda x: x * arrow_len_mm, limbus.normal))
    arrow_pts_mm = [limbus.center_mm, list(map(sum, list(zip(limbus.center_mm, long_normal))))]

    # Mirror the normal in the x direction for drawing onto the video captured as-is by camera
    arrow_trans_x = list(map(lambda v: int((v[0] / v[2] * -focal_len_x_px + prin_point_x) * scale), arrow_pts_mm))
    arrow_trans_y = list(map(lambda v: int((v[1] / v[2] * focal_len_y_px + prin_point_y) * scale), arrow_pts_mm))

    arrow_trans_tuple = list(zip(arrow_trans_x, arrow_trans_y))

    cv2.circle(img, arrow_trans_tuple[0][:2], 3, color, -1)
    cv2.line(img, arrow_trans_tuple[0][:2], arrow_trans_tuple[1][:2], color, thickness)


def draw_normal_v2(img, normal_vector, iris_world, device, arrow_len_mm=20, color=(0, 0, 255), thickness=1, scale=1):
    """ Draws an arrow pointing towards screen transformed by matrix
    """

    focal_len_x_px, focal_len_y_px, prin_point_x, prin_point_y = device.get_intrinsic_cam_params()
    normal_vector2 = list(normal_vector)
    print(f"list of normal vector: {normal_vector2}")
    long_normal = list(map(lambda x: x * arrow_len_mm, normal_vector))
    arrow_pts_mm = [iris_world, list(map(sum, list(zip(iris_world, long_normal))))]
    print(f"debug part: {arrow_pts_mm}")
    # Mirror the normal in the x direction for drawing onto the video captured as-is by camera
    arrow_trans_x = list(map(lambda v: int((v[0] / v[2] * -focal_len_x_px + prin_point_x) * scale), arrow_pts_mm))
    arrow_trans_y = list(map(lambda v: int((v[1] / v[2] * focal_len_y_px + prin_point_y) * scale), arrow_pts_mm))

    arrow_trans_tuple = list(zip(arrow_trans_x, arrow_trans_y))

    cv2.circle(img, arrow_trans_tuple[0][:2], 3, color, -1)
    cv2.line(img, arrow_trans_tuple[0][:2], arrow_trans_tuple[1][:2], color, thickness)


def draw_limbus(img, limbus, color=(255, 255, 255), thickness=1, scale=1):
    """ Draws the 2d ellipse of the limbus
    """

    (ell_x0, ell_y0), (ell_w, ell_h), angle = limbus.ransac_ellipse.rotated_rect

    (ell_x0, ell_y0), (ell_w, ell_h) = (ell_x0 * scale, ell_y0 * scale), (ell_w * scale, ell_h * scale)

    cv2.ellipse(img, ((ell_x0, ell_y0), (ell_w, ell_h), angle), color, thickness)


def draw_eyelids(eyelid_t, eyelid_b, eye_img):
    """ Draws the parabola for top eyelid and line for bottom eyelid (if they exist)
    """

    if eyelid_t is not None:
        a, b, c = eyelid_t
        lid_xs = [x * eye_img.shape[1] / 20 for x in range(21)]
        lid_ys = [a * x ** 2 + b * x + c for x in lid_xs]
        pts_as_array = np.array([[x, y] for (x, y) in zip(lid_xs, lid_ys)], np.int0)
        cv2.polylines(eye_img, [pts_as_array], False, (0, 255, 0))
    if eyelid_b is not None:
        a, b = eyelid_b
        start_pt, end_pt = (0, int(b)), (eye_img.shape[1], int(a * eye_img.shape[1] + b))
        cv2.line(eye_img, start_pt, end_pt, (0, 255, 0))


def draw_histogram(img, bin_width=4):
    """ Calculates and plots a histogram (good for BGR / LAB)
    """

    hist_img = np.zeros((300, 256, 3))

    bin_count = int(256 / bin_width)
    bins = np.arange(bin_count).reshape(bin_count, 1) * bin_width
    debug_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(debug_colors):
        hist_item = cv2.calcHist([img], [ch], None, [bin_count], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(hist_img, [pts], False, col)

    hist_img = np.flipud(hist_img)

    cv2.imshow('hist', hist_img)


def draw_histogram_hsv(hsv_img, bin_width=2):
    """ Calculates and plots 2 histograms next to each other: one for hue, and one for saturation and value
    """

    sv_hist_img, h_hist_img = np.zeros((300, 256, 3)), np.zeros((300, 360, 3))
    sv_bin_count, h_bin_count = int(256 / bin_width), int(180 / bin_width)

    sv_bins = np.arange(sv_bin_count).reshape(sv_bin_count, 1) * bin_width
    h_bins = np.arange(h_bin_count).reshape(h_bin_count, 1) * bin_width * 2

    debug_colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255)]

    # Use ternary conditional for outputting to 2 different hists - a bit of a hack
    for ch, col in enumerate(debug_colors):
        hist_item = cv2.calcHist([hsv_img], [ch], None, [h_bin_count if ch == 0 else sv_bin_count],
                                 [0, 180 if ch == 0 else 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((h_bins if ch == 0 else sv_bins, hist))
        cv2.polylines(h_hist_img if ch == 0 else sv_hist_img, [pts], False, col)

    sv_hist_img, h_hist_img = np.flipud(sv_hist_img), np.flipud(h_hist_img)
    h_hist_img[:, 0] = (0, 255, 0)

    cv2.imshow('sat / val hist | hue hist', np.concatenate([sv_hist_img, h_hist_img], axis=1))


# ----------------------------------------
# EXAMPLE USAGE 
# ----------------------------------------
if __name__ == '__main__':
    device = device_constants.Device(device_constants.WEBCAM)
    screen = blank_screen(device, scale=1)
    gaze_pts = [(200, 400)]
    draw_gaze_v2(screen, gaze_pts, (255, 255, 255), scale=1, return_img=True)
    # bgr_img = cv2.imread('eye_images/andreas2_l.png', 3)
    # hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    cv2.imshow("blank screen", screen)

    # draw_histogram(bgr_img)
    # draw_histogram_hsv(hsv_img)

    cv2.waitKey()
