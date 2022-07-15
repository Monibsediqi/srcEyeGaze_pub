import cv2, os, numpy as np
import image_utils
import time_profiler
import device_constants

# THIS CAN BE ACHIEVED BY MEDIAPIPE -> EXTRACTING THE EYE REGION
#                        eye_1     eye_2
#                  skin  |    nose |    skin
#                  |     |    |    |    |
eye_part_ratios = [0.05, 0.3, 0.3, 0.3, 0.05]

# Will try to find eye-pairs after rotating by these angles
angles_to_try = [0, 15, -15, 30, -30]

winname = 'Eye Extractor'



class NoEyesFound(Exception):
    def __init__(self, msg):
        self.msg = msg


class EyeRoi:

    def __init__(self, roi_x0_param, roi_y0_param, img_param): # img_param is eye_img

        self.roi_x0, self.roi_y0 = int(roi_x0_param), int(roi_y0_param)
        self.roi_h, self.roi_w = img_param.shape[:2]
        self.img = img_param

        if self.img is None: raise NoEyesFound()  # Prevent future problems with NoneType image

    def refine_pupil(self, pupil_x0, pupil_y0, full_img):

        # Copy pre-processed ROI back into full-frame (specularities removed)
        full_img[self.roi_y0:self.roi_y0 + self.roi_h,
        self.roi_x0:self.roi_x0 + self.roi_w] = self.img

        img_size = min(self.roi_w, self.roi_h)
        self.roi_h, self.roi_w = img_size, img_size
        self.roi_x0 = (self.roi_x0 + pupil_x0) - self.roi_w / 2
        self.roi_y0 = (self.roi_y0 + pupil_y0) - self.roi_h / 2

        # Ensure ROI will still lie within image boundaries (x-axis only)
        self.roi_x0 = max(self.roi_x0, 0)
        self.roi_x0 = min(self.roi_x0, full_img.shape[1] - self.roi_w)

        self.img = full_img[int(self.roi_y0):int(self.roi_y0 + self.roi_h),
                   int(self.roi_x0):int(self.roi_x0 + self.roi_w)]

        if self.img is None: raise NoEyesFound()  # Prevent future problems with NoneType image


def choose_best_eye_pair(eye_pair_rects, img):
    """ Determine between eyes / eyebrows / nostrils by comparing frequencies in image
    """

    return max(eye_pair_rects,
               key=lambda x0, y0, w, h
               : image_utils.measure_blurriness_LoG(img[y0:y0 + h, x0:x0 + w]))


# Default behaviour if fail to get eyepair
def get_eye_rois_default(frame_pyr, down_scale=4, debug=False, device=None, eye_roi_pair=None):
    """ Returns a pair of EyeRois - one for each eye in an eye pair
    """

    pyr_img = frame_pyr[down_scale].copy()
    full_frame = frame_pyr[1]

    pyr_img_grey = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2GRAY)

    _, img_w = pyr_img_grey.shape[:2]
    roi_r, roi_l = pyr_img_grey.copy(), pyr_img_grey.copy()
    roi_r[:, 0:int(img_w / 2)] = 0
    roi_l[:, int(img_w / 2):img_w] = 0
    min_eye_size = (pyr_img.shape[0] / 6, pyr_img.shape[0] / 9)

    # eye_l_rects = classifier_l_eye.detectMultiScale(roi_l, scaleFactor=1.1, minSize=min_eye_size)
    # eye_r_rects = classifier_r_eye.detectMultiScale(roi_r, scaleFactor=1.1, minSize=min_eye_size)
    eye_l_rects = eye_roi_pair[1]
    eye_r_rects = eye_roi_pair[0]


    # if len(eye_l_rects) == 0 or len(eye_r_rects) == 0:
    #     raise NoEyesFound('Did not find eyes with default behaviour')
    #
    # if len(eye_l_rects) == 1:
    #     best_eye_l_rect = eye_l_rects[0]
    # else:
    #     best_eye_l_rect = choose_best_eye_pair(eye_l_rects, pyr_img)

    rect_l_x0, rect_l_y0, rect_l_w, rect_l_h = eye_l_rects
    roi_l_x0, roi_l_y0, roi_l_w, roi_l_h = [x * down_scale for x in [rect_l_x0, rect_l_y0, rect_l_w, rect_l_h]]

    # if len(eye_r_rects) == 1:
    #     best_eye_r_rect = eye_r_rects[0]
    # else:
    #     best_eye_r_rect = choose_best_eye_pair(eye_r_rects, pyr_img)
    rect_r_x0, rect_r_y0, rect_r_w, rect_r_h = eye_r_rects
    roi_r_x0, roi_r_y0, roi_r_w, roi_r_h = [x * down_scale for x in [rect_r_x0, rect_r_y0, rect_r_w, rect_r_h]]

    eye_1_img = full_frame[roi_l_y0:(roi_l_y0 + roi_l_h), roi_l_x0:roi_l_x0 + roi_l_w]
    eye_2_img = full_frame[roi_r_y0:(roi_r_y0 + roi_r_h), roi_r_x0:roi_r_x0 + roi_r_w]
    eye_roi_1, eye_roi_2 = EyeRoi(roi_l_x0, roi_l_y0, eye_1_img), EyeRoi(roi_r_x0, roi_r_y0, eye_2_img)

    # Draw box around each eye_roi
    if debug:
        debug_img = frame_pyr[down_scale]
        start_point = (int(eye_roi_1.roi_x0/down_scale), int(eye_roi_1.roi_y0/down_scale))
        end_point = (int((eye_roi_1.roi_x0 + eye_roi_1.roi_w)/down_scale), int((eye_roi_1.roi_y0 + eye_roi_1.roi_h)/down_scale))
        cv2.rectangle(debug_img, start_point, end_point, (255, 255, 255), 2)
        cv2.rectangle(debug_img,
                      (int(eye_roi_2.roi_x0 / down_scale), int(eye_roi_2.roi_y0 / down_scale)),
                      (int((eye_roi_2.roi_x0 + eye_roi_2.roi_w) / down_scale),
                       int((eye_roi_2.roi_y0 + eye_roi_2.roi_h) / down_scale)),
                      (255, 255, 255), 2)

    return eye_roi_1, eye_roi_2

def get_eye_rois(frame_pyr, down_scale=4, debug=False, device=None, eye_roi_pair=None):
    # for angle in angles_to_try:
    #     try:
    #         return get_eye_rois_at_angle(frame_pyr, angle, down_scale, debug, device, eye_roi_pair)
    #     except NoEyesFound:
    #         continue  # Try next angle

    try:
        return get_eye_rois_default(frame_pyr, down_scale, debug, device, eye_roi_pair)
    except NoEyesFound:
        pass

    # Draw red border around debug frame
    if debug:
        debug_img = frame_pyr[down_scale]
        cv2.rectangle(debug_img, (0, 0), (debug_img.shape[1] - 1, debug_img.shape[0] - 1), (0, 0, 255), thickness=4)
        cv2.imshow(winname, debug_img)
