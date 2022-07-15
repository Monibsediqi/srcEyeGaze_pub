"""
Project: Gaze Estimation 
Written by: Monib Sediqi
Email: kh.monib@gmail.com
Date: 25 Apr 2021
"""

import cv2
import numpy as np
import image_utils
import eye_extractor

_win_name = "Pre Processing"

# FIXME: Fix the commented out statement -> small_contour

class PreProcessor:

    def __init__(self):
        self.full_debug_img = None

    def erase_specular(self, eye_img, debug=False):

        # Rather arbitrary decision on how large a specularity may be
        max_specular_contour_area = sum(eye_img.shape[:2]) / 2

        # Extract top 50% of intensities
        eye_img_grey = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img_grey_blur = cv2.GaussianBlur(eye_img_grey, (5, 5), 0)

        # Close to suppress eyelashes
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eye_img_grey_blur = cv2.morphologyEx(eye_img_grey_blur, cv2.MORPH_CLOSE, morph_kernel)

        if eye_img_grey_blur is None:
            raise eye_extractor.NoEyesFound()

        thresh_val = int(np.percentile(eye_img_grey_blur, 50))

        _, thresh_img = cv2.threshold(eye_img_grey_blur, thresh_val, 255, cv2.THRESH_BINARY)

        # Find all contours and throw away the big ones
        contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # small_contours = filter(lambda x: cv2.contourArea(x) < max_specular_contour_area, contours)

        small_contours_mask = np.zeros_like(eye_img_grey)

        cv2.drawContours(small_contours_mask, contours, -1, (0, 0, 255), -1)

        # Dilate the smallest contours found
        small_contours_mask_dilated = cv2.dilate(small_contours_mask, morph_kernel)

        removed_specular_img = cv2.inpaint(eye_img, small_contours_mask_dilated, 2, flags=cv2.INPAINT_TELEA)

        if debug:
            thresh_hierarchy = cv2.cvtColor(eye_img_grey, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(thresh_hierarchy, contours, -1, (255, 0, 0), -1)
            thresh_hierarchy = cv2.add(thresh_hierarchy, cv2.cvtColor(small_contours_mask_dilated, cv2.COLOR_GRAY2BGR))
            cv2.drawContours(thresh_hierarchy, contours, -1, (255, 0, 0), -1)
            stacked_imgs = np.concatenate([eye_img, thresh_hierarchy, removed_specular_img], axis=1)

            if debug == 1:
                self.full_debug_img = stacked_imgs
            elif debug == 2:
                self.full_debug_img = image_utils.stack_imgs_vertical([self.full_debug_img, stacked_imgs])
                cv2.imshow(_win_name, self.full_debug_img)
            elif debug == 3:
                cv2.imshow(_win_name, stacked_imgs)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()

        return removed_specular_img


########################################
#           TEST USAGE                 #
########################################

if __name__ == "__main__":
    eye_img_path = "eye_images/andreas1_l.png"
    eye_img = cv2.imread(eye_img_path)

    preProc = PreProcessor()
    eye_proc_img = preProc.erase_specular(eye_img, 3)