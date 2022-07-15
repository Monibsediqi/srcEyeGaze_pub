import numpy as np

ANDROID_TAB_A = 0x1
ANDROID_TAB_A_90_DEG_INV = 0x2
WEBCAM = 0x3

# Camera coefficients
lg_dist_coeffs = np.array([[0.09001055, -0.14723633, 0.00729122, 0.00455993, -0.61217001]])

lg_camera_mtrx = np.array([[644.15768772, 0., 329.40896567],
                          [0., 642.67774533, 226.17633021],
                          [0., 0., 1., ]])

# Screen size constants
screen_w_mm_lg_laptop, screen_h_mm_lg_laptop = 300, 110
screen_w_px_lg_laptop, screen_h_px_lg_laptop = 1366, 768
screen_x_offset_px = 70   # For taskbar on the left side of the laptop
screen_y_offset_px = 130  # For menu bar on the top side of the laptop

class Device:

    def __init__(self, device_type):

        self.device_type = device_type

        if device_type == ANDROID_TAB_A or device_type == ANDROID_TAB_A_90_DEG_INV:
            self.fx, self.fy = lg_camera_mtrx[0][0], lg_camera_mtrx[1][1]
            self.cx, self.cy = lg_camera_mtrx[0][2], lg_camera_mtrx[1][2]
            self.screen_size_mm = screen_w_mm_lg_laptop, screen_h_mm_lg_laptop
            self.screen_size_px = screen_w_px_lg_laptop, screen_h_px_lg_laptop
            self.screen_x_offset_px = screen_x_offset_px
            self.screen_y_offset_px = screen_y_offset_px
            self.offset_mm = 47, 16
            self.rot90s = 1
            self.mirror = False

        if device_type == ANDROID_TAB_A_90_DEG_INV:
            self.offset_mm = 47, -(self.screen_size_mm[1] + 16)
            self.rot90s = -1

        if device_type == WEBCAM:
            self.fx, self.fy = lg_camera_mtrx[0][0], lg_camera_mtrx[1][1]
            self.cx, self.cy = lg_camera_mtrx[0][2], lg_camera_mtrx[1][2]
            self.screen_size_mm = screen_w_mm_lg_laptop, screen_h_mm_lg_laptop
            self.screen_size_px = screen_w_px_lg_laptop, screen_h_px_lg_laptop
            self.screen_x_offset_px = screen_x_offset_px
            self.screen_y_offset_px = screen_y_offset_px
            self.offset_mm = 0, 0
            self.rot90s = 0
            self.mirror = False

    def get_intrinsic_cam_params(self):
        return self.fx, self.fy, self.cx, self.cy

    def get_dist_coeffs(self):
        return lg_dist_coeffs
