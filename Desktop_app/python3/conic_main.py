
import socket
import gaze_system as gaze_system

from datetime import datetime
use_webcam = True
import copy
import random
import draw_utils
import cv2
import mediapipe as mp

import anatomical_constants
import device_constants
import image_utils
from utility import imageUtils
import find_eye_center_combined


debug = True
recording = False
import eye_extractor
import device_constants


random.seed(123)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

__focal_len_x_px = device_constants.lg_camera_mtrx[0][0]
__focal_len_y_px = device_constants.lg_camera_mtrx[1][1]
__cx_px = device_constants.lg_camera_mtrx[0][2]
__cy_px = device_constants.lg_camera_mtrx[1][2]

__focal_len_z_px = (__focal_len_x_px + __focal_len_y_px) / 2

__winname = "main_gaze_tf"

r = random.randint(0, 255)
g = random.randint(0, 255)
b = random.randint(0, 255)

_color = (r, g, b)
_font = cv2.FONT_HERSHEY_SIMPLEX
_fontScale = 1
_thickness = 2

device = device_constants.Device(device_constants.WEBCAM)
g_sys = gaze_system.GazeSystem(device, debug, recording)

def mp_full_facemesh(debug=False):
    __eye_size = (36, 60)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.1) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.undistort(image, device_constants.lg_camera_mtrx, device_constants.lg_dist_coeffs)
            image_original = copy.deepcopy(image)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the eye and iris annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # right and left indices of individual iris
                    left_eye_points = [469, 471]
                    right_eye_points = [474, 476]

                    iris_left_x0_px = face_landmarks.landmark[468].x
                    iris_left_y0_px = face_landmarks.landmark[468].y

                    iris_right_x0_px = face_landmarks.landmark[473].x
                    iris_right_y0_px = face_landmarks.landmark[473].y

                    left_eye_diameter_px = image_utils.get_distance(image, face_landmarks,
                                                                    left_eye_points)
                    right_eye_diameter_px = image_utils.get_distance(image, face_landmarks,
                                                                     right_eye_points)
                    # distance from camera to eyes
                    # Using iris_r_px / focal_len_px = iris_r_mm / distance_to_iris_mm
                    iris_left_z_mm = (anatomical_constants.limbus_r_mm * 2 * __focal_len_z_px) / left_eye_diameter_px
                    iris_right_z_mm = (anatomical_constants.limbus_r_mm * 2 * __focal_len_z_px) / right_eye_diameter_px

                    # world coordinates
                    iris_left_x_mm = - iris_left_z_mm * (iris_left_x0_px - __cx_px) / __focal_len_x_px
                    iris_left_y_mm = - iris_left_z_mm * (iris_left_y0_px - __cy_px) / __focal_len_y_px
                    left_iris_world = (iris_left_x_mm, iris_left_y_mm, iris_left_z_mm)

                    iris_right_x_mm = - iris_right_z_mm * (iris_right_x0_px - __cx_px) / __focal_len_x_px
                    iris_right_y_mm = iris_right_z_mm * (iris_right_y0_px - __cy_px) / __focal_len_y_px
                    right_iris_world = (iris_right_x_mm, iris_right_y_mm, iris_right_z_mm)

                    left_eye_crop = imageUtils.getLeftEye(image_original,
                                                          landmarks=face_landmarks.landmark)
                    right_eye_crop = imageUtils.getRightEye(image_original,
                                                            landmarks=face_landmarks.landmark)
                    # resize to 60 x 36
                    left_eye_crop = cv2.resize(left_eye_crop, dsize=(__eye_size[1], __eye_size[0]))
                    right_eye_crop = cv2.resize(right_eye_crop, dsize=(__eye_size[1], __eye_size[0]))


                    right_eye = imageUtils.getRightEyeRect(image_original, face_landmarks.landmark)
                    left_eye = imageUtils.getLeftEyeRect(image_original, face_landmarks.landmark)
                    eye_rect_pairs = [right_eye, left_eye]

                    # pupil_x0, pupil_y0 = eye_center_locator_combined.find_pupil(right_eye, debug_index=3)
                    # right_eye_roi = eye_extractor.EyeRoi(pupil_x0,pupil_y0, right_eye)



                    gaze_pt = g_sys.get_gaze_from_frame(image_original, eye_rect_pairs)
                    (x, y) = gaze_pt
                    scr_img = draw_utils.blank_screen(device, scale=1)
                    scr_img = draw_utils.draw_gaze_v2(scr_img, [gaze_pt], (255, 0, 0), scale=1,
                                                      return_img=True)
                    cv2.putText(scr_img, f"Gaze X: {round(gaze_pt[0], 1)}, Y: {round(gaze_pt[1])}",
                                (10, 25),
                                _font, _fontScale, (0, 0, 255),
                                _thickness)
                    cv2.imshow(__winname + "11", scr_img)
                    print(f"x: {x}, y: {y}")

                    # x_scr_mm_left, y_scr_mm_left = gaze_geometry.get_gaze_point_mm_tf(left_iris_world,
                    #                                                                   normal_vector_left)
                    # x_scr_mm_right, y_scr_mm_right = gaze_geometry.get_gaze_point_mm_tf(right_iris_world,
                    #                                                                     normal_vector_right)

                    # create a black screen to visualize gaze
                    # scr_img = draw_utils.blank_screen(device, scale=1)
                    # draw_utils.draw_center_point(scr_img, thickness=2)
                    if debug:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_iris_connections_style())

                        __debug_imgs = image_utils.stack_imgs([left_eye_crop, right_eye_crop], vertical=False)
                        cv2.imshow(__winname, __debug_imgs)
                        cv2.imshow(__winname + "face_landmark", image)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break
    cap.release()


if __name__ == "__main__":
    mp_full_facemesh(debug=True)

