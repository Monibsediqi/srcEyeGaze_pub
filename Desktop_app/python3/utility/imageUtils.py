"""
Project: Gaze Estimation 
Company: Mayfarm Soft
Written by: Monib Sediqi
Email: kh.monib@gmail.com
Date: 18 Jun 2021

"""

import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

TAG = "Image_utils: "


def preprocessEye(eyeImage):
    # preprocess eye
    grayEye = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY)
    grayEye = cv2.medianBlur(grayEye, 3)
    equalizedEye = cv2.equalizeHist(grayEye)
    blurEye = cv2.GaussianBlur(equalizedEye, (7, 7), 0)
    _, thresh = cv2.threshold(blurEye, 80, 255, cv2.THRESH_BINARY_INV)
    img_erode = cv2.erode(thresh, None, iterations=2)
    img_dilate = cv2.dilate(img_erode, None, iterations=3)

    return img_dilate

def findContours(thresholdEyeImage):
    """
    :param thresholdEyeImage:
    :return: xIrisCenter, yIrisCenter
    """
    contours, _ = cv2.findContours(thresholdEyeImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours) != 0:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        if w is not None:
            return x, y, w, h


def blobDetector(eyeImage):
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1700      # In pixel

    detector = cv2.SimpleBlobDetector_create(detector_params)

    gray_frame = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(gray_frame, 45, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)  # 1
    img = cv2.dilate(img, None, iterations=4)  # 2
    img = cv2.medianBlur(img, 5)  # 3
    keypoints = detector.detect(img)
    return keypoints


def showEyeBboxSize(originalImage, rightEye=None, leftEye=None):
    if rightEye is not None:
        eyeHeight, eyeWidth, channel = rightEye.shape
        bbox_w = eyeWidth
        bbox_h = eyeHeight
        width_of_rect = "RightEye bbox_W:" + str(bbox_w)
        height_of_rect = "RightEye bbox_H:" + str(bbox_h)
        cv2.putText(img=originalImage, text=str(width_of_rect), org=(300, 30), fontScale=1, color=(235, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img=originalImage, text=str(height_of_rect), org=(300, 60), fontScale=1, color=(235, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        if leftEye is not None:
            eyeHeight, eyeWidth, channel = leftEye.shape
            bbox_w = eyeWidth
            bbox_h = eyeHeight
            width_of_rect = "LeftEye bbox_W:" + str(bbox_w)
            height_of_rect = "LeftEye bbox_H:" + str(bbox_h)
            cv2.putText(img=originalImage, text=str(width_of_rect), org=(300, 90), fontScale=1, color=(0, 255, 0),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(img=originalImage, text=str(height_of_rect), org=(300, 120), fontScale=1, color=(0, 255, 0),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    else:
        cv2.putText(img=originalImage, text="No bbox for None input.", org=(300, 30), fontScale=1, color=(235, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)


# FIXME: Fix the hough Circle and test it on a still image
def houghCircle(eyeBinaryImage, xIrisCenter = None, yIrisCenter= None):
    # Loads an image
    src = eyeBinaryImage
    # Check if image is loaded fine

    rows = src.shape[0]
    circles = cv2.HoughCircles(eyeBinaryImage, cv2.HOUGH_GRADIENT, 1, rows / 8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)

    cv2.imshow("detected circles", src)


def showCenter(image):
    image_h, image_w, c = image.shape
    center = (int(image_w / 2), int(image_h / 2))
    image = cv2.circle(image, center, 1, (2, 2, 255), 5)
    return image

def centerTheImage(image, eyeImg):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imgH, imgW, = image.shape
    eyeH, eyeW = eyeImg.shape

    # compute xoff and yoff for placement of upper left corner of eye image
    yoff = round((imgH - eyeH) / 2)
    xoff = round((imgW - eyeW) / 2)

    # use numpy indexing to place the eye image in the center of actual image
    result = image.copy()
    result[yoff:yoff + eyeH, xoff:xoff + eyeW] = eyeImg

    return result


def getLandmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    landmarks = results.multi_face_landmarks[0].landmark
    return landmarks, results


def getRightEye(image, landmarks):
    # eye_right landmarks (253, 257, 359, 463)
    eye_top = int(landmarks[443].y * image.shape[0])
    eye_left = int(landmarks[465].x * image.shape[1])
    eye_bottom = int(landmarks[450].y * image.shape[0])
    eye_right = int(landmarks[446].x * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye


def getRightEyeRect(image, landmarks):

    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[464].x * image.shape[1])
    eye_bottom = int(landmarks[253].y * image.shape[0])
    eye_right = int(landmarks[446].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h


def getLeftEye(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133)
    eye_top = int(landmarks[223].y * image.shape[0])
    eye_left = int(landmarks[226].x * image.shape[1])
    eye_bottom = int(landmarks[230].y * image.shape[0])
    eye_right = int(landmarks[245].x * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye


def getLeftEyeRect(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
    eye_top = int(landmarks[27].y * image.shape[0])
    eye_left = int(landmarks[226].x * image.shape[1])
    eye_bottom = int(landmarks[23].y * image.shape[0])
    eye_right = int(landmarks[244].x * image.shape[1])

    cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h


# Draw the face mesh annotations on the image.
def drawFaceMesh(image, results, ret= False ):
    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # print('face landmarks', face_landmarks)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        # cv2.imshow('MediaPipe FaceMesh', image)
    if ret:
        return image


def resize(img, scale_percent):
    # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return dim


def locateIris(image, center, radius=0):
    cv2.circle(image, center, radius, (255, 0, 0), 3)


def showEyeDirection(image, eye_image, x_iris, y_iris):
    text = {
        "right": "RIGHT",
        "left": "LEFT",
        "up": "UP",
        "down": "DOWN",
        "center": "CENTER"}
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin0 = (0, 30)
    origin1 = (0, 60)
    origin2 = (0, 90)
    font_scale = 1
    color = (0, 255, 122)
    thickness = 2
    eye_h, eye_w, c = eye_image.shape
    height_threshold = int(eye_h / 2)
    width_threshold = int(eye_w / 2)

    center_range_x = eye_image[width_threshold - 2: width_threshold + 3]
    center_range_y = eye_image[height_threshold - 2: height_threshold + 3]

    # measure x_iris along the width to calculate left and right
    # measure y_iris along the height to calculate up and down
    if x_iris in center_range_x and y_iris in center_range_y:
        cv2.putText(image, text["center"], origin0, font, font_scale, color, thickness)
    else:
        if x_iris > width_threshold:
            cv2.putText(image, text["right"], origin1, font, font_scale, color, thickness)
        elif x_iris < width_threshold:
            cv2.putText(image, text["left"], origin1, font, font_scale, color, thickness)
        if y_iris > height_threshold:
            cv2.putText(image, text["down"], origin2, font, font_scale, color, thickness)
        elif y_iris < height_threshold:
            cv2.putText(image, text["up"], origin2, font, font_scale, color, thickness)
