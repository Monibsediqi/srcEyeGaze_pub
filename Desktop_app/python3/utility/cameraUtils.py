import cv2
from utility import imageUtils
from PIL import Image
import numpy as np
import tflite_tester

TAG = "cameraUtils"

def eye_calibration(threshEyeImage, wRef, hRef):

    # LOOP OVER FRAMES, CALL THIS METHOD AND GET THE coeffs x and y
    xReferencePoint = wRef/2       # center point
    yReferencePoint = hRef/2

    contours, _ = cv2.findContours(threshEyeImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours) != 0:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        xIrisCenter = x + w / 2
        yIrisCenter = y + h / 2

        xCoeff = xIrisCenter - xReferencePoint
        yCoeffs = yIrisCenter - yReferencePoint

        return xCoeff, yCoeffs
    else:
        return None, None

def start_calibration(iris_points, ref_points):

    x_iris_center, y_iris_center = iris_points[0], iris_points[1]
    x_ref_point, y_ref_point = (ref_points[0]), (ref_points[1])
    print("x_ref_point:{}, y_ref_point:{}".format(x_ref_point, y_ref_point))
    xCoeff = x_iris_center - x_ref_point
    yCoeff = y_iris_center - y_ref_point
    print("xCoeff:{}, yCoeff:{}".format(xCoeff, yCoeff))
    return xCoeff, yCoeff


def calib_process():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    xCoeffs = []
    yCoeffs = []
    xRef = []
    yRef = []

    calibrationFlag = False

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)

        frame = imageUtils.showCenter(frame)
        if cv2.waitKey(13) == ord('s'):
            calibrationFlag = True

        if calibrationFlag:
            cv2.putText(img=frame, text="Calibrating ...", org=(0, 50), fontScale=1, color=(0, 255, 0),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX)
            cv2.putText(img=frame, text="Press C to exit", org=(0, 90), fontScale=1, color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX)

            landmarks, results = imageUtils.getLandmarks(frame)

            # RIGHT EYE
            rightEyeImage = imageUtils.getRightEye(frame, landmarks)
            pil_img = Image.fromarray(rightEyeImage)
            resized_eye = pil_img.resize((64, 64))
            cv_resized_img = np.array(resized_eye)

            iris_center, eye_contour = tflite_tester.get_iris_landmarks(eye_image=cv_resized_img)
            x1, y1 = (eye_contour[0][0]) + 2, (eye_contour[0][1])
            x2, y2 = eye_contour[1][0] - 2, eye_contour[1][1]
            x3, y3 = eye_contour[2][0], eye_contour[2][1] + 2
            x4, y4 = eye_contour[3][0], eye_contour[3][1] - 2

            width_ref = (x1 - x2)/2 + x2
            height_ref = (y3 - y4)/2 + y4

            width_text = "width_ref" + str(width_ref)
            height_text = "height_ref" + str(height_ref)

            cv2.putText(img=frame, text=width_text, org=(0, 120), fontScale=1, color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX)
            cv2.putText(img=frame, text=height_text, org=(0, 150), fontScale=1, color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX)

            cv2.circle(cv_resized_img, (int(x1), int(y1)), radius=0, color=(0, 255, 0), thickness=2) # Right
            cv2.circle(cv_resized_img, (int(x2), int(y2)), radius=0, color=(0, 255, 0), thickness=2) # Left
            cv2.circle(cv_resized_img, (int(x3), int(y3)), radius=0, color=(0, 255, 0), thickness=2) # Down
            cv2.circle(cv_resized_img, (int(x4), int(y4)), radius=0, color=(0, 255, 0), thickness=2) # Top

            # VISUALIZE EYE CENTER
            x, y = int(iris_center[0]), int(iris_center[1])
            cv2.circle(cv_resized_img, (x, y), radius=0, color=(0, 0, 255), thickness=2)

            print("iris x:{}, y:{}".format(iris_center[0], iris_center[1]))

            # TRY, USING THE IMAGE CENTER
            xCoeff, yCoeff = start_calibration(iris_center, [width_ref, height_ref])

            if xCoeff is not None:
                xCoeffs.append(xCoeff)
                yCoeffs.append(yCoeff)

                xRef.append(width_ref)
                yRef.append(height_ref)

            cv2.imshow("rightEyeImage", rightEyeImage)
            cv2.imshow("calib", cv_resized_img)

        cv2.imshow('win', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break
    cap.release()
    return xCoeffs, yCoeffs, xRef, yRef


def get_focal_length(iris_diameter_in_pixel):
    iris_width = 11.7  # (w) in mm (fixed across wide range population)
    distance_from_camera = 400  # (D) 400 mm (40cm) (ideal distance from eye to camera) -
    iris_diameter_in_pixel = iris_diameter_in_pixel  # (P)
    """
    Taken from here: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    The triangle similarity goes something like this: Let’s
    say we have a marker or object with a known width W.
    We then place this marker some distance D from our camera.
    We take a picture of our object using our camera and then measure the apparent width in pixels P.
    This allows us to derive the perceived focal length F of our camera:

    F = (P x D) / W 
    
    For example, let’s say I place a standard piece of 8.5 x 11in piece of paper 
    (horizontally; W = 11) D = 24 inches in front of my camera and take a photo. 
    When I measure the width of the piece of paper in the image, I notice that the perceived width of the paper is 
    P = 248 pixels. My focal length F is then:

    F = (248px x 24in) / 11in = 543.45"""

    focal_length = (iris_diameter_in_pixel * distance_from_camera) / iris_width  # ? in pixel
    return focal_length  # its' 649.0


def distance_to_camera(focal_length, iris_diameter_in_pixel):
    iris_width_in_mm = 11.7  # W
    iris_diameter_in_pixel = iris_diameter_in_pixel  # P
    focal_length = focal_length  # F

    """
     D’ = (W x F) / P
     As I continue to move my camera both closer and farther away from the object/marker, I can apply the triangle similarity
     to determine the distance of the object to the camera:  D’ = (W x F) / P Again, to make this more concrete, let’s say I
     move my camera 3 ft (or 36 inches) away from my marker and take a photo of the same piece of paper. Through automatic image processing I am able to determine that the perceived width of the piece of paper is now 170 pixels. Plugging this into the equation we now get:

     D’ = (11in x 543.45) / 170 = 35in"""

    distance_in_mm = (iris_width_in_mm * focal_length) / iris_diameter_in_pixel  # returns distance to camera in mm
    return distance_in_mm


def pixel_to_mm(image, dpi_x, dpi_y):
    img_w, img_h, _ = image.shape
    x_mm = (img_w * 25.4) / dpi_x
    y_mm = (img_h * 25.4) / dpi_y
    return x_mm, y_mm

