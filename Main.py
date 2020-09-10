import os
import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mask = cv2.imread("default-mask.png")
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

video = cv2.VideoCapture(2)

while True:
    _, image = video.read()

    landmark_img = image.copy()
    gray = cv2.cvtColor(landmark_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()


        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(landmark_img, (x,y), 2, (255,255,255), -1)
        mask_pos = image.copy()
        cv2.imshow("camera", landmark_img)

        # Proper mask coordinates
        x1_p = landmarks.part(1).x
        y1_p = landmarks.part(1).y
        x2_p = landmarks.part(16).x
        y2_p = landmarks.part(9).y

        # improper mask coordinates
        x1_im = landmarks.part(3).x
        y1_im = landmarks.part(3).y
        x2_im = landmarks.part(16).x
        y2_im = landmarks.part(9).y


        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # improper mask
        resized_mask_gray_im = cv2.resize(mask_gray, ((x2_im - x1_im), (y2_im - y1_im)))
        _, mask_mask = cv2.threshold(resized_mask_gray_im , 25, 255, cv2.THRESH_BINARY_INV)

        improper_masked_crop_img = image.copy()
        improper_masked_crop_img = cv2.bitwise_and(improper_masked_crop_img[y1_im:y2_im, x1_im:x2_im],
                                                   improper_masked_crop_img[y1_im:y2_im, x1_im:x2_im], mask=mask_mask)
        improper_masked_crop = cv2.add(improper_masked_crop_img, cv2.resize(mask, ((x2_im - x1_im), (y2_im - y1_im))))
        final_improper_mask_img = image.copy()
        final_improper_mask_img[y1_im:y2_im, x1_im:x2_im] = improper_masked_crop
        # cv2.imshow("Camera", final_improper_mask_img)

    # proper mask

        resized_mask_gray_p = cv2.resize(mask_gray, ((x2_p - x1_p), (y2_p - y1_p)))
        _, mask_mask = cv2.threshold(resized_mask_gray_p, 25, 255, cv2.THRESH_BINARY_INV)
        proper_masked_crop_img = image.copy()
        proper_masked_crop_img = cv2.bitwise_and(proper_masked_crop_img[y1_p:y2_p, x1_p:x2_p],
                                                   proper_masked_crop_img[y1_p:y2_p, x1_p:x2_p], mask=mask_mask)
        proper_masked_crop = cv2.add(proper_masked_crop_img, cv2.resize(mask, ((x2_p - x1_p), (y2_p - y1_p))))
        final_proper_mask_img = image.copy()
        final_proper_mask_img[y1_p:y2_p, x1_p:x2_p] = proper_masked_crop
        # cv2.imshow("Camera", final_proper_mask_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.release()
cv2.destroyAllWindows()