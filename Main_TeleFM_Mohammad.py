# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:26:43 2022

@author: 233264
"""
import sys
import cv2
import numpy as np
import FrailtyModule as FM
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import xlsxwriter
import os


def frailty_preprocess(video_path, gif_output, angle_output, RIGHT=True, FLIP=False, show_video=False):
    ############### Video name ################
    video_names = [os.path.splitext(os.path.basename(i))[0] for i in [video_path]]
    video_name = video_names[0]
    ############### Read video ################
    cap = cv2.VideoCapture(video_path)
    detector = FM.frailtyDetector()
    dir = 0
    segmentor = SelfiSegmentation()
    Angle_Pos = []
    outWB = xlsxwriter.Workbook()
    outSheet = outWB.add_worksheet()
    outSheet.write("A1", "Sample")
    outSheet.write("B1", "Angle Position")
    cc1 = 0

    '''Save the GIF file '''
    # Define the codec and create a VideoWriter object
    cap1 = cv2.VideoCapture(video_path)
    _, img_s = cap1.read()
    original_height = img_s.shape[0]
    original_width = img_s.shape[1]
    scale_factor = max(500 / original_height, 1)  # scale up if needed
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    dim_s = (new_width, new_height)
    cap1.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(gif_output, f'{video_name}.avi')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, dim_s)
    # print(dim)

    while True:
        success, img = cap.read()
        # print(success)
        if FLIP:
            img = cv2.flip(img, 0)
        if not RIGHT:
            img = cv2.flip(img, 1)
        # print(success)
        # img = cv2.resize(img, (1280, 720))
        # img = cv2.imread("AiTrainer/test.jpg")
        if success:
            # print('HERE')
            img1 = detector.findPose(img, False)

            lmList = detector.findPosition(img1, False)
            # lmList = detector.findPosition(img, True)

            # img = segmentor.removeBG(img1, (255, 255, 255), threshold=0.6)
            # img = img1

            # print(lmList)
            if len(lmList) != 0:
                # Right Arm
                angle = detector.findAngle(img, 12, 14, 16, True)
                # # Left Arm
                # angle = detector.findAngle(img, 11, 13, 15)
                Angle_Pos.append(angle)

            ''' Show option (proportional resize with min height 500) '''

            original_height = img.shape[0]
            original_width = img.shape[1]
            scale_factor = max(500 / original_height, 1)  # scale up if needed
            new_height = int(original_height * scale_factor)
            new_width = int(original_width * scale_factor)
            dim = (new_width, new_height)
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            if show_video:
                cv2.imshow("Image", resized_img)
                cv2.waitKey(1)
            out.write(resized_img)

            # print("Output\\{}.xlsx".format(video_names[i]))
            out_path1 = os.path.join(angle_output, f'{video_name}.xlsx')
            workbook = xlsxwriter.Workbook(out_path1)
            worksheet = workbook.add_worksheet()
            row = 0
            column = 0
            # write data to file
            for i in Angle_Pos:
                worksheet.write(row, column, i)
                row += 1
            workbook.close()
        else:
            break
    cap.release()
    out.release()

if __name__ == "__main__":
    frailty_preprocess(r"C:\Users\mrouzi\Desktop\C2SHIP_AI\Data\HML0521_DualTask.mp4",
                       gif_output=r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\gif_files',
                       angle_output=r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\Angle',
                       show_video=False,
                       RIGHT=True,
                       FLIP=False)
