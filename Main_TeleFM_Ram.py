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
import cvzone  # pip install cvzone
import xlsxwriter
import os
import glob


''' TeleCF'''
# sub_num = '089'
# task = 'D'
# video_names = [f'TEL{sub_num}_FM_BL_{task}T']
# video_list = [
#     f"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\TEL{sub_num}\BL\FM\TEL{sub_num}_FM_BL_{task}T.MOV"]
# RIGHT = False
# FLIP = True
# print(video_names)
# print(video_list)

# sub_num = '97'
# ST_DT = 'S'
# month = 'BL'
# video_list = [f"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\TEL0{sub_num}\\{month}\\FM\\TEL0{sub_num}_{month}_FM_{ST_DT}T.mov"]
# video_names = [video_list[0].split('\\')[-1].split('.')[0]]
# RIGHT = False
# FLIP = True

''' Tele-CF Fu '''
# sub_num = '043'
# month = '10M'
# type_task = 'S'
# format_v = 'mp4'
# video_list = [rf"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\TEL{sub_num}\{month}\TEL{sub_num}_{month}_FM_{type_task}T.{format_v}"]
# video_names = [f'TEL{sub_num}_{month}_FM_{type_task}T']
# RIGHT = False
# FLIP = True
# if format_v == 'mp4':
#     FLIP = False
#     RIGHT = not RIGHT
# print(video_names)
# print(video_list)

''' PAN 2024'''
# base_folder = r'Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\Feb5_Siobhan_UA_BOX\CogFrailty_Data'
#
# # # Use glob to find all .mp4 files in the folder
# # mp4_files = glob.glob(os.path.join(base_folder, '*.mp4'))
# #
# # # Print the list of .mp4 files
# # for file in mp4_files:
# #     portion = file.split('\\')[-1]
# #     file_path = base_folder + '\\' + portion
# #     print(file_path)
# video_names = ['HML0135_DualTask.mp4']
# video_list = [r'Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\Feb5_Siobhan_UA_BOX\CogFrailty_Data\HML0135_DualTask.mp4']
#
# RIGHT = False
# FLIP = True
# format_v = 'mp4'
# if format_v == 'mp4':
#     FLIP = False
#     RIGHT = not RIGHT
# print(video_names)
# print(video_list)


''' U-PAN '''
# ST_DT = ['DualTask', 'SingleTask']
# ST_DT = ST_DT[0]
# sub_num = 'z641KAAQ'
# video_list = [rf"Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\download_Feb_2023\UA_Site\{sub_num}\Cog_Frailty_Task\{sub_num}_{ST_DT}.mp4"]
# video_names = [f'{sub_num}_{ST_DT}']
# RIGHT = False
# FLIP = False
# print(video_names)
# print(video_list)

# UM Site
# ST_DT = ['DualTask', 'SingleTask']
# ST_DT = ST_DT[0]
# sub_num = 'z641KAAQ'
# video_list = [rf"Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\download_Feb_2023\UA_Site\{sub_num}\Cog_Frailty_Task\{sub_num}_{ST_DT}.mp4"]
# video_names = [f'{sub_num}_{ST_DT}']
# RIGHT = True
# FLIP = False
# print(video_names)
# print(video_list)

# # Other samples
# video_list = [r"Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\sample_data\Cog_Frailty_Task\vK6uuAAC_DualTask.mp4"]
# video_names = ['vK6uuAAC_DualTask']
# print(video_names)
# print(video_list)

''' Doha dataset '''
# video_list = [r"Z:\Projects BCM\H-42315 QNRF Electrical Stimulation (Doha)\Data\Raw Sensor\[01262023] Data backup_MG\All Data_01262023_MG_data_backup\cognitive frailty\Sub 107\Pre\IMG_7874.MOV"]
# video_names = ['IMG_7874']
# video_list = [r"Z:\Projects BCM\H-42315 QNRF Electrical Stimulation (Doha)\Data\Raw Sensor\[01262023] Data backup_MG\All Data_01262023_MG_data_backup\cognitive frailty\setting test.MOV"]
# video_names = ['setting test']
# print(video_names)
# print(video_list)

''' Doha sample data'''
# video_list = [r"Z:\Projects BCM\H-42315 QNRF Electrical Stimulation (Doha)\Data\Raw Sensor\[01262023] Data backup_MG\All Data_01262023_MG_data_backup\cognitive frailty\setting test.mov"]
# # # video_list = [r"Z:\Projects BCM\H-42315 QNRF Electrical Stimulation (Doha)\Data\Raw Sensor\Mohammad test videos\WhatsApp Video 2023-08-28 at 22.32.33.mp4"]
# # # 'setting test'
# # video_names = ['WhatsApp Video 2023-08-28 at 22.32.41 (1)']
# # # video_names = ['WhatsApp Video 2023-08-28 at 22.32.33']
# RIGHT = False
# FLIP = True
# format_v = 'mp4'
# if format_v == 'mp4':
#     FLIP = False
#     RIGHT = not RIGHT
# print(video_names)
# print(video_list)

''' Boot study '''
# # video_list = [r"Z:\Projects BCM\Boot Study\Data\Frailty Videos\02-018 baseline.MOV"]
# video_list = [r"/home/mohammad/Desktop/Boot_Study/Data/HML0138_SingleTask.mp4"]
# video_names = [i.split('/')[-1].split('.')[0] for i in video_list]
# print(video_names)
# print(video_list)
# RIGHT = False
# FLIP = True

# file_namess = [
#     # '01010 Baseline',
#     # '01010 Frailty End',
#     # '01014 EG Baseline',
#     # '01015 VC Baseline',
#     # '01016 SL Baseline',
#     # '01043_baseline',
#     # '02002 Frailty 01-11-22 Baseline',
#     # '02002 Frailty 2_0 End',
#     # '02005 AS Baseline',
#     # '02007 Frailty 01-20-22 Baseline',
#     # '02008 Frailty 01-31-22 Baseline',
#     # '02008frailyvideo End',
#     # '02009 Baseline',
#     # '02009 End',
#     # '02010 MK Frailty start',
#     # '02014 FH Baseline',
#     # '02015 SC Baseline',
#     # '03-004_baseline',
#     # '03004 baseline',
#     # '01-018 baseline 2_0_',
#     # '01-020 end',
#     '01017 Baseline'
# ]
# video_list = [rf"Z:\Projects BCM\Boot Study\Data\Frailty Videos\Analyzed\{i}.3gp" for i in file_namess]
# video_names = [i.split('\\')[-1].split('.')[0] for i in video_list]
# print(video_names)
# print(video_list)
# RIGHT = False
# FLIP = True

''' Harvard study '''
# # # video_list = [r"Z:\Projects BCM\Boot Study\Data\Frailty Videos\02-018 baseline.MOV"]
# video_list = [r"Z:\Projects BCM\Project Harvard Mohammad TeleCF\Data\VF-050-ST.MOV"]
# video_names = [i.split('\\')[-1].split('.')[0] for i in video_list]
# print(video_names)
# print(video_list)
# RIGHT = True
# FLIP = False

''' Sample video '''
# video_list = [r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\ICAMP FRAILTY\VIDEOS\T001_DT.MOV"]
# video_names = [i.split('\\')[-1].split('.')[0] for i in video_list]
# print(video_names)
# print(video_list)
# RIGHT = False
# FLIP = True

''' Best buy '''
# sub_num = '004'
# month = '6M'
# type_task = 'D'
# video_list = [rf"Z:\Projects BCM\H-51553 BestBuy Research\Phase I\Data\Raw Sensor\BBH{sub_num}\{month}\FM\BBH{sub_num}_{month}_FM_{type_task}T.MOV"]
# video_names = [f'BBH{sub_num}_{month}_FM_{type_task}T']
# RIGHT = False
# FLIP = True

''' UCLA PAN '''
# sub_num = '401'
#
# type_task = 'D'
# format_v = 'mp4'
# if type_task is 'S':
#     type_task = 'SingleTask'
# else:
#     type_task = 'DualTask'
# video_list = [rf"/home/mohammad/Desktop/Tele_CF/Data/UA BOX HEALTH PAN/Frailty Video Files/HML0{sub_num}/HML0{sub_num}_{type_task}.{format_v}"]
# video_names = [f'HML0{sub_num}_{type_task}']
# RIGHT = True
# FLIP = True
# if format_v == 'mp4':
#     FLIP = False
#     RIGHT = not RIGHT

''' UCLA PAN March subjects '''
# sub_num = '821'
# type_task = 'S'
# format_v = 'mp4'
#
# if type_task is 'S':
#     type_task = 'SingleTask'
# else:
#     type_task = 'DualTask'
#
# video_list = [rf"/home/mohammad/Desktop/Tele_CF/Data/UA BOX HEALTH PAN/videos_March21_2025/HML0{sub_num}/HML0{sub_num}_{type_task}.{format_v}"]
# video_names = [f'HML0{sub_num}_{type_task}']
# RIGHT = False
# FLIP = False
# if format_v == 'mp4':
#     FLIP = False
#     RIGHT = not RIGHT

''' UCLA PAN TEMP '''
# directory_path = '/home/mohammad/Desktop/Tele_CF/Data/UA BOX HEALTH PAN/Frailty Video Files/'
#
# patient_ids = []
#
# # Loop through the directory and check if each item is a folder
# for folder_name in os.listdir(directory_path):
#     folder_path = os.path.join(directory_path, folder_name)
#     if os.path.isdir(folder_path):
#         patient_ids.append(folder_name)
#
# # Print the list of patient IDs
# patient_ids.sort()
# patient_ids = [i[-3:] for i in patient_ids]
# print(patient_ids)
# # exit()
# task_types_all = ['SingleTask', 'DualTask']
# for sub_num in patient_ids:
#     for tt in task_types_all:
#         type_task = tt
#         format_v = 'mp4'
#         video_list = [rf"/home/mohammad/Desktop/Tele_CF/Data/UA BOX HEALTH PAN/Frailty Video Files/HML0{sub_num}/HML0{sub_num}_{type_task}.{format_v}"]
#         video_names = [f'HML0{sub_num}_{type_task}']
#         RIGHT = False
#         FLIP = True
#         if format_v == 'mp4':
#             FLIP = False
#             RIGHT = not RIGHT

# exit()

''' Sample '''
# video_list = [r"/home/mohammad/Desktop/Frailty_videos/HML0113_DualTask.mp4"]
# video_names = [i.split('/')[-1].split('.')[0] for i in video_list]
# print(video_names)
# print(video_list)
# RIGHT = True
# FLIP = False
# print(video_names)
# print(video_list)

''' Boot study March subjects '''
# import glob
# import os
#
# folder_path = r'/home/mohammad/Desktop/Boot_Study/videos-05-27-2025/'
#
# # Get all .mov and .mp4 files
# # video_list = glob.glob(os.path.join(folder_path, '*.mov')) + glob.glob(os.path.join(folder_path, '*.mp4')) + glob.glob(os.path.join(folder_path, '*.MOV'))
# video_list = [r'/home/mohammad/Desktop/Boot_Study/videos-05-27-2025/05059_DT.MP4']
# video_names = [i.split('/')[-1].split('.')[0] for i in video_list]
# # Print the list of files
# for file in video_list:
#     # print(file)
#     pass
# print(video_list)
# print(video_names)
# RIGHT = False
# FLIP = True

''' Vojtech subjects April 9 2025'''
# import glob
# import os
#
# folder_path = r'/home/mohammad/Desktop/Boot_Study/videos-05-12-2025/'
#
# # Get all .mov and .mp4 files
# # video_list = glob.glob(os.path.join(folder_path, '*.mov')) + glob.glob(os.path.join(folder_path, '*.mp4')) + glob.glob(os.path.join(folder_path, '*.MOV'))
# video_list = [rf'{folder_path}/05045_ST.mp4']
# video_names = [i.split('/')[-1].split('.')[0] for i in video_list]
# # Print the list of files
# for file in video_list:
#     # print(file)
#     pass
# print(video_list)
# print(video_names)
# RIGHT = True
# FLIP = True

''' PAN April 24 2025'''
# hml_id = rf'0734'
# folder_path = rf'/home/mohammad/Desktop/pan_videos/videos 5-21-2025/HML{hml_id}'
#
# # Get all .mov and .mp4 files
# # video_list = glob.glob(os.path.join(folder_path, '*.mov')) + glob.glob(os.path.join(folder_path, '*.mp4')) + glob.glob(os.path.join(folder_path, '*.MOV'))
# video_list = [rf'{folder_path}/HML{hml_id}_DualTask.mp4']
# video_names = [i.split('/')[-1].split('.')[0] for i in video_list]
# # Print the list of files
# for file in video_list:
#     # print(file)
#     pass
# print(video_list)
# print(video_names)
# RIGHT = True
# FLIP = False

''' Tele Exer-game data'''
# import os
#
# video_list = []
# for root, dirs, files in os.walk(r"/home/mohammad/Desktop/Tele-Exer/Data/Telexergame_05-19-2025/"):
#     for file in files:
#         if file.lower().endswith(('.mov', '.mp4')):
#             video_list.append(os.path.join(root, file))
#
# # Get all .mov and .mp4 files
# # video_list = glob.glob(os.path.join(folder_path, '*.mov')) + glob.glob(os.path.join(folder_path, '*.mp4')) + glob.glob(os.path.join(folder_path, '*.MOV'))
# video_list.sort()
# vid_id = 21
video_list = [r"C:\Users\mrouzi\Desktop\C2SHIP_AI\Data\HML0493_DualTask.mp4"]
video_names = [i.split('\\')[-1].split('.')[0] for i in video_list]
print('Here')
print(video_names)
# Print the list of files
for file in video_list:
    # print(file)
    pass
print(video_list)
print(video_names)
RIGHT = True
FLIP = False

for k in range(len(video_names)):
    # break
    # cap = cv2.VideoCapture('Frailty_1.mp4')
    print(video_names[k])
    cap = cv2.VideoCapture(video_list[k])

    detector = FM.frailtyDetector()
    dir = 0
    segmentor = SelfiSegmentation()
    Angle_Pos = []
    outWB = xlsxwriter.Workbook()
    outSheet = outWB.add_worksheet()
    outSheet.write("A1", "Sample")
    outSheet.write("B1", "Angle Position")
    cc1 = 0

    '''Save the video '''
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(rf'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\gif_files\{video_names[k]}.avi', fourcc,
                          30.0, (500, 700))

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

            ''' Show option '''
            width = 500
            height = 700
            dim = (width, height)
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            out.write(resized_img)
            cv2.imshow("Image", resized_img)
            cv2.waitKey(1)

            # print("Output\\{}.xlsx".format(video_names[i]))
            workbook = xlsxwriter.Workbook(
                # rf'Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\FU data phase II zoom\Angles\{video_names[k]}.xlsx'
                # "Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Results\Results_Mohammad\\{}.xlsx".format(video_names[k])
                # rf'Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Analysis\python_outputs\UA_Site\{video_names[k]}.xlsx'
                # rf'Z:\Projects BCM\H-42315 QNRF Electrical Stimulation (Doha)\Analysis\Mohammad_Analysis\Angles\{video_names[k]}.xlsx'
                # rf'Z:\Projects BCM\Boot Study\Results\Angles\{video_names[k]}.xlsx',
                # rf'Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Data\sample_data\Cog_Frailty_Task\{video_names[k]}.xlsx'
                # rf'Z:\Projects BCM\Project Harvard Mohammad TeleCF\Results\angles\{video_names[k]}.xlsx',
                # rf'Z:\Projects BCM\H-51553 BestBuy Research\Phase I\Results\Mohammad_frailty_results\angles\{video_names[k]}.xlsx'
                # rf'Z:\Projects BCM\H-61802 U19-PAN (Precision Aging Network)\Results\pre_process_angles\{video_names[k]}.xlsx'

                ################### UCLA Era ###############################
                # rf'/home/mohammad/Desktop/Tele_CF/Results/Angle/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/Tele_CF/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/Tele_CF/boot_study/Angle/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/Boot_Study/results-05-27-2025/Angel/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/Frailty_videos/sample_video/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/vojtech_data/Results/Angel/{video_names[k]}.xlsx'
                # rf'/home/mohammad/Desktop/pan_videos/Angles/{video_names[k]}.xlsx'
                rf'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\Angle\{video_names[k]}.xlsx'
            )
            worksheet = workbook.add_worksheet()
            # declare data
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
