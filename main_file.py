import os
from Main_TeleFM_Mohammad import frailty_preprocess
from mohammad_signal_analysis import frailty_postprocess

''' Parameters Given By participant '''
# Video is going to be uploaded by participant:
video_path = r"C:\Users\mrouzi\Desktop\C2SHIP_AI\Data\HML0560 DualTask.mp4"

# There should be a radiobutton for task type which has Single Task option (passing ST to backend) and
# Dual task option (passing DT to backend).
task_type = 'DT'

# There should be another two radiobutton for right and left side. Each has two option of False and True.
RIGHT = True
FLIP = False

''' Default folders in the server '''
gif_output = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\gif_files'
angle_output = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\Angle'
video_names = [os.path.splitext(os.path.basename(i))[0] for i in [video_path]]
sub_id = video_names[0]
excel_file = rf"{angle_output}\{sub_id}.xlsx"
frailty_index_file = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\frailty_index'
video_file_location = os.path.dirname(video_path)
angular_acc = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\angular_acc'


frailty_preprocess(video_path=video_path,
                   gif_output=gif_output,
                   angle_output=angle_output,
                   show_video=False,
                   RIGHT=RIGHT,
                   FLIP=FLIP)


frailty_postprocess(excel_file=excel_file,
                    frailty_index_file=frailty_index_file,
                    task_type=task_type,
                    video_file_location=video_file_location,
                    angular_acc=angular_acc)
