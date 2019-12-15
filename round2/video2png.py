import os
import glob

# file_name = ['SDR_4K_1', 'SDR_4K_2', 'SDR_4K_3', 'SDR_4K_4']
# file_name = ['SDR_540p']
# base_dir = '/data/datasets/VideoSR/'
# save_dir = '/data/datasets/VideoSR/image/test_lr/'

file_name = ['hres']
base_dir = '../dataset/'
save_dir = '../dataset/img_hr/'

for fi in file_name:
    new_name = base_dir + fi + '/*.mp4'
    video_lists = sorted(glob.glob(new_name))
    for video in video_lists:
        video_name = video.split('/')[-1]
        video_name = video_name.split('.')[0]
        if not os.path.exists(save_dir + video_name):
            os.mkdir(save_dir + video_name)
        command = 'ffmpeg -i {0} -vsync 0 {1}/%3d.png -y'.format(video, save_dir + video_name)
        os.system(command)