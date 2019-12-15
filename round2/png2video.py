import os

base_dir = 'dataset'
file_name = sorted(os.listdir(base_dir))

for f in file_name:
	name = base_dir + f
	command = 'ffmpeg -r 24000/1001 -i {0}/%3d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {1}.mp4'.format(name, f)
	os.system(command)