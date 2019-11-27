import argparse
import os
import os.path as path
import shutil
import subprocess as sp
import typing
from sys import platform
from glob import glob
import numpy as np

import threading

import utils
import yuv_stream_helper as yuvhelper


FFMPEG_BIN = utils.FFMPEG_BIN
FFPROBE_BIN = utils.FFPROBE_BIN

VIDEO_FILE_EXTS = ['.mp4']

def save_yuv_to_npz(npz_path:str, data:typing.List[np.ndarray]):
    assert len(data) == 3
    y = data[0]
    u = data[1]
    v = data[2]
    np.savez_compressed(npz_path, y=y, u=u, v=v)

def worker():
    while True:
        param = tasks.get()
        if param is None:
            break
        save_yuv_to_npz(param[0], param[1])
        tasks.task_done()

def video2npz(video_path:str, npz_dir:str, multi_thread:bool=False):
    assert path.isfile(video_path)
    assert utils.is_empty_or_not_exist(npz_dir)
    if not path.exists(npz_dir):
        os.makedirs(npz_dir)

    tempfile = utils.TempFileInMem()
    sp.run([FFMPEG_BIN, '-i', video_path, '-vcodec', 'rawvideo', tempfile.path], check=True, stdout=sp.PIPE, stderr=sp.PIPE)
    # sp.run('{} -i {} -vcodec rawvideo {}'.format(FFMPEG_BIN, video_path, tempfile.path),
    #         check=True, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

    video_info_dict = utils.get_video_info_dict(video_path)

    if multi_thread:
        from queue import Queue
        global tasks
        tasks = Queue()
        threads = []
        num_worker_threads = max(2, os.cpu_count() // 2, os.cpu_count() - 8)
        # num_worker_threads = 2
        for i in range(num_worker_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

    with open(tempfile.path, mode='rb') as f:
        reader = yuvhelper.RawYUVReader(stream=f, width=video_info_dict['width'], height=video_info_dict['height'], pix_fmt=video_info_dict['pix_fmt'])

        while True:
            frame_pos = reader.frame_position
            try:
                y,u,v = reader.read_one_frame_via_list()
            except EOFError as e:
                break
            else:
                npz_path = path.join(npz_dir, '{:03d}.npz'.format(frame_pos))
                assert not path.isfile(npz_path)
                if not multi_thread:
                    np.savez_compressed(npz_path, y=y,u=u,v=v)
                elif multi_thread:
                    tasks.put([npz_path, [y,u,v]] )
    
    if multi_thread:
        tasks.join()
        for i in range(len(threads)):
            tasks.put(None)
        for t in threads:
            t.join()

    return video_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos to npz files. One folder for each video file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing input video files')
    parser.add_argument('--npz_parent_dir', type=str, required=True, help='Directory containing output folders which will be filled by npz files')
    parser.add_argument('--multi_thread', action='store_true', help='set to use multi-thread processing')
    args = parser.parse_args()

    assert path.isdir(args.video_dir)
    if not utils.is_empty_or_not_exist(args.npz_parent_dir):
        i = input('overwrite existing files ? (y/N)')
        if 'y' in i or i == '':
            shutil.rmtree(args.npz_parent_dir)

    if not path.exists(args.npz_parent_dir):
        os.makedirs(args.npz_parent_dir)

    files_in_video_path = glob(path.join(args.video_dir, '*'))
    files_in_video_path.sort()
    
    video_paths = []
    for i in files_in_video_path:
        for ext in VIDEO_FILE_EXTS:
            if i.endswith(ext):
                video_paths.append(i)

    for i,video_path in enumerate(video_paths, 1):
        video_name = path.split(video_path)[1]
        video2npz(video_path, # path to video1
            path.join(args.npz_parent_dir, video_name), # path to video1 npz dir
            args.multi_thread # weather multi thread
            )
        print('{}/{}: {}'.format(i, len(video_paths), video_name))
    print('Done.')
