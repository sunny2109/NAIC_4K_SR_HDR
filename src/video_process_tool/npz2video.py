import argparse
import os
import os.path as path
import shutil
import subprocess as sp
import typing
import sys
from glob import glob
import numpy as np

import threading

import utils
import yuv_stream_helper as yuvhelper


FFMPEG_BIN = utils.FFMPEG_BIN
FFPROBE_BIN = utils.FFPROBE_BIN

VIDEO_FILE_EXTS = ['.mp4']


def npz2video(npz_dir:str, video_path:str, pix_fmt:str, crf:int, multi_thread:bool=False):
    assert path.isdir(npz_dir)
    assert not path.exists(video_path)

    npz_files = glob(path.join(npz_dir, '*.npz'))
    npz_files.sort()
    # print('  The dir "{}" containes {} npz files'.format(npz_dir, len(npz_files)))

    f0 = np.load(npz_files[0])
    height, width = f0['y'].shape

    tempfile = utils.TempFileInMem()
    with open(tempfile.path, mode='xb') as f:
        writer = yuvhelper.RawYUVWriter(f, width=width, height=height, pix_fmt=pix_fmt)
        
        for _ in npz_files:
            frame_pos = writer.frame_position
            supporse_filename = '{:03d}'.format(frame_pos)
            actual_filename = path.split(npz_files[frame_pos])[1]
            assert actual_filename.startswith(supporse_filename)
            
            data = np.load(npz_files[frame_pos])

            writer.write_one_frame_via_list([data['y'], data['u'], data['v']])

    vcodec = ['h264', 'hevc'][1]

    # block ffmpeg to avoid stdin stdout stderr loss after program exit
    if multi_thread:
        ffmpeg_lock.acquire()

    sp.run([FFMPEG_BIN, '-f', 'rawvideo', '-s', '{}x{}'.format(width, height),
        '-r', '24000/1001', '-pix_fmt', str(pix_fmt),
        '-i', tempfile.path, '-preset', 'slow', '-vcodec', vcodec, '-crf', str(crf), video_path],
        check=True, stdout=sp.PIPE, stderr=sp.PIPE)

    if multi_thread:
        ffmpeg_lock.release()

    return video_path

def worker():
    while True:
        params = tasks.get()
        if params is None:
            break
        npz2video(params[0][0], params[0][1], params[0][2], params[0][3], multi_thread=True)
        print('{}/{}: {}'.format(params[1][0], params[1][1], params[1][2]))
        sys.stdout.flush()
        tasks.task_done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos to npz files. One folder for each video file.')
    parser.add_argument('--npz_parent_dir', type=str, required=True, help='Directory of a folder containing folders containing npz files')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory of a folder containing video files')
    parser.add_argument('--pix_fmt', type=str, required=True, help='pixel format used for output video. '
                        'Possible values are yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p10le, yuv444p10le')
    parser.add_argument('--crf', type=int, default=18, required=False, help='Set the quality/size tradeoff. '
                        'Valid range is 0 to 63, higher numbers indicating lower quality and smaller output size. [18 ~ 23]')
    parser.add_argument('--multi_thread', action='store_true', help='set to use multi-thread processing')
    args = parser.parse_args()

    assert path.isdir(args.npz_parent_dir)
    assert args.pix_fmt in yuvhelper.PIX_FMT_FACTOR.keys()
    if not utils.is_empty_or_not_exist(args.video_dir):
        i = input('overwrite existing files ? (y/N)')
        if 'y' in i or i == '':
            shutil.rmtree(args.video_dir)

    if not path.exists(args.video_dir):
        os.makedirs(args.video_dir)

    npz_all_dirs = glob(path.join(args.npz_parent_dir, '*'))
    npz_all_dirs.sort()

    npz_dirs = []
    for npz_dir in npz_all_dirs:
        if not path.isdir(npz_dir):
            continue
        for i in VIDEO_FILE_EXTS:
            if npz_dir.endswith(i):
                npz_dirs.append(npz_dir)

    if not args.multi_thread:
        #### single thread:
        for i, npz_dir in enumerate(npz_dirs, 1):
            video_name = path.split(npz_dir)[1]
            npz2video(npz_dir, 
                path.join(args.video_dir, video_name),
                args.pix_fmt,
                args.crf
                )
            print('{}/{}: {}'.format(i, len(npz_dirs), video_name))
        print('Done.')

    if args.multi_thread:
        #### multi thread:
        from queue import Queue
        global tasks
        tasks = Queue()
        global ffmpeg_lock
        ffmpeg_lock = threading.Lock()
        threads = []
        num_worker_threads = max(2, os.cpu_count() // 2, os.cpu_count() - 8)
        num_worker_threads = 8

        for i, npz_dir in enumerate(npz_dirs, 1):
            video_name = path.split(npz_dir)[1]
            tasks.put([[npz_dir, path.join(args.video_dir, video_name), args.pix_fmt, args.crf],
                [i, len(npz_dirs), video_name]])

        for i in range(num_worker_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        tasks.join()

        for i in range(num_worker_threads):
            tasks.put(None)
        for t in threads:
            t.join()

