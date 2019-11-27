import os
import os.path as path
from random import randint
import subprocess as sp
from sys import platform
import numpy as np

if platform.startswith('linux'):
    FFMPEG_BIN = 'ffmpeg'
    FFPROBE_BIN = 'ffprobe'
if platform.startswith('win32'):
    FFMPEG_BIN = 'ffmpeg.exe'
    FFPROBE_BIN = 'ffprobe.exe'

class TempFileInMem(object):
    def __init__(self, prefix:str='py_ffmpeg_read_', dir_path:str='/dev/shm', ext:str='yuv'):
        assert prefix != None
        assert dir_path != None
        assert ext != None
        while True:
            self.path = path.join(dir_path, 
                '{}_{:04}.{}'.format(prefix, randint(0,9999), ext))
            if not path.exists(self.path):
                break

    def __del__(self):
        if path.exists(self.path):
            os.remove(self.path)

def is_empty_or_not_exist(dir_path:str):
    if not path.exists(dir_path):
        return True
    elif len(os.listdir(dir_path)) == 0:
        return True
    return False

def get_video_info_dict(video_path: str) -> dict:
    ''' get video info via ffprobe

    Arguments:
        video_path: path to video file
    '''
    proc = sp.run([FFPROBE_BIN, '-show_streams', video_path], stdout=sp.PIPE, stderr=sp.PIPE, check=True)
    # proc = sp.run('{} -show_streams {}'.format(FFPROBE_BIN, video_path), stdout=sp.PIPE, stderr=sp.PIPE, shell=True, check=True)
    info_str = proc.stdout.decode()
    info_list = info_str.split(os.linesep)
    info_dict = {x.split('=')[0] : x.split('=')[1] for x in info_list if '=' in x}
    keys = info_dict.keys()
    if 'width' in keys:
        info_dict['width'] = int(info_dict['width'])
    if 'height' in keys:
        info_dict['height'] = int(info_dict['height'])
    if 'nb_frames' in keys:
        info_dict['nb_frames'] = int(info_dict['nb_frames'])
    return info_dict


def double_dim(inpt:np.ndarray, dim:int):
    ''' double inpt nparray along dim

    Arguments:
        inpt -- numpy array 
        dim -- double size at
    '''
    shape = list(inpt.shape)
    shape[dim] = shape[dim] * 2
    out = np.stack((inpt, inpt), axis=dim+1).reshape(shape)
    return out

def half_dim(inpt:np.ndarray, dim:int, method:int=1):
    ''' half inpt nparray along dim

    Arguments:
        inpt -- numpy array
        dim -- half size at
        method -- 0,1,2 means average, first value, second value
    '''
    assert len(inpt.shape) == 2

    if dim == 0:
        inpt1 = inpt[::2,:]
        inpt2 = inpt[1::2,:]
    elif dim == 1:
        inpt1 = inpt[:,::2]
        inpt2 = inpt[:,1::2]
    
    assert inpt1.shape == inpt2.shape

    if method==0:
        out = (inpt1.astype('float') + inpt2.astype('float')) / 2
        out = out.round().astype('int')
    elif method==1:
        out = inpt1
    elif method==2:
        out = inpt2
    return out