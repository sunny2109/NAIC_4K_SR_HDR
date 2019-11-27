import cv2
import numpy as np
import utils
from glob import glob
from os import path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test npzs from npz2video.py.')
    parser.add_argument('--npz_dir', type=str, required=True, help='Directory containing npz files')
    args = parser.parse_args()

    assert path.isdir(args.npz_dir)

    fs = glob(path.join(args.npz_dir, '*.npz'))
    fs.sort()

    cv2.namedWindow('test_npz', cv2.WINDOW_KEEPRATIO)
    for i in fs:
        data = np.load(i)
        y = data['y']
        u = data['u']
        v = data['v']

        u2 = u
        if u2.shape[0] != y.shape[0]:
            u2 = utils.double_dim(u2, 0)
        if u2.shape[1] != y.shape[1]:
            u2 = utils.double_dim(u2, 1)

        v2 = v
        if v2.shape[0] != y.shape[0]:
            v2 = utils.double_dim(v2, 0)
        if v2.shape[1] != y.shape[1]:
            v2 = utils.double_dim(v2, 1)
        
        assert u2.shape == y.shape and v2.shape == y.shape

        yuv = np.stack([y,u2,v2], axis=-1)

        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imshow('test_npz', img)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break
