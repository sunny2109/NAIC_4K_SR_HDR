import io
import typing
import numpy as np

SUPPORT_CODEC = ['h264', 'hevc']

PIX_FMT_FACTOR = {
    # pixel_format : byte length , uv channel factor
    # uv channel factor: width factor, height factor
    'yuv420p':{'b': 1, 'uv':{'w': 2, 'h': 2}},
    'yuv422p':{'b': 1, 'uv':{'w': 2, 'h': 1}},
    'yuv444p':{'b': 1, 'uv':{'w': 1, 'h': 1}},
    'yuv420p10le':{'b': 2, 'uv':{'w': 2, 'h': 2}},
    'yuv422p10le':{'b': 2, 'uv':{'w': 1, 'h': 2}},
    'yuv444p10le':{'b': 2, 'uv':{'w': 1, 'h': 1}},
}

class RawYUVReader(object):
    def __init__(self, stream:io.BufferedIOBase, width:int, height:int, pix_fmt:str):
        assert isinstance(stream, io.BufferedIOBase) and stream.readable() == True
        assert type(width) == int and width > 0
        assert type(height) == int and height > 0
        assert type(pix_fmt) == str and pix_fmt in PIX_FMT_FACTOR.keys()

        if stream.seekable():
            stream.seek(0)

        self.stream = stream
        self.width = width
        self.height = height
        self.pix_fmt = pix_fmt
        self.frame_position = 0
        self.channel = 0 # should cycle among y(0),u(1),v(2)
        
        self.byte = PIX_FMT_FACTOR[self.pix_fmt]['b']
        self.uv_width = self.width // PIX_FMT_FACTOR[self.pix_fmt]['uv']['w']
        self.uv_height = self.height // PIX_FMT_FACTOR[self.pix_fmt]['uv']['h']
        if self.byte == 1:
            self.dtype = 'uint8'
        elif self.byte == 2:
            self.dtype = 'uint16'
        else:
            raise NotImplementedError()
    
    def read_y_channel(self) -> np.ndarray:
        assert self.channel == 0
        data = self.stream.read(self.height * self.width * self.byte)
        if data == b'':
            raise EOFError('EOF reach')
        self.channel = 1
        return np.frombuffer(data, dtype=self.dtype).reshape(self.height, self.width)
    
    def read_u_channel(self) -> np.ndarray:
        assert self.channel == 1
        data = self.stream.read(self.uv_height * self.uv_width * self.byte)
        if data == b'':
            raise EOFError('EOF reach')
        self.channel = 2
        return np.frombuffer(data, dtype=self.dtype).reshape(self.uv_height, self.uv_width)

    def read_v_channel(self) -> np.ndarray:
        assert self.channel == 2
        data = self.stream.read(self.uv_height * self.uv_width * self.byte)
        if data == b'':
            raise EOFError('EOF reach')
        self.channel = 0
        self.frame_position += 1
        return np.frombuffer(data, dtype=self.dtype).reshape(self.uv_height, self.uv_width)
    
    def read_one_frame_via_list(self) -> typing.List[np.ndarray]:
        return [self.read_y_channel(),
                self.read_u_channel(),
                self.read_v_channel()]

class RawYUVWriter(object):
    def __init__(self, stream:io.BufferedIOBase, width:int, height:int, pix_fmt:str):
        assert isinstance(stream, io.BufferedIOBase) and stream.writable() == True
        assert type(width) == int and width > 0
        assert type(height) == int and height > 0
        assert type(pix_fmt) == str and pix_fmt in PIX_FMT_FACTOR.keys()

        if stream.seekable():
            stream.seek(0)

        self.stream = stream
        self.width = width
        self.height = height
        self.pix_fmt = pix_fmt
        self.frame_position = 0
        self.channel = 0 # should cycle among y(0),u(1),v(2)
        
        self.byte = PIX_FMT_FACTOR[self.pix_fmt]['b']
        self.uv_width = self.width // PIX_FMT_FACTOR[self.pix_fmt]['uv']['w']
        self.uv_height = self.height // PIX_FMT_FACTOR[self.pix_fmt]['uv']['h']
        if self.byte == 1:
            self.dtype = 'uint8'
        elif self.byte == 2:
            self.dtype = 'uint16'
        else:
            raise NotImplementedError()

    def write_y_channel(self, data:np.ndarray) -> int:
        assert data.dtype == self.dtype
        assert self.channel == 0
        self.channel = 1 # next channel setup to u
        return self.stream.write(data.tobytes())
    
    def write_u_channel(self, data:np.ndarray) -> int:
        assert data.dtype == self.dtype
        assert self.channel == 1
        self.channel = 2 # next channel setup to v
        return self.stream.write(data.tobytes())
    
    def write_v_channel(self, data:np.ndarray) -> int:
        assert data.dtype == data.dtype
        assert self.channel == 2
        self.channel = 0 # next channel setup to y
        self.frame_position += 1
        return self.stream.write(data.tobytes())
    
    def write_one_frame_via_list(self, channels:typing.List[np.ndarray]) -> int:
        assert len(channels) == 3
        ret = 0
        ret += self.write_y_channel(channels[0])
        ret += self.write_u_channel(channels[1])
        ret += self.write_v_channel(channels[2])
        return ret



        