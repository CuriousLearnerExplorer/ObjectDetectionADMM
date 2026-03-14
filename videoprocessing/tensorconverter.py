# For video processing and is deprecated, 
# but only experimental Windows support for torchdecoders
# See: https://github.com/pytorch/torchcodec
# Last Statis: 25th September 2025

import torch

# To suppress deprecation error
import warnings ; warnings.warn = lambda *args, **kwargs: None

from torchvision.io import write_video, VideoReader
from torchvision.transforms import v2, Resize
from torchvision.transforms.functional import crop

"""
If different video should be used, please change accordingly lines 25 - 27
and lines 31 and 32 (the sizes should be achievable, else an exception is thrown).
Also to customize the discretization, please modify lines 39 and 40.
"""

if __name__ == '__main__':
    device = "cpu"  # or e.g. "cuda" !

    # Path to saved video
    video_name = 'Parking'
    video_ending = '.MOV'
    video_path = './data/' + video_name + video_ending

    reader = VideoReader(video_path, "video")

    h = int(2160 / 12) # Original: 2160 -> 180
    w = int(3840 / 24) # Original: 3840 -> 160

    # To discretize the frames
    resizer = Resize(size=(h, w))
    frame = reader.seek(0)

    # distance between timesteps
    delta_t = 10
    max_n = 32 # how many frames we will leave

    count = 0 # counter of how many frames are extracted
    pos = 0 # position of taken frames in truncated video
    
    # To store video in tensor
    frames = torch.Tensor(size = (max_n, 3, h, w)) 

    # Discretize frame by first cropping the patch we want to regard,
    # and then take only frames after skipping delta_t - 1 frames
    for frame in reader:
        if count % delta_t == 0:
            # Take big (3*h)x(3*w)-patch and compress it to h x w-patch
            frames[pos, :, :, :] = resizer.forward(crop(frame['data'], 1600, 0, 3*h, 3*w))
            pos += 1

        count += 1
        if pos >= max_n:
            break

    discretized_video = frames 

    # take greyscale video with one channel, as the 3-channel
    # variant will lead to an incompatible preservation
    discretized_video_grey = v2.Grayscale(1)(discretized_video)[:,0,:,:]

    # For Greyscaling into 3-Channel form, we use the order that
    # is needed for the read-method
    discretized_video_grey = discretized_video_grey.repeat(3,1,1,1)
    discretized_video_grey = torch.movedim(discretized_video_grey, (0,1,2,3), (3,0,1,2))

    # Store video with 5fps (frames per second)
    write_video(filename=f"./data/Grayscale{video_name}.mp4", video_array=discretized_video_grey, fps = 5)