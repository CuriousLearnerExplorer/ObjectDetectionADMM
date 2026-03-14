# To take a screenshot of the video to observe the object
import torch
from torchvision.io import read_video, VideoReader
from torchvision.utils import save_image

"""
Screenshots the video given at video path 'video_path' at frame number 'frame_number'
Change the input and output paths and framenumber accordingly as needed, as long 
the input video has at least 'frame_number' many frames (else an exception is thrown) 
"""
if __name__ == '__main__':

    "GET VIDEO IN TENSOR FORMAT WITH GRAYSCALE VALUE"
    # Object after niters = 50 or 500 iterations
    niter = 500 # Set to 50 and 500 and execute each run
    video_path = "./detected_object/DetectedObject_{niter}_iters.mp4"

    # Frame we want to screenshot
    # We choose 24, since in this frame, both vehicles are in the video
    frame_number = 24 
    frames, _, _ = read_video(video_path, output_format='TCHW')

    new_path = './detected_object/object_{niter}.jpg'
    # Save image
    # Need to transform tensor to float and normalize [0,255] to [0,1] by internal working of save_image
    save_image(tensor = frames[frame_number].float() / 255, fp = new_path)
    

