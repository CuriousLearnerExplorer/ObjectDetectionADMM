import torch

# To suppress deprecation error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torchvision.io import write_video

"""
Let us assume the input tensors are greyscale with one output channel
in THW format (Time-Height-Width).
Take the non-zero (non-black) positions of the first tensor,
and mark the corresponding positions in the second tensor in a
specified color.
The result is saved separately in a 3-Channel format.

This is intended as a way to translate what the mathematical model
predicts for our model into the application world by painting
the corresponding pixels in a color of choice.
"""

class Detection:

    def __init__(self, video, color):
        # Video
        self.video = video

        self.color = color # triple of rbg-values
    
    def paint_detected_area(self, model, text):
        # Expansion of both tensors into THWC format
        model_helper = list()

        # The pixels where model is non-zero will be set in the original video
        # to corresponding rbg-channel value in self.color
        for i in range(3):
            model_helper.append(torch.where(model > 0, self.color[i], self.video).unsqueeze(3))
        
        # Concatenate the color channels
        detected_model = torch.cat([model_helper[0], model_helper[1], model_helper[2]], dim = 3)

        # save video with the object painted in the specified color
        # Customize name in which detected object will be stored
        write_video(filename=f'./detected_object/DetectedObject_{text}.mp4', video_array=detected_model, fps = 5)

        # No error
        return 0

