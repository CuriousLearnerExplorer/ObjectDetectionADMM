# Application of the Quasi-Stationary Proximal ADMM in Object Detection

## Background

Please read the file *Report.pdf*, which is Chapter 6 of the thesis, to understand the model, the derived algorithm and numerical evaluations.

## Installation of Necessary Packages

Please install the necessary packages by executing "pip install -r requirements.txt" in the terminal.

## Structure:
    
- **data**: This package has the original sample video *Parking.MOV* from the
Dragon Lake Parking Dataset from the Model Predictive Control (MPC) Lab (https://sites.google.com/berkeley.edu/dlp-dataset)
and stores the transformed, discretized video as described in Chaper 6.5 *Report.pdf*.

- **videoprocessing**: This submodule implements *tensorconverter.py* and *screenshot_object.py*, which are executable scripts. 
    - *tensorconverter.py* takes the video living on the path given by the variables *video_name*, *video_ending* and *video_path* and applies the discretization and makes the video grayscale.
    - *screenshot_object.py* takes a video living at the path given by *video_path*, makes a screenshot of the video at the specified *frame_number* and stores it in the path given by new_path as in the JPG-format.
    
- **detected_object**: This submodule is only meant to store the detected objects of the algorithm.

- **detection**: This submodule has *Detection.py*, which is meant to mark the area of the original video that the algorithm detects 
as the objects, to visually evaluate it. It only consists of one class.

- **algorithm**: This submodule has the file *TensorADMM.py*, which implements an algorithm instance of Algorithm 5 (see *Report.pdf*) for the specified hyperparameters in form of a class
and executes each iteration along with logging of important quantities.

- **plot_statistics**: The corresponding plot statistics, which include the augmented Lagrangian values (also cut off after 50 iterations, and individually for $\beta \in \{10, 20\}$), the objective values and the feasibility violation, as well as the sufficient descent (P2) and subgradient bound (P3) properties for 100 iterations.

## Usage:
The results of executing main.py are already stored in the corresponding submodules as described in the section **Structure**.   
In the following, we describe the necessary changes for new input videos.
<br/> <br/>

### For the Same Video from the Experiments

- Due to technical considerations, the video is not directly provided, since it is too big. Please download the sample video from https://sites.google.com/berkeley.edu/dlp-dataset and store it in the submodule */data* as *Parking.mov*. The video should live at *./data/Parking.mov*
- Please proceed with the steps for "Changes for New Video Input"; one only needs to execute the files, no change is needed.

### Changes for New Video Input

- To use the same video: 
- To use another video, please store it in the submodule **data**. 
- Then, please rename the video and image path names as well as change the size variables in the affected file
*tensorconverter.py*, like described in the heading of that file, if necessary. 
Please execute this preprocessing script first, i.e.
begin with *./videoprocessing/tensorconverter.py*.  
- All other hyperparameters may be changed in *main.py* within lines 181 and 202.
If the user wishes to see the detection result in the video, please uncomment line 231. The value of $\beta$ and numbers of iterations can be set as wished in lines 227 and 228.
Then please execute main.py. 
Warning: The code execution time might take long, around 5 minutes for 50 iterations and around an hour for 500 iterations.
- At last, if one wishes to take a screenshot of the detected object, please change *screenshot_object.py* in the manner
described in the heading of that file and execute it.  



 




