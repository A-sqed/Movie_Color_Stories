
## Overview:

This project is intended to create a timeline of the 'color story' of a video file. The output will be a PNG file wherein each vertical pixel is a histogram of the 
five primary colors as seen in each second of the movie. 

For example, a movie that is 240 frames long, shot at 24 FPS will be ten vertical 
bars wide by a set height (default is 2160 px). Therefore, the final output is composed as follows:

1. Using ing OpenCV, combine 1 second of video frames into an average image 
2. KNN 5-point clustering analyzes the image and outputs a histogram showing the proportional contribution of each cluster to the total image composition
3. Paint the histogram in chronological order to the final image 

## Example Outputs:
Fantasia:
![alt text](https://github.com/A-sqed/Movie_Color_Stories/blob/ae72674637964095978382ca96a592e618fd8839/example_output/fantasia.png)

Nightmare Before Christmas:
![alt text](https://github.com/A-sqed/Movie_Color_Stories/blob/ae72674637964095978382ca96a592e618fd8839/example_output/nightmare_before_christmas.png)



## Installation:

Use pip install -r requirements.txt to install the required libraries.

Adjust location of input/output files within ___main___.py
