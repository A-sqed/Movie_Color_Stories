```
## Overview:

This project is intended to create a timeline of the 'color story' of a video file. The output will be a PNG file wherein each vertical pixel is a histogram of the 
five primary colors are seen in each second of the movie. 

For example, a movie that is 240 frames long, shot at 24 FPS will be ten vertical 
cars wide. Therefore, the final output is composed as follows:

1. Using ing OpenCV, combine 1 second of video frames into an average image 
2. KNN 5-point clustering analyzes the image and outputs a histogram showing the proportional contribution of each cluster to the total image composition
3. Paint the histogram in chronological order to the final image 
```
```
## Example Outputs:
Fantasia:
![alt text]()

Nightmare Before Christmas:
![alt text]()

```

```
## Installation:

Use pip install -r requirements.txt to install the required libraries.

Adjust location of input/output files within ___main___.py
```