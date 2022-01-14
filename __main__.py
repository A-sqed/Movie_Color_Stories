################################################################################
# Author: Adrian Adduci
# Contact: faa2160@columbia.edu
################################################################################

from decimal import ROUND_DOWN
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
from PIL import Image
import PIL
import tqdm
import math
import pathlib
import io 
import base64
path = pathlib.Path(__file__).parent.absolute()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
#fh = logging.FileHandler('.log')
logging.basicConfig(format="%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
                    datefmt="%H:%M:%S",
                    filename="debug.log")
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#fh.setLevel(logging.INFO)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#ch.setFormatter(formatter)
# add the handlers to the logger
#logger.addHandler(fh)
# Print to console
#logger.addHandler(ch)

################################################################################
# Helpers
################################################################################

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    
    # Size the Bar 
    bar = np.zeros((5, 500, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)

        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        
        startX = endX

    # return the bar chart
    return bar

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width*height

def frame_to_average(frames, canvas_width = 3840):
    #4k Horizontal 3840 
    #1080 Horizontal 1920
    average =  frames / canvas_width
    average = math.floor(average)
    stub_frames = frames % canvas_width
    return average, stub_frames

def average_frames(frame_buffer_list):
    logger.info(f"Finding an average image of {len(frame_buffer_list)} frames")

    # Assuming all images are the same size, get dimensions of first image
    im =  base64.b64decode(frame_buffer_list[0])
    image = Image.open(io.BytesIO(im))
    w, h = image.size
    N = len(frame_buffer_list)
    
    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h,w,3),np.float)
    
    # Build up average pixel intensities, casting each image as an array of floats
    for io_buf in frame_buffer_list:
        im =  base64.b64decode(io_buf)
        imarr = np.array(Image.open(io.BytesIO(im)), dtype=np.float)
        arr = arr+imarr/N
    
    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)
    
    # Generate, save and preview final image
    average_image = Image.fromarray(arr, mode="RGB")
    
    # Check how frames are being averaged
    #average_image.save("Average.png")
    #average_image.show()
    return average_image  
 
def plot_histogram(average_image):
        
        # Dont send frame, send average of buffer writes to jpeg
        img = np.asarray(average_image) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Confirmed Same
        #cv2.imshow('test', img)

        logger.info(f"Reshaping Frame")
        
        #represent as row*column,channel number
        img = img.reshape((img.shape[0] * img.shape[1],3)) 

        #cluster number
        n_clusters = 6
        logger.info(f"KMeans, Clusters: {n_clusters}, Frame: {f}")
        
        # Fit the model using the image 
        clt = KMeans(n_clusters=n_clusters) 
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)
        
        plt.axis("off")
        plt.imshow(bar)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        im = Image.open(buffer)
        return im

################################################################################
# Output Settings
################################################################################

path = "B:\\Videos\\_BLACK_22.mp4"
print("Video Path: ", path)

cap = cv2.VideoCapture(path)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

target_fps = 1 #How many FPS for each histo
histo_count = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(cap.get(cv2.CAP_PROP_FPS))
hop = round(fps / target_fps)
stub_period = False

frame_buffer_list = []
frame_list = []

max_canvas_width = 3840 #4K max width
total_histos = frame_count/fps
#slice_width = math.floor(max_canvas_width / total_histos)
slice_width = 50
target_canvas_width = math.ceil(slice_width*total_histos)
#slice_height = math.ceil(target_canvas_width/2)
slice_height = 1900
target_canvas_height = slice_height

# New image to hold final canvas & make it white
final_image = Image.new("RGB", (target_canvas_width, slice_height), (255, 255, 255))
num_frames_to_average, stub_frames = frame_to_average(frame_count, target_canvas_width)

print(f"Approx Video Length: {round(frame_count/(fps*60))} Minutes, at {fps} FPS - {frame_count} Total Frames")
print(f"Setting slice to {slice_width} pixels (w) and canvas to {target_canvas_width}(W) x {target_canvas_height}(H)")

# Set range to be a mod of FPS target 
for f in range(46000, 46024):
    
    # Isolate the frame
    cap.set(1,f)
    
    if (frame_count - f) == stub_frames:
        stub_period = True 
    
    ret, frame = cap.read()
    
    if not ret: 
        break

    # encode
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
        
    # Hold histogram image for processing
    frame_buffer_list.append(jpg_as_text)
    
    # If you have processed enough frames for target fps 
    if len(frame_buffer_list) == hop: 

        # Average the list of frames seen
        averaged_frame = average_frames(frame_buffer_list)
        
        # Find a k-cluster histogram of the average 
        image = plot_histogram(averaged_frame)
        histo_count += 1
        
        # Change Size of histo
        image = image.resize((slice_height, slice_width))
        image = image.rotate(-90, PIL.Image.NEAREST, expand = 1)
        image.show()
        frame_buffer_list = []
    
        logger.info("{:.2%} of total video analyzed".format(f/frame_count))

cap.release()
cv2.destroyAllWindows()
