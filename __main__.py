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

# Make the dict a list
def average_images(img_dict, num_frames_to_average):
    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(img_dict[1].size)
    N = num_frames_to_average+1
    
    # Create a numpy array of floats to store the average (assume RGB images)
    arr=np.zeros((h,w,3),np.float)
    
    for im in imlist:
        imarr=np.array(Image.open(im),dtype=np.float)
        arr=arr+imarr/N
    
#path = ("B:\\_projects\\_the_little_mermaid.")
slice_height = 1900
slice_width = 500
canvas_width = 3840
images_processed = 1

# New image to hold final canvas 
final_image = Image.new("RGB", (canvas_width, slice_height), (255, 255, 255))

path = "B:\\Videos\\_BLACK_13.mp4"
print("Video Path: ", path)

cap = cv2.VideoCapture(path)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames_to_average, stub_frames = frame_to_average(frame_count, canvas_width)
stub_period = False
frame_dict = {key:[] for key in range(1, num_frames_to_average+1)}

print(f"Total Frame Count: {frame_count}, Averaging {num_frames_to_average} Frames")


for f in range(41000,41001):
    
    if (frame_count - f) == stub_frames:
        stub_period = True 
        
    ret, frame = cap.read()    

    logger.info(f"Converting Frame {f}")
    _, JPEG = cv2.imencode('.jpeg', frame)
    img = cv2.cvtColor(JPEG, cv2.COLOR_BGR2RGB)

    logger.info(f"Reshaping Frame {f}")
    #represent as row*column,channel number
    img = img.reshape((img.shape[0] * img.shape[1],3)) 

    #cluster number
    n_clusters = 4
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
    im = im.resize((slice_height, slice_width))
    im = im.rotate(-90, PIL.Image.NEAREST, expand = 1)   
    
    # Hold histogram image for processing
    frame_dict[images_processed] = im
    
    if images_processed == num_frames_to_average+1: 
        
        # Send to Averaging method 
        
        images_processed = 1
    else:
        images_processed += 1
    
    buffer.close()
    im.show()
    #plt.show()


cap.release()
cv2.destroyAllWindows()

