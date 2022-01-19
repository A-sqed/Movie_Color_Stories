################################################################################
# Author: Adrian Adduci
# Contact: faa2160@columbia.edu
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
################################################################################

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
from PIL import Image
import PIL
from tqdm import tqdm
import math
import pathlib
import io 
import base64, warnings
from matplotlib import cm

################################################################################
# Globals
################################################################################

warnings.filterwarnings("ignore", category=DeprecationWarning) #frameon, matplotlib
path = pathlib.Path(__file__).parent.absolute()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
                    datefmt="%H:%M:%S",
                    filename="debug.log")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

path = "B:\\Videos\\_BLACK_22.mp4"
final_img_path = "final.png"
cap = cv2.VideoCapture(path)

try:
    if (cap.isOpened()== False):
        print("Video Path: ", path)
except ValueError as e:
    print("Error opening video stream or file")
    print(f"Unexpected {e=}, {type(e)=}")

frame_buffer_list = []
frame_list = []


histo_count = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
frames_per_histo = math.floor(fps)
total_histos = math.floor(frame_count/frames_per_histo)

# Movie dicates the width of the canvas
slice_width = 1
target_canvas_width = math.ceil(slice_width*total_histos)
slice_height = 2160 #math.ceil(target_canvas_width*.75)

# Blank Canvas 
final_image = Image.new("RGB", (target_canvas_width, slice_height), (255, 255, 255))
final_image = final_image.save(final_img_path)

print(f"Approx Video Length: {round(frame_count/(fps*60))} Minutes, at {fps} FPS - {frame_count} Total Frames")
print(f"Total K-Means to Review: {total_histos}")
print(f"Setting slice to {slice_width} pixel(s) and canvas to {target_canvas_width}(W) x {slice_height}(H)")

################################################################################
# Function defs 
################################################################################

def find_histogram(clt):
    """
    create a histogram of k clusters
    :param: clt
    :return: hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_histo(hist, centroids, slice_width, slice_height):
    """
    Plot a bar chart from histpgram
    param: histogram, centroid, width of bar, height of bar
    return:  np array of plot
    """

    # Size the Bar    
    y = slice_width #50
    x = slice_height #500
    #logger.info(f"Creating a bar that is {x} x {y}")
    rgb = 3
    bar = np.zeros((x, y, rgb), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):

        # plot the relative percentage of each cluster
        endX = startX + (percent * y)
        cv2.rectangle(bar, 
                      (int(startX), 0), 
                      (int(endX), x),
                      color.astype("uint8").tolist(), 
                      -1)
        
        startX = endX

    # return the bar chart
    return bar

def average_frames(frame_buffer_list):
    """
    Find and average image from a set of list of images
    param: list of images encoded in base64
    return: PIL image object
    """
    
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
    arr=np.array(np.round(arr), dtype=np.uint8)
    
    # Generate, save and preview final image
    average_image = Image.fromarray(arr, mode="RGB")
    #average_image.show()
    
    return average_image  
 
def paint_canvas(average_image, slice_width, slice_height, n_clusters=5):
        logger.info(f"Plotting Histo, Clusters: {n_clusters}, Frame {f-frames_per_histo} to {f}")
        
        # Dont send frame, send average of buffer writes to png
        img = np.asarray(average_image) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #represent as row*column,channel number
        img = img.reshape((img.shape[0] * img.shape[1],3)) 
        
        # Fit the model using the image 
        clt = KMeans(n_clusters=n_clusters) 
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_histo(hist, clt.cluster_centers_, slice_width, slice_height)
        im = Image.fromarray(bar)
        
        return im
        """
        im.show()
        buffer = io.BytesIO()
        #im.save(buffer,format='png')
        
        im.save(buffer, format='png', 
                    bbox_inches='tight', 
                    frameon=False, 
                    pad_inches=0.0)
        
        buffer.seek(0)
        

        dpi=96 # dots per inch
        plt.axis("off")
        plt.margins(0)
        plt.imshow(bar, interpolation='antialiased', origin='upper')
        
        buffer = io.BytesIO()
        #im.save(buffer,format='png')
        
        plt.savefig(buffer, format='png', 
                    bbox_inches='tight', 
                    frameon=False, 
                    pad_inches=0.0)
        
        buffer.seek(0)
        im = Image.open(buffer)

        return im
        """
################################################################################
# Output Settings
################################################################################

# Set range to be a mod of FPS target for f in range(0, frame_count):
pbar = tqdm(range(frame_count))

for f in pbar:
    pbar.set_description("Processing Histogram %s" % str(histo_count+1))
    
    # Isolate the frame
    cap.set(1,f)
        
    ret, frame = cap.read()
    
    # Stop if video runs out
    if not ret: 
        break

    # encode
    retval, buffer = cv2.imencode('.png', frame)
    png_as_text = base64.b64encode(buffer)
        
    # Hold histogram image for processing
    frame_buffer_list.append(png_as_text)
    
    # If you have processed enough frames for target fps 
    if len(frame_buffer_list) == frames_per_histo: 

        # Average the list of frames seen
        averaged_frame = average_frames(frame_buffer_list)
        
        # Find a k-cluster histogram of the average 
        image = paint_canvas(averaged_frame, slice_width, slice_height)
        
        # Change Size of histo
        #image = image.rotate(-90, resample=PIL.Image.NEAREST, expand = 1)
        #print(f"Image size before resize: {image.size}")
        #image = image.resize((slice_width, slice_height))

        final = Image.open(final_img_path)
        final_copy = final.copy()      
        final_copy.paste(image, ( histo_count*slice_width, 0))
        final_copy.save(final_img_path, 'png')
        
        frame_buffer_list = []
        histo_count += 1
        logger.info("{:.2%} of total video analyzed".format(f/frame_count))

print(f"Final image at {path+final_img_path} complete")
cap.release()
cv2.destroyAllWindows()


# create file handler which logs even debug messages
#fh = logging.FileHandler('.log')
#fh.setLevel(logging.INFO)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#ch.setFormatter(formatter)
# add the handlers to the logger
#logger.addHandler(fh)
# Print to console
#logger.addHandler(ch)