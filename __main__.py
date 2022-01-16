################################################################################
# Author: Adrian Adduci
# Contact: faa2160@columbia.edu
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
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
import base64, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #frameon, matplotlib
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
    x = 50
    y = 500
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

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width*height

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
    arr=np.array(np.round(arr), dtype=np.uint8)
    
    # Generate, save and preview final image
    average_image = Image.fromarray(arr, mode="RGB")
    
    # Check how frames are being averaged
    #average_image.save("Average.png")
    #average_image.show()
    return average_image  
 
def plot_histogram(average_image):
        
        # Dont send frame, send average of buffer writes to png
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
        plt.margins(0)
        #print("Here, this is what matters:")
        plt.imshow(bar, interpolation='antialiased', origin='upper')
        #print("Here, this is what matters:")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', frameon=False, pad_inches=0.0)
        buffer.seek(0)
        im = Image.open(buffer)
        return im

################################################################################
# Output Settings
################################################################################

path = "B:\\Videos\\_BLACK_22.mp4"
final_img_path = "final.png"
cap = cv2.VideoCapture(path)

try:
    if (cap.isOpened()== False):
        print("Video Path: ", path)
except ValueError as e:
    print("Error opening video stream or file")
    print(f"Unexpected {e=}, {type(e)=}")


target_fps = 1 #How many FPS for each histo
histo_count = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(cap.get(cv2.CAP_PROP_FPS))
frames_per_histo = round(fps / target_fps)
total_histos = frame_count/frames_per_histo

frame_buffer_list = []
frame_list = []

canvas_resize_width = 3840 #4K max width

#slice_width = 25
target_canvas_width = 3840
#target_canvas_width = math.ceil(slice_width*total_histos)
slice_width = math.floor(target_canvas_width / total_histos)
slice_height = math.ceil(target_canvas_width*.75)
#slice_height = 1900
target_canvas_height = slice_height

final_image = Image.new("RGB", (target_canvas_width, target_canvas_height), (255, 255, 255))
final_image = final_image.save(final_img_path)

print(f"Approx Video Length: {round(frame_count/(fps*60))} Minutes, at {fps} FPS - {frame_count} Total Frames")
print(f"Setting slice to {slice_width} pixels (w) and canvas to {target_canvas_width}(W) x {target_canvas_height}(H)")

# Set range to be a mod of FPS target 
for f in range(0, frame_count):
    
    # Isolate the frame
    cap.set(1,f)
        
    ret, frame = cap.read()
    
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
        #averaged_frame.show(title='Average frame')
        
        # Find a k-cluster histogram of the average 
        image = plot_histogram(averaged_frame)
        
        # Change Size of histo
        image = image.resize((slice_height, slice_width))
       # print(f"Image before rotate size: {image.size}")
        image = image.rotate(-90, resample=PIL.Image.NEAREST, expand = 1)
        #print(f"Image after rotate size: {image.size}")
        #image.show(title="Histo of average")
        #image.save(f"Histo_{f}.png")
        #print(image.size)
        frame_buffer_list = []
        final = Image.open(final_img_path)
        final_copy = final.copy()      
        final_copy.paste(image, ( histo_count*slice_width, 0))
        #print(f"New Histo Size is: {image.size}")
        #print(f"Printing new imagee at X: {histo_count*slice_width}, Y:{0}")
        final_copy.save(final_img_path)
        
        histo_count += 1
        logger.info("{:.2%} of total video analyzed".format(f/frame_count))
        print("{:.2%} of total video analyzed".format(f/frame_count))

print(f"Final image at {final_img_path} complete")
cap.release()
cv2.destroyAllWindows()
