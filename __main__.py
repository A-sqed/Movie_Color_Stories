################################################################################
# Author: Adrian Adduci
# Contact: faa2160@columbia.edu
################################################################################

import base64
import io
import logging
import math
import pathlib
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from FaissKMeans import FaissKMeans
from matplotlib import cm
from PIL import Image
from sklearn.cluster import KMeans
from datetime import date 
from tqdm import tqdm
import sys, getopt

################################################################################
# Globals
################################################################################

warnings.filterwarnings("ignore", category=DeprecationWarning)  # frameon, matplotlib
path = pathlib.Path(__file__).parent.absolute()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
    datefmt="%H:%M:%S",
    filename="debug.log",
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

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
    x = slice_width #slice_width  # 50
    y = slice_height  # 500
    rgb = 3
    
    first = True
    reordered_histo = []
    bar = np.zeros((y, x, rgb), dtype="uint8")

    # If Even Clusters - Proceed as normal 
    if len(hist) % 2 == 0:
        reordered_histo = zip(hist, centroids)
    else:
        for (percent, color) in zip(reversed(hist), reversed(centroids)):
            if first == True:
                reordered_histo.append((percent, color))
                first = False
            else:
                reordered_histo.insert(0,(percent/2, color))
                reordered_histo.append((percent/2, color))
    
    startY = 0
    
    for tup in reordered_histo:
        percent = tup[0]
        color = tup[1]
        
        #print(f"Percent: {percent}, and color {color}")
        # plot the relative percentage of each cluster
        endY = startY + (percent * y)
        #X,Y 
        start = (0, int(startY))
        end = (x, int(endY) )

        #print(f"Starting at {start} ending at {end}")
        # Draw top-left -> Bottom Right  
        cv2.rectangle(
            bar, start, end, color.astype("uint8").tolist(), -1
        )

        startY = endY

    #bar = np.rot90(bar,1) # Time counterclockwise
    
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
    im = base64.b64decode(frame_buffer_list[0])
    image = Image.open(io.BytesIO(im))
    w, h = image.size
    N = len(frame_buffer_list)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for io_buf in frame_buffer_list:
        im = base64.b64decode(io_buf)
        imarr = np.array(Image.open(io.BytesIO(im)), dtype=np.float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    average_image = Image.fromarray(arr, mode="RGB")
    # average_image.show()

    return average_image

def paint_canvas(average_image, slice_width, slice_height, n_clusters=clusters):
    logger.info(
        f"Plotting Histo, Clusters: {n_clusters}, Frame {f-frames_per_histo} to {f}"
    )

    # Dont send frame, send average of buffer writes to png
    img = np.asarray(average_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # represent as row*column,channel number
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Fit the model using the image
    #clt = KMeans(n_clusters=clusters, verbose=1) #CPU
    clt = FaissKMeans(n_clusters=clusters) #GPU
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_histo(hist, clt.cluster_centers_, slice_width, slice_height)
    im = Image.fromarray(bar)
    #print(f"Size: {im.size}")
    return im

################################################################################
# Main
################################################################################

def main(argv):
   
    file_path = ""
    final_img_name = "final.png"
    final_img_path = str(path)+"\\"+str(final_img_name)
    
     
    try:
        opts = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print(f'{path} -i {file_path} -o {final_img_name}')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print("-i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file_path = arg
        elif opt in ("-o", "--ofile"):
            final_img_name = arg
            final_img_path = str(file_path)+"\\"+str(final_img_name)
        if file_path == final_img_path:
            print("Cannot overwrite the movie file")
            sys.exit()

    cap = cv2.VideoCapture(file_path)
       
    try:
        if cap.isOpened() == False:
            print("Video Path: ", file_path)
    except ValueError as e:
        print("Error opening video stream or file")
        print(f"Unexpected {e=}, {type(e)=}")
        sys.exit()

    frame_buffer_list = []

    histo_count = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
    frames_per_histo = math.floor(fps)
    total_histos = math.floor(frame_count / frames_per_histo)

    # Movie dicates the width of the canvas
    slice_width = 1
    target_canvas_width = math.ceil(slice_width * total_histos)
    slice_height = 2160  # math.ceil(target_canvas_width*.75)
    clusters = 3

    # Blank Canvas
    final_image = Image.new("RGB", (target_canvas_width, slice_height), (255, 255, 255))
    final_image = final_image.save(final_img_path)

    print(f"Video Length: {round(frame_count/(fps*60))} min, at {fps} FPS - {frame_count} Frames")
    print(f"Total K-Means to Review: {total_histos}")
    print(f"Slice: {slice_width} pixel(s), Canvas: {target_canvas_width}(W) x {slice_height}(H)")
    print(f"Saving the final image image at {final_img_path}")
    print(f"Beginning at {date.today()}")

   ################################################################################
    # Output Settings
    ################################################################################

    # Set range to be a mod of FPS target for f in range(0, frame_count):
    #pbar = tqdm(range(frame_count))

    for f in range(frame_count):
        #pbar.set_description("Processing Histogram %s" % str(histo_count + 1))
        percent = "{:.2%}".format(f/frame_count)
        print(f"Reviewing Frame {f} of {frame_count} - {percent}", end="\r")
        # Isolate the frame
        cap.set(1, f)

        ret, frame = cap.read()

        # Stop if video runs out
        if not ret:
            break

        # encode
        retval, buffer = cv2.imencode(".png", frame)
        png_as_text = base64.b64encode(buffer)

        # Hold histogram image for processing
        frame_buffer_list.append(png_as_text)

        # If you have processed enough frames for target fps
        if len(frame_buffer_list) == frames_per_histo:

            # Average the list of frames seen
            averaged_frame = average_frames(frame_buffer_list)

            # Find a k-cluster histogram of the average
            image = paint_canvas(averaged_frame, slice_width, slice_height)

            #image = image.rotate(-90, resample=PIL.Image.NEAREST, expand=1)
            final = Image.open(final_img_path)
            final_copy = final.copy()
            final_copy.paste(image, (histo_count * slice_width, 0))
            final_copy.save(final_img_path, "png")

            frame_buffer_list = []
            histo_count += 1
            logger.info("{:.2%} of total video analyzed".format(f / frame_count))

    print(f"Final image at {final_img_path}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])