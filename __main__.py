################################################################################
# Author: Adrian Adduci
# Contact: faa2160@columbia.edu
################################################################################

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
from PIL import Image
import PIL
import tqdm
import pathlib
from io import BytesIO
path = pathlib.Path(__file__).parent.absolute()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler('.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

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

#path = ("B:\\_projects\\_the_little_mermaid.")
path = "B:\\Videos\\_BLACK_13.mp4"
print("Video Path: ", path)

cap = cv2.VideoCapture(path)

if (cap.isOpened()== False):
  print("Error opening video stream or file")
 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total Frame Count: {frame_count}")
#input("Press Enter to continue...")

for f in range(41000,41001):
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
    bar = bar.rotate(90, PIL.Image.NEAREST, expand = 1)
    plt.axis("off")
    plt.imshow(bar)
    plt.show()


cap.release()
cv2.destroyAllWindows()

