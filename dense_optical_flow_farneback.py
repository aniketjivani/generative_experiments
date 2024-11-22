# %%
# limitations of optical flow methods:
# good results for Lambertian surfaces (diffusely reflecting surface) under constant illumination, 
# in general they struggle to handle non-homogeneous brightness conditions, outliers, flow discontinuities, large motions, atmospheric effects (fog, mist), low light (https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Optical_Flow_in_the_Dark_CVPR_2020_paper.pdf - synthesize training data based on noise model for low light) - so not a great replacement for shooting more frames in all scenarios.

# %%
import cv2
import numpy as np

# Hue (H) - represents the type of color. 0 - 360 degrees, red is 0, green is 120, blue is 240
# Saturation (S) - 0 - 100%, 0% is gray, 100% is fully saturated color (VIVIDNESS)
# Value (V) - 0 - 100%, 0% is black, 100% is white. (BLACK TO FULLY BRIGHT)

# Use Farneback optical flow and visualize the flow field via HSV and vector field plots.
# See: https://medium.com/@RiwajNeupane/motion-and-object-tracking-42eb1e6a5443

# Bolt-detection video source: https://github.com/intel-iot-devkit/sample-videos/blob/master/bolt-detection.mp4

example="bolt"
# example="shibuya"


# %%

if example == "bolt":
    cap = cv2.VideoCapture('video_samples/bolt-detection.mp4') # 1024 x 576
elif example == "shibuya":
    cap = cv2.VideoCapture('video_samples/shibuya.mp4') # 1280 x 720

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

print(width, height)
# %%

# Source: https://github.com/chuanenlin/optical-flow/blob/master/dense-solution.py

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

# %%
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

# %%
while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv2.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# %%
# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()
