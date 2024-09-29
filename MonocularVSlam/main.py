import cv2
from display import Display
from extractor import Frame, denormalize, match_frames, IRt, add_ones
import numpy as np
from global_functions import triangulate
from pointmap import Map, Point
 
### Camera intrinsics
# define principal point offset or optical center coordinates
W, H = 1920//2, 1080//2
 
# define focus length
F = 270
 
# define intrinsic matrix and inverse of that
K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))
Kinv = np.linalg.inv(K)
 
# image display initialization
display = Display(W, H)
 
# initialize a map
mapp = Map()
mapp.create_viewer()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
 
    # previous frame f2 to the current frame f1.
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
 
    idx1, idx2, Rt = match_frames(f1, f2)
     
    # X_f1 = E * X_f2, f2 is in world coordinate frame, multiplying that with
    # Rt transforms the f2 pose wrt the f1 coordinate frame
    f1.pose = np.dot(Rt, f2.pose)
 
 
    # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
    # returns an array of size (n, 4), n = feature points
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
 
 
    # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]
 
 
    # Reject points without enough "Parallax" and points behind the camera
    # returns, A boolean array indicating which points satisfy both criteria.
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
 
    for i, p in enumerate(pts4d):
        # If the point is not good (i.e., good_pts4d[i] is False), 
        # the loop skips the current iteration and moves to the next point.
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)
 
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
 
        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))
 
    # 2-D display
    display.paint(img)
 
    # 3-D display
    mapp.display()
 
if __name__== "__main__":
    cap = cv2.VideoCapture("/home/faleivac/Documents/GitHub/TFG_FL_SLAM/Dataset/video_prueba_3.mp4")
    counter_frame = 0
    while cap.isOpened():
        print(f"Working on frame {counter_frame}")
        ret, frame = cap.read()
        counter_frame += 1
        if ret == True:
            process_frame(frame)
        else:
            wait = input("Write something and enter to close ")
            print("Ok")
            break