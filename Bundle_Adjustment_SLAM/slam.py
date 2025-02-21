import numpy as np
import cv2
import os,sys,time,g2o
from triangulation import triangulate
from Camera import denormalize, normalize, Camera
from display import Display
from match_frames import generate_match
from descriptor import Descriptor, Point


F= int(os.getenv("F","500")) # Focal point of the camera
#W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(W,H)
W, H = 1920//2, 1080//2 # Width and height of frames (commonly half of Full HD resolution sizes 1920, 1080)
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]]) # Intrinsic matrix of the system
desc_dict = Descriptor()
# if os.getenv("D3D") is not None:
desc_dict.create_viewer()

# disp = None
# if os.getenv("D2D") is not None:
#disp = Display(W, H)
trajectory = []


def calibrate(image):
    # camera intrinsics...<================Check this
    image = cv2.resize(image, (W,H))
    return image

def generate_SLAM(image):
    image = calibrate(image)
    frame = Camera(desc_dict, image, K)
    if frame.id == 0:
        return
    frame1 = desc_dict.frames[-1]
    frame2 = desc_dict.frames[-2]

    x1,x2,Id = generate_match(frame1,frame2)
    frame1.pose =np.dot(Id,frame2.pose)
    trajectory.append(frame1.pose.copy())

    for i,idx in enumerate(x2):
        if frame2.pts[idx] is not None:
            frame2.pts[idx].add_observation(frame1,x1[i])
    # homogeneous 3-D coords
    #print("Thisis a test1")
    pts4d = triangulate(frame1.pose, frame2.pose, frame1.key_pts[x1], frame2.key_pts[x2])
    pts4d /= pts4d[:, 3:]
    unmatched_points = np.array([frame1.pts[i] is None for i in x1])
    #print("Adding:  %d points" % np.sum(unmatched_points))
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
          continue
        pt = Point(desc_dict, p)
        pt.add_observation(frame1, x1[i])
        pt.add_observation(frame2, x2[i])

    for pt1, pt2 in zip(frame1.key_pts[x1], frame2.key_pts[x2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(image, (u1, v1), color=(0,255,0), radius=1)
        cv2.line(image, (u1, v1), (u2, v2), color=(255, 255,0))

    # 2-D display
    """if disp is not None:
        disp.display2D(image) """
    # 3-D display
    #desc_dict.display()

    #desc_dict.max_frame = frame.id

    if frame.id % 20 == 0:  # Por ejemplo, optimiza cada 5 fotogramas
        #print("Ejecutando optimización de Bundle Adjustment")
        desc_dict.optimize()

if __name__ == "__main__":
    """if len(sys.argv) < 2:
        print("%s takes in .mp4 as an arg" %sys.argv[0])
        exit(-1)
    print("Thisis a test-1")

    cap = cv2.VideoCapture(sys.argv[1]) # Can try Realtime(highly unlikely though) """
    #cap = cv2.VideoCapture("/home/faleivac/Documents/GitHub/TFG_FL_SLAM/MonocularVSlam/output_video.mp4")
    #/home/faleivac/Documents/GitHub/TFG_FL_SLAM/SetDeDatos/video_prueba_1.mp4

    if len(sys.argv) < 2:
        print("%s takes in .mp4 as an arg" %sys.argv[0])
        exit(-1)

    dataset_path = sys.argv[1]
    video_path = os.path.join(dataset_path, 'rgb.mp4')

    if not os.path.exists(video_path):
        print(f"No se encontró el archivo 'rgb.mp4' en la ruta {dataset_path}")
        sys.exit(-1)


    cap = cv2.VideoCapture(video_path) 
    counter_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        counter_frame += 1
        if ret == True:
          #if counter_frame % 3 == 0:
            frame1 = cv2.resize(frame, (720,400)) #Resizing the original window
            """cv2.imshow("Frame",frame1)    
            if cv2.waitKey(1) & 0xFF == ord('q'):   #Quit Condition
                break"""
            generate_SLAM(frame)
        else:
          
          break
    cap.release() 
    cv2.destroyAllWindows()

    trajectory_array = np.array(trajectory)
    np.save('slam_optimize.npy', trajectory_array)
    print("Trayectoria guardada en 'slam_monocular.npy'")