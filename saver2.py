import cv2
import time
from masker import Masker
from people_detector import PeopleDetector
from liner import Liner
import os
from vidgear.gears import WriteGear

output_params = {
    "-vcodec":"libx264",
      "-crf": 0, 
      "-preset": "fast",
      "-pix_fmt": "yuv420p",
      "-acodec": "aac",
      "-ar": 22050,
      "-threads": "16"
      } 


weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)

video_path = "ituvideos/3.avi" 
video = cv2.VideoCapture(video_path)    

INPUT_VIDEO_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
INPUT_VIDEO_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

output_path = "ituvideos/3_output.mp4"
writer = WriteGear(output_path, compression_mode=True, logging=True, **output_params)


i = 0
while True:
    start = time.time()
    success, frame = video.read()
    if not success:
        break
    
    mask_file_path = "masks/3.png"
    img_masker = Masker(frame)
    white = img_masker.load_mask(mask_file_path)
    masked_image = img_masker.apply_mask(white)

    resized_frame = detector.resize(masked_image)
    idxs, boxes, confidences = detector.detect(resized_frame)
    boxes_filtered = [boxes[i] for i in idxs]
    
    resized_frame_for_show = detector.resize(frame)

    img_liner = Liner()
    centers = img_liner.calc_centers(boxes_filtered)
    show = img_liner.add_dots(resized_frame_for_show, centers)

    pt1 = (500, 240)
    pt2 = (750, 280)

    # number of people under the line
    n = len([center for center in centers if center[1] > pt1[1] and center[1] > pt2[1]])

    # number of people in the queue
    m = len(centers)

    l1 = (280,550)
    l2 = (750,80)

    total_distance = img_liner._calc_distance(l1, l2)
    L = total_distance / 1280

    # time to pass the line
    current_time = time.time()
    t = current_time - start

    # calculate the flow 
    flow = n / t
    density = m / L
    if flow == 0:
        speed = 1 / density
    else:    
        speed = flow / density

    # normalize the speed between 0 and 1
    speed = speed / 10

    show,mwt = img_liner.put_remaining_times_3(show, centers, speed, l1,l2)

    show = cv2.line(show, l1, l2, (255, 0, 255), 1)

    # draw line with cv2.line
    show = cv2.line(show, pt1, pt2, (0, 255, 255), 1)
    # image = detector.rectangle_detections(resized_frame_for_show, idxs, boxes, confidences)

    # put text on the image for flow and density
    show = cv2.putText(show, "Flow: {:.2f}".format(flow), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    show = cv2.putText(show, "Density: {:.2f}".format(density), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    show = cv2.putText(show, "Speed: {:.2f}".format(speed), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    show = cv2.putText(show, "MWT: {:.2f}".format(mwt), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    writer.write(show)

    i += 1
    
    if i == 2400:
        break

    # cv2.imshow("show", show)
    # cv2.waitKey(1)
    
            
        

video.release()
writer.close()
cv2.destroyAllWindows()

print("Done!")