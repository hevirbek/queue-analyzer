import cv2
import os
from people_detector import PeopleDetector
from liner import Liner

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)
img_liner = Liner()

video_path = "videos/test5.webm"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret:
        idxs, boxes, confidences = detector.detect(frame)
        boxes_filtered = [boxes[i] for i in idxs]

        centers = img_liner.calc_centers(boxes_filtered)
        outliers_removed = img_liner.remove_outliers(centers)
        m,b = img_liner.find_best_line(outliers_removed)

        endpoints = img_liner.get_endpoints(boxes_filtered)
        image = img_liner.add_endpoint_lines(frame, m, endpoints[0], endpoints[1])

        centers_in_queue =img_liner.get_centers_in_queue(centers, m, endpoints[0], endpoints[1])
        image = img_liner.add_dots(image, centers_in_queue)

        image = img_liner.put_queue_size_text(frame, len(centers_in_queue))
        image = detector.resize(image)

        cv2.imshow("Frame", image)
        cv2.waitKey(1)
    else:
        break


cap.release()
cv2.destroyAllWindows()
