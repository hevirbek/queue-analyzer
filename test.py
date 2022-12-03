import os
import cv2
from people_detector import PeopleDetector

image_path = "images/test2.jpg"

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)

image = cv2.imread(image_path)

idxs, boxes, confidences = detector.detect(image)

people_count = len(idxs)

image = detector.rectangle_detections(image, idxs, boxes, confidences)
image = detector.resize(image)
image = detector.put_count_text(image, people_count)

cv2.imshow("Image", image)
cv2.waitKey(0)


# video_path = "videos/test4.mp4"

# weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
# configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)

# vs = cv2.VideoCapture(video_path)

# while True:
#     (grabbed, frame) = vs.read()

#     if not grabbed:
#         break

#     idxs, boxes, confidences = detector.detect(frame)
#     frame = detector.rectangle_detections(frame, idxs, boxes, confidences)
#     frame = detector.resize(frame)
#     frame = detector.put_count_text(frame, len(idxs))

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break


# vs.release()
# cv2.destroyAllWindows()
