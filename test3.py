import cv2
import os
from masker import Masker
from people_detector import PeopleDetector
from liner import Liner

image_path = "images/test5.jpg"
mask_file_path = "masks/" + image_path.split("/")[-1].split(".")[0] + "_mask.png"

img = cv2.imread(image_path)
# img_masker = Masker(img)

# white = img_masker.load_mask(mask_file_path)
# masked_image = img_masker.apply_mask(white)

masked_image = img

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)
img_liner = Liner()

idxs, boxes, confidences = detector.detect(masked_image)
people_count = len(idxs)

boxes_filtered = [boxes[i] for i in idxs]

image = detector.rectangle_detections(masked_image, idxs, boxes, confidences)

image = detector.resize(masked_image)
image = detector.put_count_text(masked_image, people_count)

cv2.imshow("Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

centers = img_liner.calc_centers(boxes_filtered)
img_liner.show_dots(img, centers)

outliers_removed = img_liner.remove_outliers(centers)
m,b = img_liner.find_best_line(outliers_removed)
img_liner.show_line_from_equation(img, m, b)

endpoints = img_liner.get_endpoints(boxes_filtered)
img_liner.show_endpoint_lines(img, m, endpoints[0], endpoints[1])

centers_in_queue =img_liner.get_centers_in_queue(centers, m, endpoints[0], endpoints[1])
img_liner.show_dots(img, centers_in_queue)

image = img_liner.put_queue_size_text(img, len(centers_in_queue))
cv2.imshow("Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
- Video bulunacak
- Endpointlerdeki bug düzeltilecek
- Video üzerinde test edilecek

"""