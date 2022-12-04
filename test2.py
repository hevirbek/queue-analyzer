import os
import cv2
from people_detector import PeopleDetector
from image_polluter import ImagePolluter
import matplotlib.pyplot as plt

image_path = "images/test2.jpg"


weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])


detector = PeopleDetector(
    weightsPath=weightsPath, configPath=configPath)

polluter = ImagePolluter()

image = cv2.imread(image_path)
image_darkened = polluter.darken(image, 0.5)
image_lightened = polluter.lighten(image, 0.5)
image_blurred = polluter.blur(image, 2)
image_noised = polluter.add_noise(image, 1.2)
image_salt_pepper = polluter.add_salt_pepper_noise(image, 10)

list_of_images = [image, image_darkened, image_lightened,
                  image_blurred, image_noised, image_salt_pepper]

list_of_labels = ["Original", "Darkened", "Lightened",
                  "Blurred", "Noised", "Salt and Pepper"]


fig1 = plt.figure(figsize=(6, 10))

for i, im in enumerate(list_of_images, start=1):
    fig1.add_subplot(3, 2, i)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(list_of_labels[i-1])

fig2 = plt.figure(figsize=(6, 10))

for i, im in enumerate(list_of_images, start=1):
    idxs, boxes, confidences = detector.detect(im)
    im = detector.rectangle_detections(im, idxs, boxes, confidences)
    im = detector.resize(im)
    im = detector.put_count_text(im, len(idxs))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig2.add_subplot(3, 2, i)
    plt.imshow(im)
    plt.title(list_of_labels[i-1])

plt.show()
