import numpy as np
import cv2
import os

from typing import Tuple


LABELS_PATH = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(LABELS_PATH).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
PERSON_ID = LABELS.index("person")


class PeopleDetector:
    def __init__(self, weightsPath: str, configPath: str, conf: float = 0.5, thresh: float = 0.3):
        self.conf = conf
        self.thresh = thresh
        self.weightsPath = weightsPath
        self.configPath = configPath
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, list, list]:
        (H, W) = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        boxes = []
        confidences = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.conf and classID == PERSON_ID:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf, self.thresh)
        return idxs, boxes, confidences

    def rectangle_detections(self, image, idxs, boxes, confidences):
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[PERSON_ID]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(
                    LABELS[PERSON_ID], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        return image

    def put_count_text(self, image, count):
        cv2.putText(image, "People Count: " + str(count), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        (H, W) = image.shape[:2]
        if W > 1000:
            image = cv2.resize(image, (1000, int(H * 1000 / W)))
        elif H > 1000:
            image = cv2.resize(image, (int(W * 1000 / H), 1000))
        return image
