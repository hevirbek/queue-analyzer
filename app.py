from flask import Flask, render_template, Response
import cv2
import os
from people_detector import PeopleDetector
from masker import Masker
from liner import Liner
import time

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)

app = Flask(__name__)

def process_video():
    while True:
        video_path = "ituvideos/22.avi" 
        video = cv2.VideoCapture(video_path)    
        
        while True:
            start = time.time()
            success, frame = video.read()
            if not success:
                break
            
            mask_file_path = "masks/mask.png"
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

            pt1 = (650, 80)
            pt2 = (800, 80)

            # number of people under the line
            n = len([center for center in centers if center[1] > pt1[1] and center[1] > pt2[1]])

            # number of people in the queue
            m = len(centers)

            l1 = (300,60)
            l2 = (750,80)
            l3 = (950,400)

            total_distance = img_liner._calc_distance(l1, l2) + img_liner._calc_distance(l2, l3)
            L = total_distance / 1000

            # time to pass the line
            current_time = time.time()
            t = current_time - start
            print(n,t)

            # calculate the flow 
            flow = n / t
            density = m / L
            speed = flow / density

            show = cv2.line(show, l1, l2, (255, 0, 255), 1)
            show = cv2.line(show, l2, l3, (255, 0, 255), 1)

            # draw line with cv2.line
            show = cv2.line(show, pt1, pt2, (0, 255, 255), 1)
            # image = detector.rectangle_detections(resized_frame_for_show, idxs, boxes, confidences)

            # put text on the image for flow and density
            show = cv2.putText(show, "Flow: {:.2f}".format(flow), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            show = cv2.putText(show, "Density: {:.2f}".format(density), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            show = cv2.putText(show, "Speed: {:.2f}".format(speed), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', show)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        video.release()
        cv2.destroyAllWindows()
                   

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)