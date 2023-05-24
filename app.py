from flask import Flask, render_template, Response
import cv2
import os
from people_detector import PeopleDetector
from masker import Masker

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
detector = PeopleDetector(weightsPath=weightsPath, configPath=configPath)

app = Flask(__name__)

def process_video():
    while True:
        video_path = "ituvideos/22.avi" 
        video = cv2.VideoCapture(video_path)    
        
        while True:
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

            image = detector.rectangle_detections(resized_frame_for_show, idxs, boxes, confidences)

            ret, buffer = cv2.imencode('.jpg', image)
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