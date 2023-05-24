import cv2
import numpy as np
from typing import Tuple




class Liner:
    def __init__(self) -> None:
        pass

    def _calc_center(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = box
        return (x + w // 2, y + h // 2)
    
    def _calc_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calc_centers(self, boxes: list) -> list:
        return [self._calc_center(box) for box in boxes]
    

    def get_endpoints(self, boxes: list) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        top = min(boxes, key=lambda box: box[1])
        bottom = max(boxes, key=lambda box: box[1] + box[3])
        return top[1], bottom[1] + bottom[3]
    
    def show_dots(self, image: np.ndarray, centers: list, radius=500) -> None:
        black = image.copy()
        black[:] = 0
        for center in centers:
            cv2.circle(black, center, radius, (0, 255, 0), -1)
        cv2.imshow("Dots", black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def add_dots(self, image: np.ndarray, centers: list) -> np.ndarray:
        for center in centers:
            cv2.circle(image, center, 2, (0, 255, 0), -1)
        return image
    
    def remove_outliers(self, centers: list) -> list:
        distances = [self._calc_distance(centers[i], centers[i+1]) for i in range(len(centers)-1)]
        mean = np.mean(distances)
        std = np.std(distances)
        return [center for center in centers if self._calc_distance(center, centers[distances.index(max(distances))]) < mean + std]    
    
    def find_best_line(self, centers: list) -> Tuple[int, int]:
        m = 0
        b = 0
        max_inliers = 0
        best_m, best_b = 0, 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                try:
                    m = (centers[j][1] - centers[i][1]) / (centers[j][0] - centers[i][0])
                except ZeroDivisionError:
                    continue
                b = centers[i][1] - m * centers[i][0]
                inliers = 0
                for center in centers:
                    if abs(center[1] - m * center[0] - b) < 10:
                        inliers += 1
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_m = m
                    best_b = b

        return best_m, best_b

    def draw_line(self, image: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        return cv2.line(image, start_point, end_point, color, thickness)
    

    def show_line_from_equation(self, image: np.ndarray, m: int, b: int, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> None:
        black = image.copy()
        black[:] = 0
        width = image.shape[1]
        cv2.line(black, (0, int(b)), (width, int(m * width + b)), color, thickness)
        cv2.imshow("Line", black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_line(self, image: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> None:
        black = image.copy()
        black[:] = 0
        cv2.line(black, start_point, end_point, color, thickness)
        cv2.imshow("Line", black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def show_endpoint_lines(self, image: np.ndarray, slope: float, top: int, bottom: int, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> None:
        black = image.copy()
        # black[:] = 0

        width = image.shape[1]

        if slope >= 0:
            top_x1 = 0
            top_y1 = top
            top_x2 = width
            top_y2 = int(slope * width + top)

            bottom_x1 = 0
            bottom_y1 = bottom
            bottom_x2 = width
            bottom_y2 = int(slope * width + bottom)
        else:
            top_x1 = width
            top_y1 = top
            top_x2 = 0
            top_y2 = int(slope * width + top)

            bottom_x1 = width
            bottom_y1 = bottom
            bottom_x2 = 0
            bottom_y2 = int(slope * width + bottom)

        cv2.line(black, (top_x1, top_y1), (top_x2, top_y2), color, thickness)
        cv2.line(black, (bottom_x1, bottom_y1), (bottom_x2, bottom_y2), color, thickness)

        cv2.imshow("Endpoint Lines", black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def add_endpoint_lines(self, image: np.ndarray, slope: float, top: int, bottom: int, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        width = image.shape[1]

        if slope >= 0:
            top_x1 = 0
            top_y1 = top
            top_x2 = width
            top_y2 = int(slope * width + top)

            bottom_x1 = 0
            bottom_y1 = bottom
            bottom_x2 = width
            bottom_y2 = int(slope * width + bottom)
        else:
            top_x1 = width
            top_y1 = top
            top_x2 = 0
            top_y2 = int(slope * width + top)

            bottom_x1 = width
            bottom_y1 = bottom
            bottom_x2 = 0
            bottom_y2 = int(slope * width + bottom)

        image = cv2.line(image, (top_x1, top_y1), (top_x2, top_y2), color, thickness)
        image = cv2.line(image, (bottom_x1, bottom_y1), (bottom_x2, bottom_y2), color, thickness)

        return image
    

    def is_in_area(self, point: Tuple[int, int], slope: float, top: int, bottom: int) -> bool:
        x, y = point
        return y > slope * x + top and y < slope * x + bottom
    
    def get_centers_in_queue(self, centers: list, slope: float, top: int, bottom: int) -> list:
        return [center for center in centers if self.is_in_area(center, slope, top, bottom)]
    
    
    def put_queue_size_text(self, image: np.ndarray, queue_size: int) -> np.ndarray:
        cv2.putText(image, "Queue Size: " + str(queue_size), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return image

