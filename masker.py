import numpy as np
import cv2
import os

class Masker:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.areas = []
        self.roi = []
        self.roi_area = 0

    def select_roi(self):
        self.roi = cv2.selectROI("Select ROI", self.image, fromCenter=False, showCrosshair=True)
        self.areas.append(self.roi)

    def select_multiple_roi(self):
        while True:
            mask = self.prepare_mask()
            masked_image = self.apply_mask(mask)
            self.image = masked_image
            self.select_roi()
            if input("Select more roi? (y/n): ") != "y":
                break

    def prepare_mask(self):
        white = np.ones(self.image.shape, dtype=np.uint8) * 255
        # make roi areas black
        for area in self.areas:
            x, y, w, h = area
            white[y:y + h, x:x + w] = 0
        return white
    
    def apply_mask(self, white):
        return cv2.bitwise_and(self.image, white)
    

    def save_mask(self,mask,path):
        cv2.imwrite(path,mask)
        print("Mask saved to {}".format(path))

    
    def load_mask(self,path):
        mask = cv2.imread(path)
        return mask
    


if __name__ == "__main__":
    image_path = "images/test2.jpg"
    mask_file_path = "masks/" + image_path.split("/")[-1].split(".")[0] + "_mask.png"

    img = cv2.imread(image_path)
    img_masker = Masker(img)

    if os.path.isfile(mask_file_path):
        print("Mask file found. Loading mask from {}".format(mask_file_path))
        
        white = img_masker.load_mask(mask_file_path)
        masked_image = img_masker.apply_mask(white)

        cv2.imshow("Frame", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Mask file not found. Creating mask...")

        img_masker.select_multiple_roi()

        cv2.destroyAllWindows()

        white = img_masker.prepare_mask()
        mask_file_path = "masks/" + image_path.split("/")[-1].split(".")[0] + "_mask.png"
        img_masker.save_mask(white,mask_file_path)
        masked_image = img_masker.apply_mask(white)

        cv2.imshow("Frame", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

        

        
        
        
    