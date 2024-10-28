import cv2
import torch
from PIL import Image
import time
import numpy as np
class yolov5(object):
    def __init__(self, path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

    def imageDetectorfromFrame(self, frame, floatorint=True):
        # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        if floatorint:
            # frame = np.floor(frame * 255)
            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        imgs = [frame]
        # Inference
        # t1 = time.time()
        results = self.model(imgs)  # includes NMS
        # t2 = time.time()
        # print("detector time:" + str(t2-t1))
        results.render()
        # return results.imgs[0]
        return results.xyxy[0].cpu().data.numpy(), results.imgs[0]


    def imageDetectorfromFile(self, filename='test.jpg'):
        # img1 = Image.open('test.jpg')  # PIL image
        img2 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)  # OpenCV image (BGR to RGB)
        print(img2)
        imgs = [img2]  # batch of images

        # Inference
        results = self.model(imgs)  # includes NMS

        # Results
        # results.print()
        # results.show()
        # results.imgs
        results.render()
        # for img in results.imgs:
        # img_base64 = Image.fromarray(results.imgs[0])
        #
        # frame = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)
        # img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        # print(img_base64.getdata())
        # img_base64.show()
            # cv2.imshow("detecte", img_base64)
            # time.sleep(5)
        # results.save()  # or .show()

        # Data
        print(results.xyxy[0])  # print img1 predictions (pixels)
        #                   x1           y1           x2           y2   confidence        class
        # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
        #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
        #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
        print(results.imgs[0])
        return results.imgs[0]

if __name__ == '__main__':
    yolov5 = yolov5(path="best.pt")
    image = yolov5.imageDetectorfromFile(filename='./images/0-7.jpg')
    image = Image.fromarray(image)
    data = image.getdata()
    print(data)
    image.show()
    image.save("./0-7-.jpg")
