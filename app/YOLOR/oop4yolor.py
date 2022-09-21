import os
import time

import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


class OOP4YOLOR:
    def __init__(self):
        self.device = os.getenv("yolor_device")

        self.conf_thres = float(os.getenv("yolor_conf_thres"))
        self.iou_thres = float(os.getenv("yolor_iou_thres"))
        self.imgsz = int(os.getenv("yolor_imgsz"))

        weights = [os.getenv("yolor_weights")]
        cfg = os.getenv("yolor_cfg")
        self.names = os.getenv("yolor_names")

        # Initialize
        self.device = select_device(self.device)
        # half precision only supported on CUDA
        self.half = self.device.type != 'cpu'  

        # Load model
        if self.device == 'cpu':
            self.model = Darknet(cfg, self.imgsz)
        else:
            self.model = Darknet(cfg, self.imgsz).cuda()

        self.model.load_state_dict(torch.load(weights[0], map_location=self.device)['model'])
        self.model.to(self.device).eval()
        # to FP16
        if self.half:
            self.model.half()

        # Get names
        self.names = self.load_classes(self.names)
        
        # Run inference
        # init img and run once
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device) 
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None 
    
        print("Model init done.")

    def load_classes(self, path):
        # Loads *.names file at 'path'
        # filter removes empty strings (such as last line)
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  

    def detect(self, source):
        agnostic_nms = False 
        augment = False 
        classes = None
        return_data = []
        try:
            dataset = LoadImages(source, img_size=self.imgsz, auto_size=64)
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes, agnostic=agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                        # Retrun results
                        for *xyxy, conf, cls in det:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                            # normalized xywh
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            tmp_json = {
                                "name": f"{self.names[int(cls)]}",
                                "conf": f"{conf}",
                                "xywh": f"{xywh}",
                                "c1c2": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                }
                            return_data.append(tmp_json)
                    # Print time (inference + NMS)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))
            return return_data
        except Exception as error:
            print(f"{error}")
            print("Waiting for 1 second")
            time.sleep(1)


if __name__ == '__main__':
    # Image Path
    source = f""
    with torch.no_grad():
        Model4Me = OOP4YOLOR()
        Model4Me.detect(source)

