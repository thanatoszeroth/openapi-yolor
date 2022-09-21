import os
import sys
import cv2
import time
import base64
import shutil
import numpy as np
from typing import List
from fastapi import APIRouter
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from starlette.responses import FileResponse 

print(f"Python Main Path : {os.path.abspath(os.path.join(os.getcwd()))}")
print(f"Current Path : {os.path.abspath(os.path.dirname(__file__))}")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print(f"Upper Path : {os.path.abspath(os.path.join(os.getcwd(), '..'))}")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import torch
from oop4yolor import OOP4YOLOR

class Base64Image(BaseModel):
    base64_image: str

router = APIRouter()
yolor = OOP4YOLOR()

def plot_one_box(c1, c2, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

@router.post("/api4yolor/detect_image", tags = ["YOLOR Detection"])
async def detect_image(file: UploadFile = File(...)):
    if f"{file.filename[-4:]}" == ".jpg" or f"{file.filename[-4:]}" == ".png":
        with open(f"static/tmp/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with torch.no_grad():
            result = yolor.detect(f"static/tmp/{file.filename}")
        try:
            os.remove(f"static/tmp/{file.filename}")
        except OSError as e:
            print(e)
        return {
            "message": "Success",
            "file_name": file.filename,
            "detect_result": result}
    else:
        return {
            "message": "Type Error",
            "file_name": file.filename,
            "detect_result": "None"}


@router.post("/api4yolor/detect_image_save", tags = ["YOLOR Detection"])
async def detect_image_save(file: UploadFile = File(...)):
    if f"{file.filename[-4:]}" == ".jpg" or f"{file.filename[-4:]}" == ".png":
        with open(f"static/tmp/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with torch.no_grad():
            result = yolor.detect(f"static/tmp/{file.filename}")
        if result != []:
            im0 = cv2.imread(f"static/tmp/{file.filename}")
            for Nid in range(len(result)):
                plot_one_box(
                    (result[Nid]['c1c2'][0], result[Nid]['c1c2'][1]), 
                    (result[Nid]['c1c2'][2], result[Nid]['c1c2'][3]), 
                    im0, 
                    label = f"{result[Nid]['name']} {result[Nid]['conf']}", 
                    color = [0, 0, 255], 
                    line_thickness = 2
                    )
            cv2.imwrite(f"static/tmp/{time.strftime('%Y%m%d%H%M%S', time.localtime())}.jpg", im0)
        return FileResponse(f"static/tmp/{time.strftime('%Y%m%d%H%M%S', time.localtime())}.jpg")
    else:
        return {
            "message": "Type Error",
            "file_name": file.filename,
            "detect_result": "None"}

@router.post("/api4yolor/detect_base64_image", tags = ["YOLOR Detection"])
async def detect_base64_image(request: Base64Image):
    SaveFileName = f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}.jpg"
    try:
        ImageArray = base64.b64decode(request.base64_image)
        ImageFile = np.fromstring(ImageArray, np.uint8) 
        img = cv2.imdecode(ImageFile, cv2.COLOR_BGR2RGB)  
        cv2.imwrite(f"static/tmp/{SaveFileName}", img)
    except Exception as e:
        print(f"{e}")
        return {"message": f"String is not base64 image"}

    with torch.no_grad():
        result = yolor.detect(f"static/tmp/{SaveFileName}")
    try:
        os.remove(f"static/tmp/{SaveFileName}")
    except OSError as e:
        print(e)
    return {
        "message": "Success",
        "file_name": f"{SaveFileName}",
        "detect_result": result}

@router.post("/api4yolor/detect_base64_image_save", tags = ["YOLOR Detection"])
async def detect_base64_image_save(request: Base64Image):
    SaveFileName = f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}.jpg"
    try:
        ImageArray = base64.b64decode(request.base64_image)
        ImageFile = np.fromstring(ImageArray, np.uint8) 
        img = cv2.imdecode(ImageFile, cv2.COLOR_BGR2RGB)  
        cv2.imwrite(f"static/tmp/{SaveFileName}", img)
    except Exception as e:
        print(f"{e}")
        return {"message": f"String is not base64 image"}

    with torch.no_grad():
        result = yolor.detect(f"static/tmp/{SaveFileName}")
    return {
        "message": "Success",
        "file_name": f"{SaveFileName}",
        "detect_result": result}

@router.post("/api4yolor/clear_tmp_file", tags = ["YOLOR Detection"])
async def clear_tmp_file():
    try:
        shutil.rmtree('static/tmp/')
        os.makedirs(f"static/tmp")
        return {"message": "Success",}
    except Exception as error:
        print(f"{error}")
        return {"message": "Error",}