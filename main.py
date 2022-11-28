from fastapi import FastAPI, File
import mediapipe as mp
import cv2
from character.detect_shape import face_classifi
from character.skincolor import skin_detect
from analyze.relationship import user_relationship
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json

now = datetime.now()

class Item(BaseModel):
    id : str
    time : str
    x : str
    z : str
    
class coorperates(BaseModel):
    coorperate : Optional[List[Item]] = None

app = FastAPI()

@app.get("/")
def hello():
    return{"Hello":"World"}

@app.post("/facedata")
async def face_analysis(file: bytes= File(...)):
    image = np.array(Image.open(io.BytesIO(file)))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    fshape = face_classifi(image)
    sColor = skin_detect(image)
    print("얼굴 데이터 완료 ",now)
    return {"face shape" : fshape,
            "skin color" : sColor}
    
@app.post("/user-analysis")
async def user_correlation(item:coorperates):
    result = json.dumps(user_relationship(item))
    result = json.loads(result)
    return result