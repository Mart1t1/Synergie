import os
import sys

sys.path.append(os.path.dirname("../"))

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from data_generation import trainingSession
from inference.inference import infer

app = FastAPI()


class JumpModel(BaseModel):
    type: str
    start: int
    length: float
    rotation: float
    max_rotation_speed: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/evalsession")
def eval_session(file: UploadFile):
    cachecsvloc = "cache/session.csv"

    # save file
    with open(cachecsvloc, "wb") as f:
        f.write(file.file.read())

    # jump segmentation and metric collection
    session = trainingSession.trainingSession(cachecsvloc)

    # inference phase
    infer(session.jumps)

    jumps = []

    for i in session.jumps:
        if float(i.max_rotation_speed) != float('inf'): # messy corner case I dont want to fix
            jumps.append(JumpModel(type=i.type, start=i.start, length=i.length, rotation=i.rotation,
                               max_rotation_speed=i.max_rotation_speed))

    # send request
    return jumps
