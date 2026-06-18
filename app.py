from fastapi import FastAPI, File,UploadFile, HTTPException,Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.sankey import UP
from pydantic import BaseModel 
from ultralytics import YOLO
import cv2 
import numpy as np

import base64
from pathlib import Path

BASE_DIR =Path(__file__).resolve().parent

app = FastAPI(title = 'Facial Emotion Detection system')
app.mount("/static",StaticFiles(directory=str(BASE_DIR / "static")),name = 'static')

template= Jinja2Templates(directory=str(BASE_DIR / "templates"))


#LOAD YOLO MODEL
model = YOLO(str(BASE_DIR / "best.pt"))


class WebcamPlayload(BaseModel):
    image:str




def run_detection(cv_image: np.ndarray):
    # Run inference
    results = model.predict(
        source=cv_image,
        conf=0.34,
        imgsz=640,
        verbose=False
    )

    if not results:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        )

    r = results[0]

    # Extract detections
    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy
        confs = r.boxes.conf
        classes = r.boxes.cls

        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy() # pyright: ignore[reportAttributeAccessIssue]

        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()# pyright: ignore[reportAttributeAccessIssue]

        if hasattr(classes, "cpu"):
            classes = classes.cpu().numpy()# pyright: ignore[reportAttributeAccessIssue]

        classes = classes.astype(int)# pyright: ignore[reportAttributeAccessIssue]






        for (x1, y1, x2, y2), conf, cls in zip(
            xyxy, confs, classes
        ):
            detections.append({
                "label": model.names.get(cls, str(cls)),
                "confidence": float(conf),
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })

    # Draw bounding boxes
    annotated = r.plot()

    if annotated is None:
        annotated = cv_image

    # Convert image to base64
    success, buffer = cv2.imencode(".jpg", annotated)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode image."
        )

    image_base64 = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return {
        "detections": detections,
        "image_data": f"data:image/jpeg;base64,{image_base64}"
    }


@app.get("/",response_class=HTMLResponse)
async def index(request:Request):# url, method,header -> it's a http request object sent by the browser
    return template.TemplateResponse(name='index.html',request=request)


@app.post("/detect-upload")
async def detect_upload(file: UploadFile = File(...)):

    filename=file.filename or ""
    if not filename.lower().endswith((".jpg",".png",'.jpeg',".bmp")):
        raise HTTPException(status_code=400,detail="Only JPG, JPEG and PNG files are allowed")

    contents = await file.read()  #contents is raw bytes
    nparr = np.frombuffer(contents, np.uint8) #Convert bytes → NumPy array->Unsigned 8-bit integer)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #image decode
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return run_detection(img)


@app.post("/detect-webcam")
async def detect_webcam(payload: WebcamPlayload):
    try:
        data = payload.image
        if "," in data:
            data = data.split(",")[1]
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid webcam image.")
        return run_detection(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process webcam image: {str(e)}")



if __name__ == "__main__":
    import uvicorn 
    host = "0.0.0.0"
    port=8080
    print(f"Starting Facial Emotion Detection app on http://{host}:{port}")
    uvicorn.run("app:app",host=host,port=port,reload=True)
