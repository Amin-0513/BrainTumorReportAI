import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from LLM import LLM_prediction

app = FastAPI(title="MRI LLM Report API")

llm_model = LLM_prediction()


class PredictionRequest(BaseModel):
    image_base64: str
    prediction: str


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


@app.post("/generate-report")
def generate_report(request: PredictionRequest):
    image = decode_base64_image(request.image_base64)

    report = llm_model.generate_report(
        image=image,
        prediction=request.prediction
    )

    return {
        "prediction": request.prediction,
        "report": report
    }
