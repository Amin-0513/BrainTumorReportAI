from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

class LLM_prediction_auto_symptoms:
    def __init__(self, hf_api_key=None):
        self.MODEL_ID = "google/t5gemma-2-270m-270m"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32

        print(f"Loading model on {self.DEVICE}...")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, use_auth_token=hf_api_key
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.DTYPE,
            low_cpu_mem_usage=True,
            use_auth_token=hf_api_key
        ).to(self.DEVICE)
        self.model.eval()

    def generate_report(self, image_path: str, prediction: str):
        # Load local image
        image = Image.open(image_path).convert("RGB")

        # Prompt instructing the model to infer symptoms from prediction and observations
        prompt = (
            f"<start_of_image> Analyze the provided brain MRI image.\n"
            f"AI prediction: {prediction}.\n"
            "Based on the AI prediction and observed imaging features, "
            "describe possible associated symptoms (hypothetical, educational only).\n"
            "Provide a structured, non-diagnostic report in one paragraph, "
            "including image overview, observed imaging features, inferred symptoms, "
            "and general precautions."
        )

        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.DEVICE, self.DTYPE)

        # Generate report
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        report = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return report


if __name__ == "__main__":
    HF_API_KEY = "your_huggingface_api_key_here"
    llm_service = LLM_prediction_auto_symptoms(hf_api_key=HF_API_KEY)

    # Path to your local image
    local_image_path = r"E:\Brain tumor Detection\LLMServices\lime_explanation_output12.png"

    test_prediction = "glioma"

    report = llm_service.generate_report(local_image_path, test_prediction)
    print(report)
