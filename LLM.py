import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class LLM_prediction:
    def __init__(self):
        self.MODEL_ID = "llava-hf/llava-1.5-7b-hf"

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DTYPE = torch.float16 if self.DEVICE.type == "cuda" else torch.float32

        print("Loading model... This may take a few minutes.")

        self.processor = LlavaProcessor.from_pretrained(self.MODEL_ID)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.DTYPE,
            device_map="auto"
        )

        self.model.eval()

    def generate_report(self, image: Image.Image, prediction: str):
        prompt = (
            "<image>\n\n"
            "You are a medical imaging analysis assistant for educational and research purposes only.\n\n"
            f"Analyze the provided brain MRI image and generate a structured, "
            f"non-diagnostic report. The AI prediction is: {prediction}.\n\n"
            "Include:\n"
            "1. Image overview\n"
            "2. Observed imaging features\n"
            "3. Possible associated symptoms (hypothetical and non-confirmatory)\n"
            "4. General precautions and recommended next steps\n\n"
            "Rules:\n"
            "-  diagnose diseases based on prediction \n"
            "- Use cautious language such as \"may be associated with\"\n"
            "- Avoid definitive clinical conclusions\n"
            "- This output is NOT medical advice and in one long paragraph\n"
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True
            )

        input_len = inputs["input_ids"].shape[1]
        report = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        ).strip()

        return report
