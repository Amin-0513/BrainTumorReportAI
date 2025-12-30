import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


class LLM_prediction_small:
    def __init__(self):
        self.MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32

        print(f"Loading model on {self.DEVICE}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        # Load model (NO quantization)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.DTYPE,
            low_cpu_mem_usage=True,
        ).to(self.DEVICE)

        self.model.eval()

    def generate_report(self, image: Image.Image, prediction: str):
        # Chat-style prompt REQUIRED for OneVision models
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical imaging analysis assistant for educational "
                            "and research purposes only.\n\n"
                            f"The AI prediction is: {prediction}.\n\n"
                            "Analyze the provided brain MRI image and generate a structured, "
                            "non-diagnostic report.\n\n"
                            "Include:\n"
                            "1. Image overview\n"
                            "2. Observed imaging features\n"
                            "3. Possible associated symptoms (hypothetical)\n"
                            "4. General precautions and next steps\n\n"
                            "Rules:\n"
                            "- Do NOT diagnose diseases\n"
                            "- Use cautious language\n"
                            "- Avoid definitive clinical conclusions\n"
                            "- This is NOT medical advice\n"
                            "- Write in one paragraph"
                        ),
                    },
                    {"type": "image"},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.DEVICE, self.DTYPE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Skip special tokens correctly
        report = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return report


if __name__ == "__main__":
    llm_service = LLM_prediction_small()

    test_image = Image.open(
        r"E:\Brain tumor Detection\LLMServices\lime_explanation_output12.png"
    ).convert("RGB")

    test_prediction = "glioma"

    report = llm_service.generate_report(test_image, test_prediction)
    print(report)
