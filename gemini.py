import PIL.Image
from google import genai
from google.genai import types

# 1️⃣ Configure with your API key
# Make sure to keep this key secret!
client = genai.Client(api_key="AIzaSyBEdgXS2l-N_G_r1-yLVB3kC-p_t6fA1KA") 

# 2️⃣ Path to your MRI image
image_path = r"E:\Brain tumor Detection\LLMServices\lime_explanation_output12.png"

# Load the image using PIL
try:
    img = PIL.Image.open(image_path)
except FileNotFoundError:
    print(f"Error: The file at {image_path} was not found.")
    exit()

# 3️⃣ Structured prompt
prompt_text = """
You are a medical imaging analysis assistant for educational and research purposes only.

Analyze the provided brain MRI image and generate a structured, non-diagnostic report. The AI prediction is glioma.

Include:
1. Image overview
2. Observed imaging features
3. Possible associated symptoms (hypothetical and non-confirmatory)
4. General precautions and recommended next steps

Rules:
- Do NOT diagnose diseases
- Use cautious language such as "may be associated with"
- Avoid definitive clinical conclusions
- This output is NOT medical advice
"""

# 4️⃣ Send request to Gemini
# We use gemini-1.5-flash or gemini-1.5-pro
response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[prompt_text, img]
)

# 5️⃣ Print the report
print("==== GENERATED REPORT ====\n")
print(response.text)
print("\n==========================")