# BrainTumorReportAI

<p align="center">
  <img src="images/8.png" alt="Brain Tumor Report AI Banner" width="900"/>
</p>


"This repository contains a production-ready deep learning system designed for automated analysis of brain MRI and CT scans. It integrates convolutional neural networks (CNN) for high-accuracy tumor detection with state-of-the-art explainable AI (XAI) techniques to provide transparent and interpretable predictions. The system not only classifies and localizes tumors but also generates detailed, clinically-oriented reports, assisting radiologists and medical professionals in making informed decisions. With a focus on reliability, interpretability, and real-world applicability, this framework bridges the gap between advanced AI models and practical clinical deployment for brain tumor diagnosis and monitoring."


### First Step

[![Frontend Repository](https://img.shields.io/badge/Front%20End-Repository-blue?logo=github)](https://github.com/Amin-0513/brain-tumor-frontend)




```bash
# Clone the repository
git clone https://github.com/Amin-0513/BrainTumorReportAI.git

# Navigate to project directory
cd BrainTumorReportAI

# create python environment
python -m venv llmservice

# activate python environment
llmservice\Scripts\activate

# Install dependencies
pip install -r requirments.txt

zenml init

## Start project
uvicorn llmserviceapi:app --host 0.0.0.0 --port 5000 --reload

```



## User Interface
<table>
   <tr>
    <td><img src="images/5.jpeg" width="400"/></td>
    <td><img src="images/6.jpeg" width="400"/></td>
  </tr>
  
</table>
