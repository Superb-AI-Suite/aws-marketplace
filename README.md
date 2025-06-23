# ZERO - AWS Marketplace Examples

<p align="center">
  <img src="https://asset.superb-ai.com/assets/logo/CI_superbAI_RGB_basic.png" alt="Superb AI Logo" width="300"/>
</p>

<p align="center">
  <strong>An Industrial Vision Foundation Model</strong>
  <br/><br/>
  <a href="https://aws.amazon.com/marketplace/pp/prodview-xxxxxxxxxxxxxxx">
    <img src="https://img.shields.io/badge/AWS%20Marketplace-View%20Listing-orange?style=for-the-badge&logo=amazonaws" alt="AWS Marketplace">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

This repository provides an example Jupyter Notebook on how to use the **ZERO** model package on Amazon SageMaker. **ZERO** is an industrial Vision Foundation Model (VFM) ready for immediate deployment without the need for data labeling or model retraining. It leverages zero-shot and open-world technologies to instantly detect and pinpoint new or unseen objects using simple text or image box prompts.

---

## 📖 Table of Contents
* [About the Model](#-about-the-model)
* [Key Features & Highlights](#-key-features--highlights)
* [SageMaker Deployment & Usage](#-sagemaker-deployment--usage)
  * [Step 1: Subscribe to the Model](#step-1-subscribe-to-the-model)
  * [Step 2: Set Up Environment](#step-2-set-up-environment)
  * [Step 3: Deploy the Model to an Endpoint](#step-3-deploy-the-model-to-an-endpoint)
  * [Step 4: Perform Real-time Inference](#step-4-perform-real-time-inference)
  * [Step 5: Clean Up Resources](#step-5-clean-up-resources)
* [Input/Output Interface](#-inputoutput-interface)
* [Technical Specifications](#-technical-specifications)
* [License](#-license)
* [Support](#-support)

---

## 🧠 About the Model

Traditional Vision AI demands extensive data labeling and repetitive model retraining—a process that consumes significant time, cost, and specialized expertise. Superb AI's **ZERO** brings a paradigm shift as an industrial-specialized Vision Foundation Model (VFM).

Leveraging **Open World Visual Grounding** technology, ZERO comprehends novel concepts without prior training. This zero-shot capability empowers instant AI adoption for new tasks and flexible, on-the-fly changes to detection targets, eliminating the need for additional training. Instead of time-consuming retraining, you simply describe your target in text (e.g., *"a dent on the car door"*) or provide an example image box, and ZERO adapts instantly. This dramatically cuts the time and cost of AI solution development, making AI adoption faster and more accessible.

This repository and the accompanying Jupyter Notebook (`/zero/ZERO-Marketplace.ipynb`) demonstrate how to subscribe, deploy, and run inference with ZERO on Amazon SageMaker.

## ✨ Key Features & Highlights

* **🚀 Zero-Shot Deployment:** Instantly detect untrained objects without complex data collection, labeling, or model retraining. Adapt immediately to new products, defect types, or environment changes, dramatically cutting development time and costs.

* **✍️ Flexible Multi-Prompt Input:** Deploy and operate AI instantly by simply describing your target object in text or providing an example image box. ZERO supports diverse input prompts for intuitive, human-like interaction.

* **🏭 Industrial-Specialized VFM:** Trained on invaluable, real-world data from dozens of industrial sectors including manufacturing, logistics, and retail. ZERO delivers high performance and immediate usability across complex industrial domains.

* **💻 Edge & On-Premise Ready:** Engineered for high efficiency with a lightweight 622M parameters and 1.03 TFLOPS processing, ensuring seamless operation on both cloud infrastructure and resource-constrained edge devices without requiring expensive GPU hardware.

## 🚀 SageMaker Deployment & Usage

This section provides a step-by-step guide to deploying ZERO and running inference, based on the provided Jupyter Notebook.

### Step 1: Subscribe to the Model

Before you can use the model, you need to subscribe to it on the AWS Marketplace.

1.  Navigate to the [ZERO Model Listing on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-xxxxxxxxxxxxxxx).
2.  Click the **Continue to Subscribe** button.
3.  Review the terms and conditions, then click **"Accept Terms"**.
4.  Once the subscription is active (this may take a few minutes), you can proceed. You will find the **Model Package ARN** under the "Usage Information" section.

### Step 2: Set Up Environment

Open the `zero/ZERO-Marketplace.ipynb` notebook in a SageMaker Notebook Instance. The notebook will guide you through setting up the required IAM role and initializing the boto3 and SageMaker clients.

```python
import sagemaker
from sagemaker import ModelPackage
import boto3

# Get the SageMaker execution role
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()

# Your specific Model Package ARN from AWS Marketplace
model_package_arn = '<YOUR_MODEL_PACKAGE_ARN>'
````

*Note: Replace `<YOUR_MODEL_PACKAGE_ARN>` with the ARN you obtained after subscribing.*

### Step 3: Deploy the Model to an Endpoint

Create a deployable SageMaker model from the model package and deploy it to a real-time endpoint. The notebook specifies the recommended instance type.

```python
# Create a SageMaker Model from the Model Package
model = ModelPackage(
    role=role,
    model_package_arn=model_package_arn,
    sagemaker_session=sagemaker_session
)

# Deploy the model to an endpoint
# Recommended instance type is ml.g4dn.xlarge
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='zero-endpoint' # You can choose a different name
)
```

### Step 4: Perform Real-time Inference

Once the endpoint is `InService`, you can send it image data and prompts to get predictions.

The payload should be a JSON object containing the `image` (Base64 encoded) and a `prompt`.

**Example: Using a Text Prompt**

```python
import base64
import json

# 1. Load your image and encode it
with open("your_image.jpg", "rb") as f:
    image_bytes = f.read()
base64_image = base64.b64encode(image_bytes).decode('utf-8')

# 2. Construct the payload with a text prompt
payload = {
    "image": base64_image,
    "prompt": {
        "texts": ["a person wearing a helmet", "a person not wearing a helmet"]
    }
}

# 3. Get prediction
response = predictor.predict(json.dumps(payload))
result = json.loads(response)

# The 'result' will contain bounding boxes and scores for the detected objects.
```

**Example: Using a Box (Image) Prompt**

```python
# (Image loading is the same as above)

# 2. Construct the payload with a box prompt
# The box coordinates are [x_min, y_min, x_max, y_max] in relative format.
payload = {
    "image": base64_image,
    "prompt": {
        "boxes": [[1, 2, 42, 51]]
    }
}

# 3. Get prediction
response = predictor.predict(json.dumps(payload))
result = json.loads(response)
```

### Step 5: Clean Up Resources

To avoid incurring ongoing charges, it is important to delete the SageMaker endpoint when you are finished.

```python
# Delete the endpoint
predictor.delete_endpoint()

# Delete the model configuration
predictor.delete_model()
```

## 📋 Input/Output Interface

### Input Payload (`application/json`)

The model expects a JSON object with the following structure:

```json
{
  "image": "<base64_encoded_string>",
  "prompt": {
    "texts": ["string_prompt_1", "string_prompt_2", ...],
    "boxes": [
        [x_min, y_min, x_max, y_max], 
        ...
    ]
  }
}
```

  * `image`: The raw image file encoded as a Base64 string.
  * `prompt`: A dictionary containing one or both of the following keys:
      * `texts`: A list of strings describing the objects to detect.
      * `boxes`: A list of bounding boxes (each as a list of four floats) defining regions of interest. Coordinates must be relative to image size (from 0.0 to 1.0).

### Output Response (`application/json`)

The model returns a JSON object containing the prediction results:

```json
{
    "prediction": [
        {
            "box": [x_min, y_min, x_max, y_max],
            "score": 0.95,
            "label": "string_prompt_1"
        },
        ...
    ]
}
```

  * `prediction`: A list of detected objects, where each object is a dictionary with:
      * `box`: The coordinates of the predicted bounding box.
      * `score`: The model's confidence score for the prediction (0.0 to 1.0).
      * `label`: The prompt that corresponds to this detection.

## 🛠️ Technical Specifications

  * **Model Size:** 622M parameters
  * **Performance:** 1.03 TFLOPS
  * **Recommended Instance Type:** `ml.g4dn.xlarge` or other GPU instances.


## ❓ Support

For questions, issues, or support regarding the ZERO model or this sample notebook, please open an issue in this GitHub repository.

For business inquiries or questions about Superb AI's other offerings, please contact us at [contact@superb-ai.com](mailto:contact@superb-ai.com).
