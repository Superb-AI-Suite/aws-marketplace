<p align="center">
  <img src="https://asset.superb-ai.com/assets/logo/CI_superbAI_RGB_basic.png" alt="Superb AI Logo" width="300"/>
</p>

<p align="center">
  <strong>An Industrial Vision Foundation Model</strong>
  <br/><br/>
  <a href="https://aws.amazon.com/marketplace/pp/prodview-xm2s45njvoiqk">
    <img src="https://img.shields.io/badge/AWS%20Marketplace-View%20Listing-orange?style=for-the-badge&logo=amazonaws" alt="AWS Marketplace">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

This repository provides an example Jupyter Notebook on how to use the **ZERO** model package on Amazon SageMaker. **ZERO** is an industrial Vision Foundation Model (VFM) ready for immediate deployment without the need for data labeling or model retraining. It leverages zero-shot and open-world technologies to instantly detect and pinpoint new or unseen objects using simple text or image box prompts.

## 📖 Table of Contents

- [About the Model](https://www.google.com/search?q=%23-about-the-model)
- [Key Features & Highlights](https://www.google.com/search?q=%23-key-features--highlights)
- [SageMaker Deployment & Usage](https://www.google.com/search?q=%23-sagemaker-deployment--usage) *[Step 1: Subscribe and Deploy the Model](https://www.google.com/search?q=%23step-1-subscribe-and-deploy-the-model)* [Step 2: Set Up Environment for Inference](https://www.google.com/search?q=%23step-2-set-up-environment-for-inference) *[Step 3: Perform Real-time Inference](https://www.google.com/search?q=%23step-3-perform-real-time-inference)* [Step 4: Clean Up Resources](https://www.google.com/search?q=%23step-4-clean-up-resources)
- [Input/Output Interface](https://www.google.com/search?q=%23-inputoutput-interface)
- [Technical Specifications](https://www.google.com/search?q=%23-technical-specifications)
- [License](https://www.google.com/search?q=%23-license)
- [Support](https://www.google.com/search?q=%23-support)

## 🧠 About the Model

Traditional Vision AI demands extensive data labeling and repetitive model retraining—a process that consumes significant time, cost, and specialized expertise. Superb AI's **ZERO** brings a paradigm shift as an industrial-specialized Vision Foundation Model (VFM).

Leveraging **Open World Visual Grounding** technology, ZERO comprehends novel concepts without prior training. This zero-shot capability empowers instant AI adoption for new tasks and flexible, on-the-fly changes to detection targets, eliminating the need for additional training. Instead of time-consuming retraining, you simply describe your target in text (e.g., *"a dent on the car door"*) or provide an example image box, and ZERO adapts instantly. This dramatically cuts the time and cost of AI solution development, making AI adoption faster and more accessible.

This repository and the accompanying Jupyter Notebook (`/zero/ZERO-Marketplace.ipynb`) demonstrate how to subscribe, deploy, and run inference with ZERO on Amazon SageMaker.

## ✨ Key Features & Highlights

- **🚀 Zero-Shot Deployment:** Instantly detect untrained objects without complex data collection, labeling, or model retraining. Adapt immediately to new products, defect types, or environment changes, dramatically cutting development time and costs.
- **✍️ Flexible Multi-Prompt Input:** Deploy and operate AI instantly by simply describing your target object in text or providing an example image box. ZERO supports diverse input prompts for intuitive, human-like interaction.
- **🏭 Industrial-Specialized VFM:** Trained on invaluable, real-world data from dozens of industrial sectors including manufacturing, logistics, and retail. ZERO delivers high performance and immediate usability across complex industrial domains.
- **💻 Edge & On-Premise Ready:** Engineered for high efficiency with a lightweight 622M parameters and 1.03 TFLOPS processing, ensuring seamless operation on both cloud infrastructure and resource-constrained edge devices without requiring expensive GPU hardware.

## 🚀 SageMaker Deployment & Usage

This section provides a step-by-step guide to deploying ZERO and running inference, based on the provided Jupyter Notebook.

### Step 1: Subscribe and Deploy the Model

Before you can use the model, you need to subscribe to it on the AWS Marketplace and deploy it to a SageMaker endpoint.

1. Navigate to the [ZERO Model Listing on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-xm2s45njvoiqk).
2. Click the **Continue to Subscribe** button.
3. Review the terms and conditions, then click **"Accept Terms"**.
4. Once the subscription is active, follow the deployment instructions in the "Usage Information" tab on the Marketplace listing to create a SageMaker endpoint. The recommended instance type is `ml.g4dn.xlarge`.
5. Make a note of your endpoint's name.

### Step 2: Set Up Environment for Inference

Configure your environment with your AWS credentials and initialize the Boto3 client to interact with SageMaker.

```
import boto3
import os

# Configure your AWS credentials and region
# It's recommended to use environment variables for security
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_REGION'] = 'your-aws-region' # e.g., 'ap-northeast-2'

# Initialize Boto3 session and SageMaker runtime client
boto_session = boto3.Session(region_name=os.environ['AWS_REGION'])
sm_runtime = boto_session.client("sagemaker-runtime")

# The name of the endpoint you deployed in Step 1
zero_endpoint_name = "zero-marketplace" # Or your custom endpoint name

```

### Step 3: Perform Real-time Inference

Once the endpoint is `InService`, you can send it image data and prompts to get predictions. The payload should be a JSON object containing the `search_image` and a list of `queries`.

**Example 1: Using a Text Prompt**

Here, we find all instances of "strawberry" in an image.

```
import json
import base64
from io import BytesIO
import requests
from PIL import Image

# Helper function to encode image to base64
def base64_encode(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.b64encode(data).decode("utf-8")

# 1. Load your image
image_url = "https://www.californiastrawberries.com/wp-content/uploads/2021/05/Rainbow-Fruit-Salad-1024.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# 2. Convert image to base64
buffered = BytesIO()
image.save(buffered, format="JPEG")
search_image_base64 = base64_encode(buffered.getvalue())

# 3. Construct the payload
body = {
    "search_image": f"data:image/jpeg;base64,{search_image_base64}",
    "queries": [
        {
            "prompt_image": "",
            "prompts": [
                {
                    "text": "strawberry",
                    "box": [],
                    "box_threshold": 0.1,
                    "multimodal_threshold": 0.22
                }
            ]
        }
    ]
}

# 4. Get prediction
response = sm_runtime.invoke_endpoint(
    EndpointName=zero_endpoint_name,
    Body=json.dumps(body),
    ContentType="application/json",
)

# The 'result' will contain bounding boxes for the detected objects.
result = json.load(response["Body"])
# visualize_detection(result, image)

```

**Example 2: Using a Box (Semantic) Prompt**

Provide an image, a text label, and a bounding box around an object of interest. The model will find other, similar objects in the `search_image`.

```
# (Image loading and encoding is similar to the above example)

# Construct the payload with a box prompt
# The box coordinates are [x_min, y_min, x_max, y_max] in absolute pixel format.
body = {
    "search_image": f"data:image/jpeg;base64,{search_image_base64}",
    "queries": [
        {
            "prompt_image": f"data:image/jpeg;base64,{search_image_base64}",
            "prompts": [
                {
                    "text": "potato",
                    "box": [1779.58, 2096.02, 1891.69, 2158.37],
                    "box_threshold": 0.1,
                    "multimodal_threshold": 0.3
                }
            ]
        }
    ]
}

# Get prediction from the endpoint
# ...

```

### Step 4: Clean Up Resources

To avoid incurring ongoing charges, it is important to delete the SageMaker endpoint when you are finished.

```
# Initialize a sagemaker client to delete the endpoint
sagemaker_client = boto_session.client("sagemaker")

# Delete the endpoint
sagemaker_client.delete_endpoint(EndpointName=zero_endpoint_name)

# Optionally, delete the endpoint configuration and model
sagemaker_client.delete_endpoint_config(EndpointConfigName=zero_endpoint_name)
# You will need to find the model name associated with your endpoint to delete it.
# sagemaker_client.delete_model(ModelName=model_name)

```

## 📋 Input/Output Interface

### Input Payload (`application/json`)

The model expects a JSON object with the following structure:

```
{
  "search_image": "data:image/jpeg;base64,<base64_encoded_string>",
  "queries": [
    {
      "prompt_image": "data:image/jpeg;base64,<base64_encoded_string>",
      "prompts": [
        {
          "text": "string_prompt_1",
          "box": [x_min, y_min, x_max, y_max],
          "box_threshold": 0.1,
          "multimodal_threshold": 0.22
        }
      ]
    }
  ]
}

```

- `search_image`: **Required**. The image to perform detection on, as a Base64 encoded string with a data URI prefix.
- `queries`: **Required**. A list of query objects. Each query can have its own `prompt_image`.
  - `prompt_image`: The image containing the example objects for semantic search. Can be an empty string (`""`) for simple text-based search.
  - `prompts`: A list of prompt definitions.
    - `text`: **Required**. The text label for the object you want to find.
    - `box`: A list of four numbers defining a bounding box `[x_min, y_min, x_max, y_max]`. Coordinates must be in **absolute pixel values**. For text-only prompts, provide an empty list `[]`.
    - `box_threshold`: A confidence threshold for box-only prompts.
    - `multimodal_threshold`: A confidence threshold for combined text/box prompts.

### Output Response (`application/json`)

The model returns a JSON object containing the prediction results:

```
{
    "output": [
        {
            "boxes": [
                [x1_min, y1_min, x1_max, y1_max],
                [x2_min, y2_min, x2_max, y2_max],
                ...
            ],
            "text": [
                "string_prompt_1",
                "string_prompt_2",
                ...
            ]
        }
    ]
}

```

- `output`: A list containing a single dictionary with the detection results.
  - `boxes`: A list of predicted bounding boxes. Each box corresponds to a label in the `text` list at the same index.
  - `text`: A list of labels corresponding to the detected objects.

## 🛠️ Technical Specifications

- **Model Size:** 622M parameters
- **Performance:** 1.03 TFLOPS
- **Recommended Instance Type:** `ml.g4dn.xlarge` or other GPU instances.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## ❓ Support

For questions, issues, or support regarding the ZERO model or this sample notebook, please open an issue in this GitHub repository.

For business inquiries or questions about Superb AI's other offerings, please contact us at [contact@superb-ai.com](mailto:contact@superb-ai.com).
