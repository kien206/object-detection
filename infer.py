import streamlit as st
import torch
from torchvision import transforms
import cv2
import numpy as np
import argparse
import torchvision.models.detection as tv
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define command-line arguments
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--model", type=str, required=True, help="Model name (ssd, fasterrcnn, retina or yolo)")
args = parser.parse_args()

# Replace 'YourModelCheckpoint.pth' with the actual path to your model checkpoint
model_name = args.model
if model_name == "ssd":
    model = tv.ssd300_vgg16(pretrained=False)
    model.to(device)
    path = "ssd300.pth"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
elif model_name == "fasterrcnn":
    model = tv.fasterrcnn_resnet50_fpn_v2()
    model.to(device)
    path = "ssd300.pth"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
elif model_name == 'yolo':
    model = YOLO()
    model.to(device)
    path = "yolo.pt"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
else:
    model = tv.retinanet_resnet50_fpn_v2()
    model.to(device)
    path = "yolo.pt"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Streamlit app
st.title("Object Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    with torch.no_grad():
        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)

        # Run the model
        output = model(input_tensor)

        # Process the output (modify this part based on your model's output format)
        # For example, if your model returns bounding boxes and scores:
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        # Display detected objects
        for box, score in zip(boxes, scores):
            if score > 0.5:  # Adjust the threshold as needed
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Display the processed image with detected objects
    st.image(image, caption="Result", use_column_width=True)
