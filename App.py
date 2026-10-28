import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2

import json

# Model Path
PATH = "trained.pth"

st.title("Age Estimation")


def predict(model, image):
    return model(image)


transformed_image = None


class AgeCNN(nn.Module):
    def __init__(self,input_channels,hidden_dimensions=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_channels,hidden_dimensions,(2,2)),
            nn.BatchNorm2d(hidden_dimensions),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(hidden_dimensions,hidden_dimensions*4,(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(hidden_dimensions*4,hidden_dimensions*8,(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(hidden_dimensions*16),
            nn.ReLU(),
            nn.Linear(hidden_dimensions*16,hidden_dimensions*8),
            nn.ReLU(),
            nn.Linear(hidden_dimensions*8,hidden_dimensions*4),
            nn.ReLU(),
            nn.Linear(hidden_dimensions*4,1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.network(x)


def load_state(path):
    return torch.load(path, weights_only=True,map_location='cpu')


with open("mean_std.json", "r") as file:
    mean_std = json.load(file)

# Cache resource to avoid continuous loading of model
@st.cache_resource
def load_model(PATH):
    model = AgeCNN(input_channels=3)
    model.load_state_dict(load_state(PATH))
    return model


model = load_model(PATH)
model.eval()

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
    ]
)

uploaded_file = st.file_uploader(
    "Upload an Image: ", max_upload_size=4, type=[".png", ".jpg"]
)
if uploaded_file is not None:
    if uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)

        transformed_image = transforms(image).unsqueeze(0)
        st.image(image, width=300)

        # Display Result
        with torch.no_grad():
            prediction = torch.round(model(transformed_image)).numpy()
            st.metric(
                "Predicted Age",
                f"{round(prediction.item())} Years",
            )
