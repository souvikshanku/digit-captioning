import gradio as gr
import torch

from model import DigitCaptioner
from utils import get_caption


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DigitCaptioner(input_size=29, hidden_size=256, device=device)
checkpoint = torch.load("checkpoints/epoch_15.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


def predict(image):
    inp = image["composite"]
    inp = torch.tensor(inp).view(1, 1, 64, 64)
    inp = inp.float().to(device)
    caption = get_caption(model, inp)
    return caption


head = (
  "<center>"
  "<h1>" "Draw any two digit number in the canvas" "</h1>"
  "</center>"
)

sp = gr.Sketchpad(
    image_mode="L",
    type="numpy",
    crop_size=(64, 64),
    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed")
)

gr.Interface(
    fn=predict,
    inputs=sp,
    outputs="label",
    description=head,
    live=False
).launch()
