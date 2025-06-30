import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
import cv2
import os

# Fix CUDA warnings and event loop for Colab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
torch.backends.cudnn.benchmark = True
import asyncio
if not hasattr(asyncio, '_nest_patched'):
    asyncio._nest_patched = True

# Helper to adjust image dimensions to multiple of 8
def adjust_dimensions(size):
    return size - (size % 8) if (size % 8) != 0 else size

# Generate depth map
def generate_depth_map(image):
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        depth = depth_estimator(**inputs).predicted_depth
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth = depth.astype(np.uint8)
    return Image.fromarray(depth)

# Prompt validator
def is_valid_prompt(prompt):
    keywords = ["sofa", "table", "chair", "bed", "wall", "curtain", "furniture", "lamp", "floor",
                "mirror", "painting", "vase", "bookshelf", "ceiling", "decor", "interior", "rug"]
    return any(k in prompt.lower() for k in keywords)

# Load AI models
@st.cache_resource
def load_models():
    depth_est = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to("cuda")
    img_proc = DPTImageProcessor.from_pretrained("Intel/dpt-large")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe, depth_est, img_proc

# Load models once
pipe, depth_estimator, image_processor = load_models()

# Streamlit UI
st.set_page_config(page_title="AI Interior Designer", layout="centered")
st.title("🛋️ AI Interior Designer")
uploaded = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Describe the new interior (e.g. 'add a modern sofa and wooden floor'):")

if uploaded and prompt:
    if not is_valid_prompt(prompt):
        st.error("Please enter a design prompt related to room interiors.")
        st.stop()

    # Preprocess image
    image = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image.size
    new_w, new_h = adjust_dimensions(orig_w), adjust_dimensions(orig_h)
    image = image.resize((new_w, new_h))

    # Generate depth
    depth_map = generate_depth_map(image)

    # Generate design
    with st.spinner("Designing your new room..."):
        result = pipe(
            prompt=prompt,
            image=depth_map,
            height=new_h,
            width=new_w,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8
        )
        output = result.images[0].resize((orig_w, orig_h))

    # Display
    st.subheader("Original Room")
    st.image(image, use_container_width=True)

    st.subheader("Redesigned Room")
    st.image(output, use_container_width=True)
