#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:16:24 2023

@author: oyamatoshiki
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
from PIL import Image

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

st.title("Stable Diffusion Image Generation")

text_input = st.text_input("Enter text:")

generated_image = pipe(text_input).images[0]

st.image(generated_image, caption="Generated Image", use_column_width=True)




