import gradio as gr
import torch
from PIL import Image

def process_image(img_pil, task, model_type):
    # Wrapper function ready to connect to inference.py logic
    # Returns the input image as a placeholder
    return img_pil

with gr.Blocks(title="Masked Face Super Resolution", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Masked Face Super Resolution")
    gr.Markdown("Transform masked faces and upscale them using modern Deep Learning techniques (U-Net, PatchGAN, DDPM).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Masked Input Image")
            task = gr.Radio(["Inpainting", "Super Resolution (2x)"], value="Inpainting", label="Task")
            model_type = gr.Radio(["Baseline U-Net", "GAN", "DDPM"], value="Baseline U-Net", label="Model Architecture")
            btn = gr.Button("Reconstruct", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(type="pil", label="Reconstructed Face")
            
    btn.click(process_image, inputs=[input_img, task, model_type], outputs=output_img)

if __name__ == "__main__":
    demo.launch()
