import gradio as gr
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Add src to path if needed or just import since app.py is at root
from src.models.unet import UNet

def process_image(img_pil, task, model_type):
    if img_pil is None:
        return None
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration based on task
    hr_size = 128
    lr_size = 64
    scale = 2 if task == "Super Resolution (2x)" else 1
    in_size = lr_size if task == "Super Resolution (2x)" else hr_size
    
    if model_type == "Baseline U-Net":
        checkpoint_name = "unet_sr_epochs_50.pt" if task == "Super Resolution (2x)" else "unet_inpainting_epochs_50.pt"
        checkpoint_path = Path("checkpoints") / checkpoint_name
        
        # Check if user put the file in the root folder instead of checkpoints/
        if not checkpoint_path.exists():
            checkpoint_path = Path(checkpoint_name)
            
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_name}")
            return img_pil # Return original if no model found
                
        # Load model
        model = UNet(base=32, scale_factor=scale)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Preprocess
        img = img_pil.convert('RGB')
        if img.size != (in_size, in_size):
            img = img.resize((in_size, in_size), Image.BICUBIC)
            
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        t_img = norm(TF.to_tensor(img)).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(t_img)
            
        # Postprocess
        pred = pred.squeeze(0).cpu()
        for c, (m, s) in enumerate(zip([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
            pred[c].mul_(s).add_(m)
        pred = pred.clamp(0, 1)
        out_img = TF.to_pil_image(pred)
        
        return out_img
        
    # For GAN and DDPM, not yet implemented, return original
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
