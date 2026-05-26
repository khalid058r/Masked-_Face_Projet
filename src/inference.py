import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.unet import UNet

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Masked Face Super Resolution")
    parser.add_argument("--image", type=str, required=True, help="Path to input masked image")
    parser.add_argument("--task", type=str, choices=["inpainting", "sr"], default="inpainting", help="Task to perform: inpainting or sr")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the output image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuration based on task
    hr_size = 128
    lr_size = 64
    scale = 2 if args.task == 'sr' else 1
    in_size = lr_size if args.task == 'sr' else hr_size
    
    print(f"Loading model for {args.task} from {args.checkpoint}...")
    model = UNet(base=32, scale_factor=scale)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    # Check if it's a full state dict or wrapped in a 'model' key
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    print(f"Processing image {args.image}...")
    img = Image.open(args.image).convert('RGB')
    if img.size != (in_size, in_size):
        img = img.resize((in_size, in_size), Image.BICUBIC)
        
    # Normalize image [-1, 1]
    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    t_img = norm(TF.to_tensor(img)).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        pred = model(t_img)
        
    # Denormalize output
    pred = pred.squeeze(0).cpu()
    for c, (m, s) in enumerate(zip([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
        pred[c].mul_(s).add_(m)
    pred = pred.clamp(0, 1)
    
    # Save output
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_img = TF.to_pil_image(pred)
    out_img.save(args.output)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
