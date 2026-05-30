import gradio as gr
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path

from src.models.unet import UNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")


def extract_state_dict(ckpt, model_type: str) -> dict:
    """Gère les 4 formats de checkpoint possibles."""
    if "state_G" in ckpt:          # GAN Kaggle  {'state_G':..., 'state_D':..., 'test':...}
        log.info("Format détecté : GAN Kaggle (state_G)")
        return ckpt["state_G"]
    elif "state" in ckpt:          # UNet Kaggle {'state':..., 'test':...}
        log.info("Format détecté : UNet Kaggle (state)")
        return ckpt["state"]
    elif "model" in ckpt:          # GAN local   {'model':..., 'discriminator':...}
        log.info("Format détecté : GAN local (model)")
        return ckpt["model"]
    else:                           # UNet local  OrderedDict direct
        log.info("Format détecté : UNet local (state_dict direct)")
        return ckpt


def resolve_checkpoint(prefix: str, suffix: str) -> tuple[Path | None, dict | None]:
    """
    Cherche dans l'ordre :
      1. resultatKaggle/<prefix>_<suffix>_best.pt   (meilleur modèle Kaggle)
      2. checkpoints/<prefix>_<suffix>_epochs_50.pt
      3. <prefix>_<suffix>_epochs_50.pt             (racine du projet)
    Retourne (path, métriques_test) ou (None, None).
    """
    candidates = [
        Path("resultatKaggle") / f"{prefix}_{suffix}_best.pt",
        Path("checkpoints") / f"{prefix}_{suffix}_epochs_50.pt",
        Path(f"{prefix}_{suffix}_epochs_50.pt"),
    ]
    for p in candidates:
        if p.exists():
            log.info(f"Checkpoint trouvé : {p}")
            ckpt = torch.load(p, map_location="cpu")
            test_metrics = ckpt.get("test") if isinstance(ckpt, dict) else None
            if test_metrics:
                psnr = test_metrics.get("psnr", "?")
                ssim = test_metrics.get("ssim", "?")
                lpips = test_metrics.get("lpips", "N/A")
                log.info(f"Métriques test : PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  LPIPS={lpips}")
            return p, ckpt
    log.error(f"Aucun checkpoint trouvé pour {prefix}_{suffix}")
    return None, None


def load_model(ckpt: dict, scale: int, device: str) -> UNet:
    sd = extract_state_dict(ckpt, "")
    model = UNet(base=32, scale_factor=scale)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    log.info(f"Modèle chargé — {n:,} paramètres — device={device}")
    return model


def run_inference(model: UNet, t_img: torch.Tensor) -> torch.Tensor:
    log.debug(f"Entrée — shape={t_img.shape}  min={t_img.min():.3f}  max={t_img.max():.3f}")
    with torch.no_grad():
        pred = model(t_img)
    diff = (pred - t_img).abs().mean().item()
    status = "⚠️ quasi-identique (hors distribution)" if diff < 0.05 else "✅ reconstruction active"
    log.info(f"MAD input→output : {diff:.4f}  {status}")
    log.debug(f"Sortie — min={pred.min():.3f}  max={pred.max():.3f}")
    return pred


def postprocess(pred: torch.Tensor, original_size: tuple) -> Image.Image:
    img = (pred.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    out = TF.to_pil_image(img).resize(original_size, Image.BICUBIC)
    log.info(f"Image de sortie : {out.size}")
    return out


def process_image(img_pil, task, model_type):
    log.info(f"=== Requête — task={task!r}  model={model_type!r} ===")

    if img_pil is None:
        log.warning("Aucune image fournie")
        return None

    log.info(f"Image entrée : {img_pil.size}  mode={img_pil.mode}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device : {device}")

    hr_size, lr_size = 128, 64
    scale   = 2 if task == "Super Resolution (2x)" else 1
    in_size = lr_size if task == "Super Resolution (2x)" else hr_size
    suffix  = "sr" if task == "Super Resolution (2x)" else "inpainting"

    if model_type == "DDPM":
        log.warning("DDPM non implémenté — retour de l'image originale")
        return img_pil

    prefix = "unet" if model_type == "Baseline U-Net" else "gan"
    ckpt_path, ckpt = resolve_checkpoint(prefix, suffix)
    if ckpt_path is None:
        return img_pil

    try:
        model = load_model(ckpt, scale, device)
    except Exception as e:
        log.exception(f"Erreur chargement modèle : {e}")
        return img_pil

    img = img_pil.convert("RGB")
    if img.size != (in_size, in_size):
        log.debug(f"Resize entrée : {img.size} → ({in_size},{in_size})")
        img = img.resize((in_size, in_size), Image.BICUBIC)

    norm  = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    t_img = norm(TF.to_tensor(img)).unsqueeze(0).to(device)

    try:
        pred = run_inference(model, t_img)
    except Exception as e:
        log.exception(f"Erreur inférence : {e}")
        return img_pil

    return postprocess(pred, img_pil.size)


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="Masked Face Super Resolution", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Masked Face Super Resolution")
    gr.Markdown(
        "Transform masked faces and upscale them using modern Deep Learning techniques (U-Net, PatchGAN, DDPM).\n\n"
        "**Modèles disponibles :** `resultatKaggle/*_best.pt` (priorité) → `checkpoints/` → racine du projet."
    )

    with gr.Row():
        with gr.Column():
            input_img  = gr.Image(type="pil", label="Masked Input Image")
            task       = gr.Radio(["Inpainting", "Super Resolution (2x)"], value="Inpainting", label="Task")
            model_type = gr.Radio(["Baseline U-Net", "GAN", "DDPM"], value="GAN", label="Model Architecture")
            btn        = gr.Button("Reconstruct", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="pil", label="Reconstructed Face")

    btn.click(process_image, inputs=[input_img, task, model_type], outputs=output_img)

if __name__ == "__main__":
    log.info("Démarrage — vérification des checkpoints disponibles...")
    for prefix in ("unet", "gan"):
        for suffix in ("inpainting", "sr"):
            p, ckpt = resolve_checkpoint(prefix, suffix)
            if p and ckpt and "test" in ckpt:
                m = ckpt["test"]
                log.info(f"  [{prefix}/{suffix}] PSNR={m.get('psnr',0):.2f}dB  SSIM={m.get('ssim',0):.4f}  LPIPS={m.get('lpips','N/A')}")
    demo.launch()
