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

# ── Face detection (MTCNN optionnel) ─────────────────────────────────────────
try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")
    log.info("MTCNN chargé — détection de visage activée")
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    log.warning("facenet-pytorch absent — fallback centre-crop")


def crop_face(img: Image.Image, margin: float = 0.3) -> Image.Image:
    """
    Détecte et recadre le visage.
    - Avec MTCNN : boîte détectée + marge
    - Sans MTCNN  : crop carré centré sur la moitié haute
    """
    w, h = img.size
    if HAS_MTCNN:
        boxes, _ = _mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * margin, bh * margin
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x1 = max(0, cx - side / 2)
            y1 = max(0, cy - side / 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            log.info(f"Visage détecté (MTCNN) : ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})")
            return img.crop((x1, y1, x2, y2))
        else:
            log.warning("Aucun visage détecté par MTCNN — fallback centre-crop")

    # Fallback : crop carré centré sur le tiers supérieur
    side = min(w, h)
    cx   = w // 2
    cy   = h // 3
    x1   = max(0, cx - side // 2)
    y1   = max(0, cy - side // 4)
    x2   = min(w, x1 + side)
    y2   = min(h, y1 + side)
    log.info(f"Centre-crop fallback : ({x1},{y1}) → ({x2},{y2})")
    return img.crop((x1, y1, x2, y2))


# ── Checkpoints ───────────────────────────────────────────────────────────────

CHECKPOINT_DIRS = {
    "Nouveau (50 ep)": [Path("models_nouveau_update_overfeating")],
    "Kaggle Best":     [Path("output_03_complet_baseline_and_gan"), Path("resultatKaggle")],
}
FILE_PATTERNS = [
    "{prefix}_{suffix}_best.pt",
    "{prefix}_{suffix}_epochs_50.pt",
    "{prefix}_{suffix}_epochs_20.pt",
]
_FALLBACK_DIRS = [
    Path("output_03_complet_baseline_and_gan"),
    Path("resultatKaggle"),
    Path("checkpoints"),
    Path("."),
]


def extract_state_dict(ckpt: dict) -> dict:
    if "state_G" in ckpt:
        log.info("Format : GAN Kaggle (state_G)")
        return ckpt["state_G"]
    if "state" in ckpt:
        log.info("Format : UNet Kaggle (state)")
        return ckpt["state"]
    if "model" in ckpt:
        log.info("Format : local (model)")
        return ckpt["model"]
    log.info("Format : OrderedDict direct")
    return ckpt


def resolve_checkpoint(prefix, suffix, version):
    dirs = CHECKPOINT_DIRS.get(version, [])
    all_dirs = dirs + [d for d in _FALLBACK_DIRS if d not in dirs]
    for folder in all_dirs:
        for pattern in FILE_PATTERNS:
            p = folder / pattern.format(prefix=prefix, suffix=suffix)
            if p.exists():
                log.info(f"Checkpoint : {p}")
                return p, torch.load(p, map_location="cpu")
    log.error(f"Aucun checkpoint trouvé pour {prefix}_{suffix}")
    return None, None


def load_model(ckpt, scale, device):
    model = UNet(base=32, scale_factor=scale)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(device).eval()
    log.info(f"Modèle chargé — {sum(p.numel() for p in model.parameters()):,} params")
    return model


def build_info(ckpt_path, ckpt, mad, face_cropped):
    lines = [f"**Checkpoint :** `{ckpt_path}`"]
    if face_cropped:
        lines.append("**Prétraitement :** visage détecté et recadré automatiquement")
    if isinstance(ckpt, dict) and "test" in ckpt:
        m = ckpt["test"]
        psnr  = m.get("psnr",  "?")
        ssim  = m.get("ssim",  "?")
        lpips = m.get("lpips", "N/A")
        lines.append(
            f"**PSNR :** {psnr:.2f} dB  |  **SSIM :** {ssim:.4f}  |  **LPIPS :** {lpips:.4f}"
            if all(isinstance(v, float) for v in [psnr, ssim])
            else "*Pas de métriques test*"
        )
    else:
        lines.append("*Pas de métriques test dans ce checkpoint*")

    if mad < 0.05:
        lines.append(f"**MAD :** {mad:.4f}  ⚠️ quasi-identique — modèle inactif ou hors distribution")
    else:
        lines.append(f"**MAD :** {mad:.4f}  ✅ reconstruction active")
    return "\n\n".join(lines)


# ── Inférence principale ──────────────────────────────────────────────────────

def process_image(img_pil, task, model_type, version, use_face_crop):
    log.info(f"=== task={task!r}  model={model_type!r}  version={version!r}  crop={use_face_crop} ===")

    if img_pil is None:
        return None, "Aucune image fournie."

    if model_type == "DDPM":
        return img_pil, "DDPM non implémenté."

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    scale   = 2 if task == "Super Resolution (2x)" else 1
    in_size = 64 if task == "Super Resolution (2x)" else 128
    suffix  = "sr" if task == "Super Resolution (2x)" else "inpainting"
    prefix  = "unet" if model_type == "Baseline U-Net" else "gan"

    ckpt_path, ckpt = resolve_checkpoint(prefix, suffix, version)
    if ckpt_path is None:
        return img_pil, f"Aucun checkpoint trouvé pour {prefix}/{suffix}."

    try:
        model = load_model(ckpt, scale, device)
    except Exception as e:
        log.exception(e)
        return img_pil, f"Erreur chargement modèle : {e}"

    img = img_pil.convert("RGB")
    original_size = img.size
    face_cropped = False

    if use_face_crop:
        cropped = crop_face(img)
        if cropped.size != img.size or cropped != img:
            img = cropped
            face_cropped = True

    img = img.resize((in_size, in_size), Image.BICUBIC)
    norm  = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    t_img = norm(TF.to_tensor(img)).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            pred = model(t_img)
        mad = (pred - t_img).abs().mean().item()
        log.info(f"MAD : {mad:.4f}")
    except Exception as e:
        log.exception(e)
        return img_pil, f"Erreur inférence : {e}"

    out = (pred.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    out_pil = TF.to_pil_image(out).resize(original_size, Image.BICUBIC)

    return out_pil, build_info(ckpt_path, ckpt, mad, face_cropped)


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="Masked Face Super Resolution", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Masked Face Super Resolution")
    gr.Markdown("Suppression de masques faciaux — U-Net & GAN.")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Image avec masque")
            task      = gr.Radio(
                ["Inpainting", "Super Resolution (2x)"],
                value="Inpainting", label="Tâche",
            )
            model_type = gr.Radio(
                ["Baseline U-Net", "GAN", "DDPM"],
                value="GAN", label="Architecture",
            )
            version = gr.Radio(
                ["Nouveau (50 ep)", "Kaggle Best"],
                value="Kaggle Best", label="Version du modèle",
                info="Nouveau = models_nouveau_update_overfeating/  |  Kaggle = output_03 / resultatKaggle",
            )
            face_crop = gr.Checkbox(
                value=True,
                label="Détection + recadrage automatique du visage",
                info="Recommandé pour les photos externes (désactiver pour les images du dataset)",
            )
            btn = gr.Button("Reconstruire", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Visage reconstruit")
            model_info = gr.Markdown(label="Infos")

    btn.click(
        process_image,
        inputs=[input_img, task, model_type, version, face_crop],
        outputs=[output_img, model_info],
    )

if __name__ == "__main__":
    log.info("Scan des checkpoints disponibles...")
    for v, dirs in CHECKPOINT_DIRS.items():
        for folder in dirs:
            for prefix in ("unet", "gan"):
                for suffix in ("inpainting", "sr"):
                    for pattern in FILE_PATTERNS:
                        p = folder / pattern.format(prefix=prefix, suffix=suffix)
                        if p.exists():
                            log.info(f"  [{v}] {p}")
                            break
    demo.launch()
