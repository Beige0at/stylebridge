"""StyleBridge engine.

This is where the actual style transfer happens. We push the content and style
images through a pretrained VGG19, grab features from a handful of layers, then
optimise the output pixels so the content structure survives while the style
texture gets painted on.

Style objective (pick one):
  - "gram": the classic Gram-matrix style signature (baseline).
  - "swd":  sliced Wasserstein distance (what we actually ship).

Optimiser (pick one):
  - "adam":       steady first-order steps, the safe default.
  - "adam_lbfgs": Adam to settle, then a short L-BFGS polish.
  - "lbfgs":      pure L-BFGS; quick when it behaves, flaky on SWD.

The "saliency guard" keeps faces from getting mangled: it builds a soft mask
over detected faces and pulls those pixels back toward the content.

Run it directly:
    python stylebridge_engine.py content.jpg style.jpg --out result.jpg \
        --optimizer adam --style-weight 1e4 --num-steps 500 --resolution 400 \
        --saliency-guard haar
"""

import os

# ROCm on the RX 6700 XT needs this set before torch is imported.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

# MIOpen runs through the cuDNN shim and segfaults on this card, so switch it off.
torch.backends.cudnn.enabled = False

import cv2

# --------------------------------------------------------------------------- #
# Device / model singletons (load once, reuse for every call)
# --------------------------------------------------------------------------- #
_device = None


def get_device():
    # cache the device so we don't re-query CUDA on every call
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


_vgg = None


def get_vgg():
    """Load VGG19 features once and freeze them. Shared by every transfer()."""
    global _vgg
    if _vgg is None:
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(get_device()).eval()
        # we only ever read features, never train the network
        for param in vgg.parameters():
            param.requires_grad_(False)
        _vgg = vgg
    return _vgg


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #
def load_image(image_path, max_size=400, shape=None):
    """Load an image as a normalised NCHW tensor.

    Pass shape=(H, W) to force an exact size (used for the style image so it
    lines up with the content). Otherwise max_size caps the longer edge and the
    aspect ratio is preserved.
    """
    image = Image.open(image_path).convert("RGB")
    # PIL gives (W, H). For the int case we just cap the longer edge.
    size = min(max(image.size), max_size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = in_transform(image)[:3, :, :].unsqueeze(0)   # add the batch dim
    return image.to(get_device())


# ImageNet stats, kept around so we can invert them on save.
_IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_IMG_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def save_image(tensor, path):
    """Save a tensor that's still in ImageNet-normalised space.

    We undo the normalisation first so pixel values come back exactly as they
    were. That matters for the guard blend, which expects protected pixels to
    match the content's.
    """
    img = tensor.detach().cpu().squeeze(0).numpy()                   # C,H,W
    img = img * _IMG_STD[:, None, None] + _IMG_MEAN[:, None, None]   # denormalise
    img = np.clip(img, 0, 1)                                         # back to [0,1]
    vutils.save_image(torch.from_numpy(img).unsqueeze(0), path, normalize=False)


# --------------------------------------------------------------------------- #
# Feature extraction / losses
# --------------------------------------------------------------------------- #
# VGG19 layers we pull features from. Keys are the module index inside
# .features, values are the names people actually recognise. conv1_1..conv5_1
# these feed the style loss, conv4_2 is the content reference.
STYLE_LAYERS = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",  # content
    "28": "conv5_1",
}

# How much each style layer counts. Early layers carry fine texture so they get
# more weight. The deep layers are coarse and get less.
STYLE_WEIGHTS = {
    "conv1_1": 1.0,
    "conv2_1": 0.8,
    "conv3_1": 0.5,
    "conv4_1": 0.3,
    "conv5_1": 0.1,
}


def get_features(image, model, layers=None):
    # run the image through the feature extractor and keep the layers we want
    if layers is None:
        layers = STYLE_LAYERS
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    # flatten H,W then multiply the feature matrix by its own transpose -> (C,C)
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def _make_projections(channels, num_projections, device, seed):
    """Build the random projection directions SWD slices along.

    We draw these once per run (seeded) and then keep them fixed for every step.
    Re-drawing them each iteration makes the loss jump around and makes runs
    impossible to reproduce; holding them fixed keeps the objective stationary.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    projections = torch.randn(channels, num_projections, generator=gen)
    # normalise each direction to unit length, or a few slices would dominate
    projections = projections / projections.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return projections.to(device)


def swd_loss(target_feature, style_feature, projections):
    """Sliced-Wasserstein distance between two feature maps.

    projections is (C, num_projections); each column is one slice direction.
    """
    _, c, h, w = target_feature.shape
    if projections.shape[0] != c:
        raise ValueError(
            f"projection channels ({projections.shape[0]}) != feature channels ({c})")
    # (C, H*W) -> (H*W, C). Every spatial position becomes one sample.
    target_flat = target_feature.view(c, h * w).transpose(0, 1)
    style_flat = style_feature.view(c, h * w).transpose(0, 1)
    # project all samples onto each slice direction
    target_proj = torch.matmul(target_flat, projections)
    style_proj = torch.matmul(style_flat, projections)
    # SWD compares the sorted (quantile) projections, not the raw samples
    target_proj_sorted, _ = torch.sort(target_proj, dim=0)
    style_proj_sorted, _ = torch.sort(style_proj, dim=0)
    return torch.mean((target_proj_sorted - style_proj_sorted) ** 2)


def _auto_projections(resolution, k=0.25):
    """Scale the projection count with resolution: proj ~= res * k.

    Batch A showed 64..200 projections landed on basically the same SSIM, so we
    sit at the cheap end (a quarter of the resolution) for speed.
    """
    return max(1, int(round(resolution * k)))


def _resize_mask_to(mask, spatial):
    """Match a (H,W) mask to some feature map's (H',W')."""
    if mask.shape[-2:] != spatial:
        # interpolate wants 4D, so tack on batch + channel dims and drop them after
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=spatial, mode="bilinear", align_corners=False)
        return mask[0, 0]
    return mask


# --------------------------------------------------------------------------- #
# Saliency Guard (spatial face masks)
# --------------------------------------------------------------------------- #
# Haar cascades are loaded once and cached.
_haar_face = None
_haar_profile = None
_haar_eye = None


def _get_haar_cascades():
    global _haar_face, _haar_profile, _haar_eye
    if _haar_face is None:
        # OpenCV 5 removed the cascade API. Fail with a message that says what
        # to do, rather than a bare AttributeError.
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2.data, "haarcascades"):
            raise RuntimeError(
                "OpenCV >=5 removed cv2.CascadeClassifier; the Saliency Guard "
                "needs opencv-python<5.")
        data = cv2.data.haarcascades
        _haar_face = cv2.CascadeClassifier(os.path.join(data, "haarcascade_frontalface_default.xml"))
        _haar_profile = cv2.CascadeClassifier(os.path.join(data, "haarcascade_profileface.xml"))
        _haar_eye = cv2.CascadeClassifier(os.path.join(data, "haarcascade_eye.xml"))
    return _haar_face, _haar_profile, _haar_eye


def _haar_mask_np(image_path, out_hw):
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(out_hw, dtype=np.float32)   # no image, no mask
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # flatten the histogram a bit. The detectors behave better in odd lighting.
    gray_eq = cv2.equalizeHist(gray)
    face, profile, eye = _get_haar_cascades()

    # run frontal + profile detectors at a couple of scales. Overlaps get merged
    # by the blur below, so we don't bother de-duplicating here
    boxes = []
    for det in (face, profile):
        for sf in (1.05, 1.1):
            found = det.detectMultiScale(gray_eq, scaleFactor=sf, minNeighbors=4,
                                         minSize=(40, 40))
            boxes.extend(tuple(b) for b in found)

    mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, bw, bh) in boxes:
        pad = int(0.08 * min(bw, bh))            # grow the box a little
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        mask[y0:y1, x0:x1] = 1.0
        # eyes get their own protection inside the face box
        roi = gray_eq[y0:y1, x0:x1]
        eyes = eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes:
            # only the top half of the eye box. The eye + brow carry most of the
            # identity. Leaving the lower half free lets the styliser work on the
            # eye-bag / cheek area.
            mask[y0 + ey:y0 + ey + eh // 2, x0 + ex:x0 + ex + ew] = 1.0

    # soften the edges, then push the masked core back up to 1
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=h * 0.008)
    mask = np.clip(mask * 2.0, 0.0, 1.0)
    # finally match the feature-map resolution
    mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32)


_MP_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
_mp_detector = None


def _get_mp_detector():
    global _mp_detector
    if _mp_detector is None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception:
            return None                          # mediapipe not installed
        if not _MP_MODEL.exists():
            return None                          # model file missing
        base = mp_python.BaseOptions(model_asset_path=str(_MP_MODEL))
        opts = vision.FaceLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            num_faces=4, min_face_detection_confidence=0.5)
        _mp_detector = vision.FaceLandmarker.create_from_options(opts)
    return _mp_detector


def _mediapipe_mask_np(image_path, out_hw):
    try:
        import mediapipe as mp
    except Exception:
        return _haar_mask_np(image_path, out_hw)   # fall back to Haar

    det = _get_mp_detector()
    if det is None:
        return _haar_mask_np(image_path, out_hw)   # fall back to Haar

    mp_image = mp.Image.create_from_file(str(image_path))
    res = det.detect(mp_image)
    h, w = mp_image.height, mp_image.width
    mask = np.zeros((h, w), dtype=np.float32)
    if res.face_landmarks:
        for lm in res.face_landmarks:
            # fill the convex hull of the 468 landmarks rather than their
            # bounding box, so background corners inside the box aren't protected
            pts = np.array(
                [[int(p.x * (w - 1)), int(p.y * (h - 1))] for p in lm],
                dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            if pts.shape[0] >= 3:
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
    # same soft-edge treatment as the Haar path
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=h * 0.008)
    mask = np.clip(mask * 2.0, 0.0, 1.0)
    mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32)


def saliency_guard_mask(image_path, out_hw, mode="haar"):
    if mode == "haar":
        return _haar_mask_np(image_path, out_hw)
    if mode == "mediapipe":
        return _mediapipe_mask_np(image_path, out_hw)
    return np.zeros(out_hw, dtype=np.float32)      # "off"


# --------------------------------------------------------------------------- #
# Loss assembly (built once per run)
# --------------------------------------------------------------------------- #
class _Diverged(RuntimeError):
    """Thrown inside an L-BFGS closure when the loss blows up or goes NaN.

    Raising out of .step() is safer than returning a junk loss: otherwise
    PyTorch carries on with whatever gradient happened to be left lying around.
    """


# Anything past this and we call it diverged.
_DIVERGE_LIMIT = 1e6


class _Losses:
    """Content + style loss for a single run.

    Anything that doesn't change between steps (reference features, style grams,
    SWD projections) is precomputed here so the inner loop stays cheap.
    """

    def __init__(self, vgg, content_features, style_features,
                 style_grams, num_projections, style_mode, guard,
                 guard_content_boost=2.0, seed=0):
        self.vgg = vgg
        self.content_features = content_features
        self.style_features = style_features
        self.style_grams = style_grams
        self.style_mode = style_mode
        self.guard = guard            # (H,W) tensor, or None when guard is off
        self.guard_content_boost = guard_content_boost
        # one fixed set of SWD projections per style layer, seeded so two runs
        # with the same seed come out identical
        self.projections = {}
        if style_mode == "swd":
            for i, layer in enumerate(STYLE_WEIGHTS):
                feat = style_features[layer]
                self.projections[layer] = _make_projections(
                    feat.shape[1], num_projections, feat.device, seed + i)

    def compute(self, target):
        tf = get_features(target, self.vgg)

        # ---- content: how far conv4_2 has drifted from the content image ----
        c = tf["conv4_2"]
        c_ref = self.content_features["conv4_2"]

        if self.guard is not None:
            # inside the face mask, weight the content term up so it dominates
            gm = _resize_mask_to(self.guard, c.shape[-2:])   # (H',W')
            w = 1.0 + self.guard_content_boost * gm
            c_loss = torch.mean(w * ((c - c_ref) ** 2).mean(dim=1))
        else:
            c_loss = torch.mean((c - c_ref) ** 2)

        # ---- style: weighted sum over the style layers ----------------------
        s_loss = torch.zeros((), device=c.device)
        w_sum = sum(STYLE_WEIGHTS.values())
        for layer, lw in STYLE_WEIGHTS.items():
            t = tf[layer]
            if self.style_mode == "gram":
                tg = gram_matrix(t)
                _, d, h, w = t.shape
                # divide by d*h*w to keep the per-layer scale comparable
                s_loss = s_loss + lw * torch.mean((tg - self.style_grams[layer]) ** 2) / (d * h * w)
            else:  # swd
                s = self.style_features[layer]
                s_loss = s_loss + lw * swd_loss(t, s, self.projections[layer])

        s_loss = s_loss / w_sum            # normalise the layer weights
        return c_loss, s_loss

    def total(self, target, content_weight, style_weight):
        c, s = self.compute(target)
        return content_weight * c + style_weight * s, c, s


# --------------------------------------------------------------------------- #
# Transfer engine
# --------------------------------------------------------------------------- #
def transfer(content_path, style_path, *,
             optimizer="adam",
             content_weight=1, style_weight=1e4,
             num_steps=500, lbfgs_steps=20,
             num_projections=None, resolution=400,
             saliency_guard="off",
             style_mode="swd",
             guard_content_boost=2.0,
             guard_blend=1.0,
             line_search=None,
             mask_file=None,
             seed=0,
             verbose=True):
    """
    Run a transfer and return (output_tensor, loss_history).

    optimizer: "adam" | "adam_lbfgs" | "lbfgs"
    style_mode: "swd" | "gram"
    seed: seeds the fixed SWD projections so runs repeat exactly.
    verbose: print progress to stdout.
    """
    device = get_device()
    vgg = get_vgg()

    content = load_image(content_path, max_size=resolution)
    # size the style to the content so the two line up spatially
    style = load_image(style_path, shape=content.shape[-2:])

    guard = None
    if saliency_guard in ("haar", "mediapipe"):
        # build the mask once, up front. Faces don't move during the run.
        guard = torch.from_numpy(
            saliency_guard_mask(content_path, content.shape[-2:], mode=saliency_guard)
        ).to(device)
        if mask_file:
            np.save(mask_file, guard.detach().cpu().numpy())

    if num_projections is None:
        # caller didn't set it -> scale it from the resolution
        num_projections = _auto_projections(resolution)

    # precompute the reference features that the loss compares against
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_WEIGHTS}

    losses = _Losses(vgg, content_features, style_features, style_grams,
                     num_projections, style_mode, guard, guard_content_boost,
                     seed=seed)

    def total_for(target):
        return losses.total(target, content_weight, style_weight)

    loss_history = []
    broke = {"flag": False}

    if verbose:
        print(f"[engine] optimizer={optimizer} style_mode={style_mode} res={resolution} "
              f"proj={num_projections} steps={num_steps} guard={saliency_guard}")

    # log roughly ten times over the run, whatever the step count is
    log_every = max(1, num_steps // 10)

    # ---- Adam path (used by "adam" and the first half of "adam_lbfgs") ------
    if optimizer in ("adam", "adam_lbfgs"):
        # what we optimise, the output pixels, seeded from the content
        target = content.clone().requires_grad_(True).to(device)
        opt = optim.Adam([target], lr=0.02)
        for ii in range(1, num_steps + 1):
            total, c_loss, s_loss = total_for(target)
            loss_history.append(float(total))
            opt.zero_grad()
            total.backward()
            opt.step()
            if verbose and ii % log_every == 0:
                print(f"  adam[{ii}/{num_steps}] total={total.item():.5f} "
                      f"style={s_loss.item():.4f} content={c_loss.item():.4f}")

        # ---- optional L-BFGS polish on top of Adam --------------------------
        if optimizer == "adam_lbfgs":
            if verbose:
                print("  -> L-BFGS refinement pass")
            del opt                              # free the Adam state before L-BFGS
            gc.collect()
            torch.cuda.empty_cache()
            lbfgs_kw = {}
            if line_search:
                lbfgs_kw["line_search_fn"] = line_search
            lbfgs_opt = optim.LBFGS([target], max_iter=lbfgs_steps, tolerance_grad=1e-05,
                                    **lbfgs_kw)
            # keep the Adam result so we can rewind if L-BFGS misbehaves
            backup = target.detach().clone()

            def closure():
                lbfgs_opt.zero_grad()
                total, _, _ = total_for(target)
                if not torch.isfinite(total) or abs(total) > _DIVERGE_LIMIT:
                    # bail rather than hand L-BFGS a loss with no fresh gradient
                    raise _Diverged(float(total))
                total.backward()
                loss_history.append(float(total))
                return total

            try:
                lbfgs_opt.step(closure)
                if verbose:
                    print(f"  lbfgs final={loss_history[-1]:.4f}")
            except _Diverged:
                broke["flag"] = True
                with torch.no_grad():
                    target.copy_(backup)        # rewind to the Adam result
                if verbose:
                    print("  !! L-BFGS diverged; keeping Adam result.")

    # ---- pure L-BFGS --------------------------------------------------------
    else:
        target = content.clone().requires_grad_(True).to(device)
        lbfgs_kw = {}
        if line_search:
            lbfgs_kw["line_search_fn"] = line_search
        lbfgs_opt = optim.LBFGS([target], max_iter=num_steps, tolerance_grad=1e-05,
                                **lbfgs_kw)
        # remember the last good iterate so we can rewind on divergence
        state = {"best": target.detach().clone()}

        def closure():
            lbfgs_opt.zero_grad()
            total, _, _ = total_for(target)
            if not torch.isfinite(total) or abs(total) > _DIVERGE_LIMIT:
                raise _Diverged(float(total))
            total.backward()
            loss_history.append(float(total))
            state["best"] = target.detach().clone()   # last finite iterate
            return total

        try:
            lbfgs_opt.step(closure)
        except _Diverged:
            broke["flag"] = True
            with torch.no_grad():
                target.copy_(state["best"])
            if verbose:
                print("  !! pure L-BFGS diverged (expected with SWD).")

    target = target.detach()
    if guard is not None:
        # blend protected areas back toward the content so the face survives.
        # The mask already matches the target size (it was built from it).
        mask_soft = guard.unsqueeze(0).unsqueeze(0)           # 1,1,H,W
        target = (1 - guard_blend * mask_soft) * target + guard_blend * mask_soft * content
    # no clamp here. We're still in normalised space, so values can be <0 or >1.
    # save_image() undoes the normalisation and clips at the end.
    return target, loss_history


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def content_ssim(content_path, generated_path):
    """Whole-image grayscale SSIM between content and output (higher is better).

    skimage is only needed for this one helper, so import it here to keep the
    engine itself dependency-light.
    """
    from skimage.metrics import structural_similarity as compare_ssim
    img_content = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
    img_gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if img_content is None or img_gen is None:
        raise FileNotFoundError("Image missing for SSIM")
    # resize the output to the content so SSIM has matched grids
    img_gen = cv2.resize(img_gen, (img_content.shape[1], img_content.shape[0]))
    score, _ = compare_ssim(img_content, img_gen, full=True)
    return score


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    # every flag mirrors a transfer() kwarg, see transfer() for what they do
    ap = argparse.ArgumentParser(description="StyleBridge neural style transfer")
    ap.add_argument("content")
    ap.add_argument("style")
    ap.add_argument("--out", default="output.jpg")
    ap.add_argument("--optimizer", choices=["adam", "adam_lbfgs", "lbfgs"], default="adam")
    ap.add_argument("--style-mode", choices=["swd", "gram"], default="swd")
    ap.add_argument("--content-weight", type=float, default=1)
    ap.add_argument("--style-weight", type=float, default=1e4)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--lbfgs-steps", type=int, default=20)
    ap.add_argument("--num-projections", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=400)
    ap.add_argument("--saliency-guard", choices=["off", "haar", "mediapipe"], default="off")
    ap.add_argument("--guard-content-boost", type=float, default=2.0)
    ap.add_argument("--guard-blend", type=float, default=1.0,
                    help="How strongly protected regions are blended back to content "
                         "(0 = off, 1 = full preservation).")
    ap.add_argument("--line-search", choices=["none", "strong_wolfe"], default="none",
                    help="L-BFGS line search. PyTorch offers none (default, converges) "
                         "or strong_wolfe (stalls on SWD loss).")
    ap.add_argument("--loss-file", default=None,
                    help="Write the full loss history to this JSON file.")
    ap.add_argument("--mask-file", default=None,
                    help="Write the saliency guard mask (npy) used by the run.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the fixed SWD projection directions (reproducibility).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-step progress logging.")
    args = ap.parse_args()

    t0 = time.time()
    target, history = transfer(
        args.content, args.style,
        optimizer=args.optimizer,
        style_mode=args.style_mode,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        num_steps=args.num_steps,
        lbfgs_steps=args.lbfgs_steps,
        num_projections=args.num_projections,
        resolution=args.resolution,
        saliency_guard=args.saliency_guard,
        guard_content_boost=args.guard_content_boost,
        guard_blend=args.guard_blend,
        line_search=(None if args.line_search == "none" else args.line_search),
        mask_file=args.mask_file,
        seed=args.seed,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0
    save_image(target, args.out)
    if args.loss_file:
        # dump the per-step loss so the benchmark harness can pick it up
        import json
        with open(args.loss_file, "w") as fh:
            json.dump({"loss": history}, fh)
    print(f"[engine] done in {elapsed:.2f}s, {len(history)} steps -> {args.out}")


if __name__ == "__main__":
    main()
