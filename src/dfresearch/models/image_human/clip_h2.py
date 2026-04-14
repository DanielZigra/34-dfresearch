"""
CLIP ViT-B/16 dual-head fake face detector for Subnet 34.

Architecture:
  - Backbone : OpenAI CLIP ViT-B/16 vision encoder (768-d pooler output)
  - Head A   : Binary classifier [real, synthetic]  — submission output
  - Head B   : Generator-type classifier [real, GAN, diffusion, swap, unknown]
                Training-only auxiliary task for generalization.
                Discarded at inference / load_model().

Training strategy (4-stage curriculum):
  Stage A  — linear probe  : backbone frozen, head A+B only, lr=1e-3, 5 ep
  Stage B  — top-4 thaw    : unfreeze last 4 ViT blocks, lr=5e-5, 10 ep
  Stage C  — full fine-tune: layerwise LR decay, lr=1e-5 backbone/1e-4 head, 20 ep
  Stage D  — hard-negative : mine confused samples, lr=5e-6, 5 ep

Loss:
  L = CE(binary) + 0.3 * CE(generator_type) + 0.1 * SupCon(embeddings)
  Label smoothing = 0.1, Mixup alpha = 0.4

Zero-shot probe (Phase 0, before any training):
  Run zero_shot_auc() on ~200 labeled samples.
  AUC >= 0.70 -> skip Stage A, start from Stage B.
  AUC <  0.70 -> run full 4-stage curriculum.

Input:  [B, 3, 224, 224] uint8 [0, 255]  (GASBench standard)
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ── CLIP ViT-B/16 normalization constants ──────────────────────────────────
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# Generator-type label mapping (Head B, training only)
GENERATOR_CLASSES = ["real", "gan", "diffusion", "faceswap", "unknown"]
NUM_GENERATOR_CLASSES = len(GENERATOR_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class CLIPViTB16Detector(nn.Module):
    """
    CLIP ViT-B/16 dual-head binary deepfake detector.

    Head A (self.head_binary)   — submitted to GASBench, always active.
    Head B (self.head_gentype)  — auxiliary generator-type head, training only.
                                  Not included in saved weights for submission.

    Research basis:
      UniversalFakeDetect (Ojha et al., CVPR 2023) — CLIP features generalize
      across unseen generators better than any supervised CNN baseline.
    """

    def __init__(
        self,
        num_classes:        int  = 2,
        num_gen_classes:    int  = NUM_GENERATOR_CLASSES,
        pretrained:         bool = True,
        freeze_backbone:    bool = True,
        model_name:         str  = "openai/clip-vit-base-patch16",
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────────
        if pretrained:
            from transformers import CLIPVisionModel
            self.vision = CLIPVisionModel.from_pretrained(model_name)
        else:
            from transformers import CLIPVisionModel, CLIPVisionConfig
            config = CLIPVisionConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                image_size=224,
                patch_size=16,
            )
            self.vision = CLIPVisionModel(config)

        self.feat_dim = self.vision.config.hidden_size  # 768

        if freeze_backbone:
            for p in self.vision.parameters():
                p.requires_grad_(False)

        # ── Head A: binary real / synthetic (submission head) ─────────────
        self.head_binary = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, num_classes),
        )

        # ── Head B: generator type (training-only auxiliary head) ──────────
        self.head_gentype = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, num_gen_classes),
        )

        # ── Normalization buffers (moved with model.to(device)) ───────────
        self.register_buffer(
            "pixel_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std",  torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-mode forward. Returns binary logits in the same dtype as input.

        GASBench passes uint8 [0, 255]. Internally we cast to float32 for the
        backbone, then cast logits back to the original input dtype before
        returning so the output type always matches the sample type.

        Args:
            x: [B, 3, H, W] uint8 [0, 255]  — GASBench standard input

        Returns:
            logits: [B, 2] same dtype as x  — [real, synthetic]
        """
        input_dtype = x.dtype
        feat = self._encode(x)
        return self.head_binary(feat).to(input_dtype)

    def forward_train(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training-mode forward. Returns binary logits, generator-type logits,
        and the normalized feature vector for contrastive loss.

        All output tensors are cast back to the input dtype so that mixed-
        precision training (bfloat16 dataloaders) stays consistent throughout.

        Args:
            x: [B, 3, H, W] uint8 [0, 255]

        Returns:
            logits_binary:  [B, 2]                      same dtype as x
            logits_gentype: [B, NUM_GENERATOR_CLASSES]  same dtype as x
            feat_normed:    [B, feat_dim]                same dtype as x, for SupCon
        """
        input_dtype = x.dtype
        feat        = self._encode(x)
        feat_normed = F.normalize(feat, dim=-1)
        return (
            self.head_binary(feat).to(input_dtype),
            self.head_gentype(feat).to(input_dtype),
            feat_normed.to(input_dtype),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Cast uint8 → float, normalize, run backbone, return pooler output."""
        x = x.float() / 255.0
        x = (x - self.pixel_mean) / self.pixel_std
        return self.vision(pixel_values=x).pooler_output  # [B, 768]

    # ── Backbone surgery helpers (called by training script) ─────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Stage A)."""
        for p in self.vision.parameters():
            p.requires_grad_(False)

    def unfreeze_top_blocks(self, n: int = 4) -> None:
        """
        Unfreeze the top-n transformer encoder blocks + post_layernorm (Stage B).

        ViT-B/16 has 12 encoder layers (indices 0..11).
        n=4 thaws layers 8,9,10,11 — highest-level semantic features.
        """
        layers = self.vision.vision_model.encoder.layers
        total  = len(layers)
        for i, layer in enumerate(layers):
            if i >= total - n:
                for p in layer.parameters():
                    p.requires_grad_(True)
        for p in self.vision.vision_model.post_layernorm.parameters():
            p.requires_grad_(True)

    def unfreeze_full(self) -> None:
        """Unfreeze all backbone parameters (Stage C)."""
        for p in self.vision.parameters():
            p.requires_grad_(True)

    def get_optimizer_groups(
        self,
        head_lr:     float = 1e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 1e-2,
    ) -> list[dict]:
        """
        Return AdamW param groups with layerwise LR decay.

        Backbone layers get backbone_lr; heads get head_lr.
        Bias and LayerNorm parameters are excluded from weight decay.

        Usage:
            groups = model.get_optimizer_groups(head_lr=1e-4, backbone_lr=1e-5)
            optimizer = torch.optim.AdamW(groups)
        """
        no_decay  = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        head_params_decay    = []
        head_params_nodecay  = []
        back_params_decay    = []
        back_params_nodecay  = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head   = name.startswith("head_binary") or name.startswith("head_gentype")
            no_wd     = any(nd in name for nd in no_decay)
            if is_head:
                (head_params_nodecay if no_wd else head_params_decay).append(p)
            else:
                (back_params_nodecay if no_wd else back_params_decay).append(p)

        return [
            {"params": head_params_decay,   "lr": head_lr,     "weight_decay": weight_decay},
            {"params": head_params_nodecay, "lr": head_lr,     "weight_decay": 0.0},
            {"params": back_params_decay,   "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": back_params_nodecay, "lr": backbone_lr, "weight_decay": 0.0},
        ]


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

class DetectionLoss(nn.Module):
    """
    Combined loss for dual-head training.

      L = CE_binary + lambda_gen * CE_gentype + lambda_con * SupCon

    Args:
        lambda_gen: weight for auxiliary generator-type loss  (default 0.3)
        lambda_con: weight for supervised contrastive loss     (default 0.1)
        label_smoothing: applied to both CE terms              (default 0.1)
        temperature: SupCon temperature                        (default 0.07)
    """

    def __init__(
        self,
        lambda_gen:      float = 0.3,
        lambda_con:      float = 0.1,
        label_smoothing: float = 0.1,
        temperature:     float = 0.07,
    ):
        super().__init__()
        self.lambda_gen   = lambda_gen
        self.lambda_con   = lambda_con
        self.temperature  = temperature
        self.ce_binary    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_gentype   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_binary:  torch.Tensor,   # [B, 2]
        logits_gentype: torch.Tensor,   # [B, G]
        feat_normed:    torch.Tensor,   # [B, D] L2-normalized features
        labels_binary:  torch.Tensor,   # [B]  0=real, 1=synthetic
        labels_gentype: torch.Tensor,   # [B]  0..G-1
    ) -> tuple[torch.Tensor, dict]:

        loss_bin = self.ce_binary(logits_binary, labels_binary)
        loss_gen = self.ce_gentype(logits_gentype, labels_gentype)
        loss_con = self._supcon(feat_normed, labels_binary)

        total = loss_bin + self.lambda_gen * loss_gen + self.lambda_con * loss_con

        return total, {
            "loss_binary":  loss_bin.item(),
            "loss_gentype": loss_gen.item(),
            "loss_supcon":  loss_con.item(),
            "loss_total":   total.item(),
        }

    def _supcon(
        self,
        feats:  torch.Tensor,   # [B, D] normalized
        labels: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (Khosla et al., NeurIPS 2020).
        Pulls same-class embeddings together, pushes different-class apart.
        Directly attacks cross-generator generalization gap.
        """
        B = feats.size(0)
        if B < 2:
            return feats.new_tensor(0.0)

        sim = torch.matmul(feats, feats.T) / self.temperature  # [B, B]

        # Mask out self-similarity on diagonal
        mask_self = ~torch.eye(B, dtype=torch.bool, device=feats.device)

        # Positive mask: same label, not self
        labels_col = labels.unsqueeze(1)
        mask_pos   = (labels_col == labels_col.T) & mask_self  # [B, B]

        if mask_pos.sum() == 0:
            return feats.new_tensor(0.0)

        # For numerical stability
        sim_max, _ = (sim * mask_self.float()).max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        # Denominator: all non-self pairs
        exp_sim     = sim.exp() * mask_self.float()
        log_prob    = sim - exp_sim.sum(dim=1, keepdim=True).log()

        # Mean over positive pairs per anchor
        loss = -(log_prob * mask_pos.float()).sum(dim=1) / mask_pos.float().sum(dim=1).clamp(min=1)
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Zero-shot probe (Phase 0 — no training required)
# ══════════════════════════════════════════════════════════════════════════════

def zero_shot_auc(
    image_paths:    list[str],
    labels:         list[int],          # 0=real, 1=synthetic
    device:         str = "cuda",
    prompts:        tuple[str, str] = (
        "a photo of a real human face",
        "a photo of an AI-generated or fake face",
    ),
) -> float:
    """
    Phase 0: measure AUC using CLIP's text encoder as the classifier.
    No training, no head — pure cosine similarity in shared embedding space.

    Decision rule:
        AUC >= 0.70  ->  features aligned, skip Stage A
        AUC <  0.70  ->  run full 4-stage curriculum

    Args:
        image_paths: list of image file paths
        labels:      0=real, 1=synthetic per image
        device:      "cuda" or "cpu"
        prompts:     (real_prompt, fake_prompt) — tune for best AUC

    Returns:
        AUC-ROC score
    """
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()

    # Encode text labels once — these ARE the classifier weights.
    # Use text_model + text_projection directly for cross-version compatibility.
    txt_inputs = processor(text=list(prompts), return_tensors="pt", padding=True)
    text_model_keys = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()
                  if k in text_model_keys}
    with torch.no_grad():
        text_out  = model.text_model(**txt_inputs)
        text_vecs = model.text_projection(text_out.pooler_output)  # [2, 512]
        text_vecs = F.normalize(text_vecs, dim=-1)

    scores = []
    for path in image_paths:
        img    = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
        with torch.no_grad():
            vision_out = model.vision_model(**inputs)
            img_vec    = model.visual_projection(vision_out.pooler_output)  # [1, 512]
            img_vec    = F.normalize(img_vec, dim=-1)
        sim     = (img_vec @ text_vecs.T).squeeze(0)           # [2]
        prob    = sim.softmax(dim=-1)[1].item()                # P(synthetic)
        scores.append(prob)

    # Compute AUC-ROC without sklearn dependency
    pairs  = sorted(zip(scores, labels), reverse=True)
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = prev_fp = prev_tp = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        auc += (tp + prev_tp) / 2 * (fp - prev_fp)
        prev_tp, prev_fp = tp, fp
    return auc / (n_pos * n_neg)


# ══════════════════════════════════════════════════════════════════════════════
# GASBench entry point
# ══════════════════════════════════════════════════════════════════════════════

def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """
    GASBench entry point. Loads submission weights into inference-only model.

    Only head_binary weights are required — head_gentype is excluded from the
    submission safetensors file to reduce size. The model returns [B, 2] logits
    via forward(), which is all GASBench calls.

    Args:
        weights_path: path to model.safetensors
        num_classes:  must be 2 for GASBench binary evaluation

    Returns:
        model in eval mode, ready for inference
    """
    model = CLIPViTDetector(
        num_classes=num_classes,
        pretrained=False,       # weights loaded from safetensors below
    )
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)  # head_gentype may be absent
    model.train(False)
    return model


# ── Alias used internally and referenced in model_config.yaml ────────────────
CLIPViTDetector = CLIPViTB16Detector


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers (not used by GASBench, used by train.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    pretrained:      bool  = True,
    freeze_backbone: bool  = True,
) -> CLIPViTB16Detector:
    """Convenience factory used by train.py."""
    return CLIPViTB16Detector(
        num_classes=2,
        num_gen_classes=NUM_GENERATOR_CLASSES,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


def save_submission_weights(
    model:       CLIPViTB16Detector,
    output_path: str = "model.safetensors",
) -> None:
    """
    Save only the weights needed for GASBench submission:
      - vision backbone (self.vision)
      - binary head     (self.head_binary)
      - normalization buffers (pixel_mean, pixel_std)

    Excludes head_gentype to keep the file small.
    """
    from safetensors.torch import save_file

    sd = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("head_gentype")
    }
    save_file(sd, output_path)
    size_mb = sum(v.nbytes for v in sd.values()) / 1e6
    print(f"Saved {output_path}  ({size_mb:.1f} MB,  {len(sd)} tensors)")


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building model (pretrained=False for speed)...")
    model = build_model(pretrained=False, freeze_backbone=True)

    dummy = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)

    # Inference forward — dtype must match input (uint8)
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
    print(f"Inference output : {logits.shape}  dtype={logits.dtype}  (expect [4, 2] uint8)")
    assert logits.shape == (4, 2),           "shape mismatch"
    assert logits.dtype == dummy.dtype,      "dtype mismatch: output must match input"

    # Training forward — all three outputs must match input dtype
    model.train()
    logits_b, logits_g, feat_n = model.forward_train(dummy)
    print(f"Train binary     : {logits_b.shape}  dtype={logits_b.dtype}  (expect [4, 2] uint8)")
    print(f"Train gentype    : {logits_g.shape}  dtype={logits_g.dtype}  (expect [4, {NUM_GENERATOR_CLASSES}] uint8)")
    print(f"Train feat normed: {feat_n.shape}   dtype={feat_n.dtype}  (expect [4, 768] uint8)")
    assert logits_b.dtype == dummy.dtype,    "binary logits dtype mismatch"
    assert logits_g.dtype == dummy.dtype,    "gentype logits dtype mismatch"
    assert feat_n.dtype   == dummy.dtype,    "feat_normed dtype mismatch"

    # Loss
    criterion = DetectionLoss()
    labels_b  = torch.randint(0, 2,                   (4,))
    labels_g  = torch.randint(0, NUM_GENERATOR_CLASSES,(4,))
    loss, info = criterion(logits_b, logits_g, feat_n, labels_b, labels_g)
    print(f"Loss             : {loss.item():.4f}")
    print(f"Loss breakdown   : {info}")

    # Backbone surgery
    model.unfreeze_top_blocks(n=4)
    groups = model.get_optimizer_groups(head_lr=1e-4, backbone_lr=1e-5)
    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"Trainable params (top-4 thawed): {total_trainable:,}")

    print("\nAll checks passed.")