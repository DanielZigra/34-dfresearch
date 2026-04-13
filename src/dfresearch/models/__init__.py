"""Baseline deepfake detection models for image, image_human, video, and audio modalities."""

import importlib

_MODEL_PATHS = {
    "image": {
        "efficientnet-b4": ("dfresearch.models.image.efficientnet", "EfficientNetB4Detector"),
        "clip-vit-l14": ("dfresearch.models.image.clip_vit", "CLIPViTDetector"),
        "smogy-swin": ("dfresearch.models.image.smogy_swin", "SMOGYSwinDetector"),
        "siglip-dinov2-ensemble": ("dfresearch.models.image.siglip_dinov2_ensemble", "SigLIPDinov2EnsembleDetector"),
    },
    "video": {
        "r3d-18": ("dfresearch.models.video.r3d", "R3D18Detector"),
        "videomae": ("dfresearch.models.video.videomae", "VideoMAEDetector"),
        "hiera": ("dfresearch.models.video.hiera", "HieraDetector"),
    },
    "audio": {
        "wav2vec2": ("dfresearch.models.audio.wav2vec2", "Wav2Vec2Detector"),
        "ast": ("dfresearch.models.audio.ast_model", "ASTDetector"),
        "wavlm": ("dfresearch.models.audio.wavlm", "WavLMDetector"),
    },
    "image_human": {
        "deepfake-detector-v2": ("dfresearch.models.image_human.deep_fake_detector_v2", "DeepFakeDetectorV2Detector"),
        "airealnet": ("dfresearch.models.image_human.airealnet", "AIRealNetDetector"),
        "real-vs-ai-face": ("dfresearch.models.image_human.real_vs_ai_face", "RealVsAIFaceDetector"),
        "human-faces-ai-vs-real": ("dfresearch.models.image_human.human_faces_ai_vs_real", "HumanFacesAIvsRealDetector"),
        "siglip-dinov2-ensemble": ("dfresearch.models.image_human.siglip_dinov2_ensemble","SigLIPDinov2EnsembleDetector"),
    },
}

MODEL_REGISTRY: dict[str, dict[str, type]] = {}


def _resolve(modality: str, name: str):
    """Lazily import and cache a model class."""
    if modality not in MODEL_REGISTRY:
        MODEL_REGISTRY[modality] = {}
    if name not in MODEL_REGISTRY[modality]:
        module_path, class_name = _MODEL_PATHS[modality][name]
        mod = importlib.import_module(module_path)
        MODEL_REGISTRY[modality][name] = getattr(mod, class_name)
    return MODEL_REGISTRY[modality][name]


def get_model(modality: str, name: str, **kwargs):
    """Instantiate a model by modality and name."""
    if modality not in _MODEL_PATHS:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(_MODEL_PATHS)}")
    if name not in _MODEL_PATHS[modality]:
        raise ValueError(f"Unknown {modality} model: {name}. Choose from {list(_MODEL_PATHS[modality])}")
    cls = _resolve(modality, name)
    return cls(**kwargs)


def list_models(modality: str | None = None) -> dict[str, list[str]]:
    """List available model names, optionally filtered by modality."""
    if modality:
        return {modality: list(_MODEL_PATHS.get(modality, {}))}
    return {m: list(names) for m, names in _MODEL_PATHS.items()}
