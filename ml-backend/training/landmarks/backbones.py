"""Allowlisted pretrained backbones for the landmark model (permissive licences only).

    python -m training.landmarks.backbones list
    python -m training.landmarks.backbones fetch [id ...]      # download + sha256-verify
    python -m training.landmarks.backbones prepare-endopos     # once, on the laptop

Every candidate names an exact weight artefact (torchvision weight enum, or a
Hugging Face repo pinned to a commit, or our own re-saved checkpoint) and the
sha256 of that file. ``load_pretrained`` refuses anything not listed here and
any file whose hash differs; there is no generic ``pretrained=True`` path.
The ImageNet heads are dropped; the result is ``architectures.build(arch)``
with pretrained backbone weights and a fresh 10-way head.

    id              arch        weights                                         licence
    effb0_in1k      effb0       torchvision EfficientNet_B0 IMAGENET1K_V1       BSD-3
    r50_in1k        r50         torchvision ResNet50 IMAGENET1K_V2              BSD-3
    regy32_in1k     regy32      torchvision RegNet_Y_3_2GF IMAGENET1K_V2        BSD-3
    cnxt_t_in1k     cnxt_t      torchvision ConvNeXt_Tiny IMAGENET1K_V1         BSD-3
    dinov2_s14      dinov2_s14  timm vit_small_patch14_dinov2.lvd142m @4610ca1  Apache-2.0
    dinov2_b14      (probe)     timm vit_base_patch14_dinov2.lvd142m @4685c99   Apache-2.0
    regy32_endopos  regy32      our endoscopy-position-classification ckpt      MIT (ours)

``dinov2_b14`` is a frozen-probe ceiling only (no serving architecture).
``regy32_endopos`` comes from a checkpoint that also holds optimizer state;
``prepare-endopos`` loads it ONCE with ``weights_only=False`` (our own file),
keeps the backbone and re-saves a plain fp32 state_dict into the gitignored
weights cache. Everything after that loads with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from training.landmarks.common import ML_BACKEND, sha256_file

WEIGHTS_DIR = ML_BACKEND / "training" / "landmarks" / "cache" / "weights"
ENDOPOS_SRC = Path("/Users/steven/code/endoscopy-position-classification/paper_results/best_model.pth")
ENDOPOS_FILE = "regy32_endopos_backbone.pt"
# sha256 of best_model.pth as read on 2026-09-28 (provenance; the clean file is what is pinned).
ENDOPOS_SRC_SHA256 = "10438ec4567ad23e6fc78b9731ffa3040d6598422c1ec4be9930faeacfdad619"


@dataclass(frozen=True)
class Candidate:
    id: str
    arch: str | None  # architectures.build name; None = probe-only
    source: str  # torchvision | hf | local
    weights: str  # torchvision "EnumClass.MEMBER" | HF repo id | local file name
    sha256: str  # of the downloaded / re-saved weight file
    license: str
    embed_dim: int
    revision: str | None = None  # HF commit, pinned
    filename: str | None = None  # file inside the HF repo
    head_prefix: str = ""  # ImageNet head keys dropped before loading
    notes: str = ""

    @property
    def servable(self) -> bool:
        return self.arch is not None


ALLOWLIST: dict[str, Candidate] = {c.id: c for c in (
    Candidate("effb0_in1k", "effb0", "torchvision", "EfficientNet_B0_Weights.IMAGENET1K_V1",
              "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934", "BSD-3-Clause", 1280,
              head_prefix="classifier.1."),
    Candidate("r50_in1k", "r50", "torchvision", "ResNet50_Weights.IMAGENET1K_V2",
              "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca", "BSD-3-Clause", 2048,
              head_prefix="fc."),
    Candidate("regy32_in1k", "regy32", "torchvision", "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2",
              "9180c971ec98fcbe80c86a69724e6738acc91618d526b1172a30fcf0c8f1030f", "BSD-3-Clause", 1512,
              head_prefix="fc."),
    Candidate("cnxt_t_in1k", "cnxt_t", "torchvision", "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
              "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d", "BSD-3-Clause", 768,
              head_prefix="classifier.2."),
    Candidate("dinov2_s14", "dinov2_s14", "hf", "timm/vit_small_patch14_dinov2.lvd142m",
              "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081", "Apache-2.0", 384,
              revision="4610ca143709d58a633b6397a74412c2c3842454", filename="model.safetensors"),
    Candidate("dinov2_b14", None, "hf", "timm/vit_base_patch14_dinov2.lvd142m",
              "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b", "Apache-2.0", 768,
              revision="4685c99dabffe5affac90bd99dbffd25801ae58d", filename="model.safetensors",
              notes="frozen-probe GPU ceiling only; never trained or served"),
    Candidate("regy32_endopos", "regy32", "local", ENDOPOS_FILE,
              "1b78ab53553d1b12d5f1b19d21252fd78e949980c80425977ac75b646a971c12", "MIT (ours)", 1512,
              notes="RegNetY-3.2GF backbone from endoscopy-position-classification best_model.pth"),
)}


class NotAllowed(ValueError):
    """Weights that are not on the allowlist (or fail their hash)."""


def get_candidate(candidate_id: str) -> Candidate:
    try:
        return ALLOWLIST[candidate_id]
    except KeyError:
        raise NotAllowed(f"backbone {candidate_id!r} is not on the allowlist {sorted(ALLOWLIST)}") from None


def _torchvision_weights(c: Candidate):
    import torchvision.models as tvm

    enum_name, member = c.weights.split(".")
    return getattr(getattr(tvm, enum_name), member)


def weights_path(candidate_id: str, download: bool = True) -> Path:
    """Local path of the candidate's weight file, downloaded if needed and verified."""
    c = get_candidate(candidate_id)
    if c.source == "torchvision":
        import torch

        url = _torchvision_weights(c).url
        path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path), progress=False)
    elif c.source == "hf":
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(c.weights, c.filename, revision=c.revision,
                                    local_files_only=not download))
    elif c.source == "local":
        path = WEIGHTS_DIR / c.weights
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m training.landmarks.backbones prepare-endopos`"
                                    " on the laptop and copy cache/weights/ to the GPU box")
    else:  # pragma: no cover - guarded by the allowlist
        raise NotAllowed(f"unknown source {c.source}")
    got = sha256_file(path)
    if got != c.sha256:
        raise NotAllowed(f"{c.id}: sha256 {got} of {path.name} != pinned {c.sha256}")
    return path


def _load_state(c: Candidate, path: Path) -> dict:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def _probe_only_backbone(c: Candidate):
    import timm
    from torch import nn

    m = timm.create_model(c.weights.split("/", 1)[1], pretrained=False, num_classes=0,
                          img_size=224, dynamic_img_size=True, dynamic_img_pad=True)
    assert isinstance(m, nn.Module)
    return m, int(m.num_features)


def load_pretrained(candidate_id: str, allow_probe_only: bool = False):
    """``LandmarkNet`` with allowlisted, hash-verified backbone weights (CPU, eval).

    The head is freshly initialised. Probe-only candidates need
    ``allow_probe_only=True`` and cannot be exported (their ``arch`` is not a
    serving architecture).
    """
    from app.ml.landmarks.architectures import LandmarkNet, build

    c = get_candidate(candidate_id)
    if not c.servable and not allow_probe_only:
        raise NotAllowed(f"{c.id} is a frozen-probe reference only")
    path = weights_path(candidate_id)
    state = _load_state(c, path)

    if c.servable:
        net = build(c.arch)
    else:
        backbone, dim = _probe_only_backbone(c)
        net = LandmarkNet(backbone, dim, 10, c.id)

    if c.source == "hf":
        from timm.models.vision_transformer import checkpoint_filter_fn

        state = checkpoint_filter_fn(state, net.backbone)  # resamples pos_embed to 224
    if c.head_prefix:
        state = {k: v for k, v in state.items() if not k.startswith(c.head_prefix)}
    missing, unexpected = net.backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise NotAllowed(f"{c.id}: weights do not fit the backbone "
                         f"(missing {list(missing)[:5]}, unexpected {list(unexpected)[:5]})")
    return net.eval()


def prepare_endopos(src: Path = ENDOPOS_SRC) -> tuple[Path, str]:
    """Re-save our endoscopy-position RegNetY checkpoint as a clean backbone state_dict."""
    import torch

    from app.ml.landmarks.architectures import build

    # Our own MIT-licensed checkpoint; it pickles optimizer/scheduler state, so
    # this is the ONLY weights_only=False load in the code base.
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    backbone = {k: v.detach().float().clone() if v.is_floating_point() else v.detach().clone()
                for k, v in state.items() if not k.startswith("fc.")}
    ref = build("regy32").backbone.state_dict()
    if set(backbone) != set(ref) or any(backbone[k].shape != ref[k].shape for k in ref):
        raise NotAllowed("endopos checkpoint is not a torchvision regnet_y_3_2gf backbone")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS_DIR / ENDOPOS_FILE
    torch.save(dict(sorted(backbone.items())), out)
    digest = sha256_file(out)
    if hashlib.sha256(src.read_bytes()).hexdigest() != ENDOPOS_SRC_SHA256:
        print(f"warning: {src.name} differs from the checkpoint pinned on 2026-09-28", file=sys.stderr)
    if digest != ALLOWLIST["regy32_endopos"].sha256:
        print(f"warning: re-saved sha256 {digest} != pinned; update the allowlist deliberately",
              file=sys.stderr)
    return out, digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("ids", nargs="*")
    p = sub.add_parser("prepare-endopos")
    p.add_argument("--src", type=Path, default=ENDOPOS_SRC)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for c in ALLOWLIST.values():
            role = "serve" if c.servable else "probe-only"
            print(f"{c.id:15s} {str(c.arch):11s} {role:10s} {c.license:13s} {c.weights}"
                  + (f"@{c.revision[:7]}" if c.revision else ""))
        return 0
    if a.cmd == "fetch":
        for cid in a.ids or list(ALLOWLIST):
            try:
                print(cid, weights_path(cid))
            except (NotAllowed, FileNotFoundError) as exc:
                print(cid, "FAILED:", exc)
        return 0
    out, digest = prepare_endopos(a.src)
    print(f"wrote {out}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
