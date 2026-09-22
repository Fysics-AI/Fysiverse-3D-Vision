# Model downloads

Model weights are not stored in Git. Downloaded artifacts are written below
`models/`; required source repositories are written below `third_party/src/`.
The authoritative file lists, revisions, dependencies, and integrity checks are
defined in `configs/models.json` and `configs/sources.json`.

## Model sources and destinations

| Group | Official source | Local directory | Access |
| --- | --- | --- | --- |
| `layout` | [Fysics-AI/Fysiverse-3D-Vision](https://huggingface.co/Fysics-AI/Fysiverse-3D-Vision) | `models/layout` | Public |
| `g2vlm` | [InternRobotics/G2VLM-2B-MoT](https://huggingface.co/InternRobotics/G2VLM-2B-MoT) | `models/g2vlm` | Public |
| `grounded_sam2` | [SAM2](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) | `models/grounded_sam2` | Public |
| `bert_base_uncased` | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | `models/bert_base_uncased` | Public |
| `sam3d` | [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) | `models/sam3d` | Gated |
| `moge` | [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl) | `models/moge` | Public |
| `dinov2` | [DINOv2 ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) | `models/dinov2` | Public |
| `flux` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `models/flux/FLUX.2-klein-9B` | Gated |
| `trellis2` | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | `models/trellis2/TRELLIS.2-4B` | Public |
| `trellis_image_large` | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | `models/trellis_image_large` | Public |
| `dinov3_vitl16` | [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `models/dinov3_vitl16` | Provider terms apply |
| `rmbg2` | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | `models/rmbg2` | Provider terms apply |

Review every upstream model card and license before use or redistribution.
SAM 3D Objects and FLUX require accepting their provider terms before
download.

## Download commands

Download all configured model groups and their required source repositories:

```bash
bash scripts/download_all_models.sh --model-source auto
```

Download one group and its transitive model/source dependencies:

```bash
bash scripts/download_model.sh layout
bash scripts/download_model.sh grounded_sam2
bash scripts/download_model.sh flux
bash scripts/download_model.sh trellis2
```

Valid group names are `layout`, `g2vlm`, `grounded_sam2`,
`bert_base_uncased`, `sam3d`, `moge`, `dinov2`, `flux`, `trellis2`,
`trellis_image_large`, `dinov3_vitl16`, and `rmbg2`.

The `layout` selection also resolves its runtime model and source dependencies.
The `trellis2` selection includes its image decoder, DINOv3, and background
removal dependencies. A full environment-and-model setup is also available:

```bash
bash scripts/bootstrap_envs.sh --download-models --model-source auto
```

## Authentication

Public repositories normally download without credentials. For a gated
Hugging Face repository, accept its terms and authenticate with an access
token:

```bash
huggingface-cli login
# Newer installations may also provide:
# hf auth login
```

Alternatively, set `HF_TOKEN` for the download process. Do not place tokens in
configuration files or commit them to Git. ModelScope credentials, when
required, can be supplied through `MODELSCOPE_API_TOKEN`.

## Provider selection

`--model-source` supports three modes:

| Mode | Behavior |
| --- | --- |
| `auto` | Try upstream first; use a configured ModelScope route after a retryable network failure. |
| `upstream` | Use only the configured Hugging Face repository or official file URL. |
| `modelscope` | Use configured ModelScope repositories and stop if an incomplete dependency has no mapping. |

Authentication, permission, and license-acceptance errors do not trigger a
fallback. ModelScope applies to model artifacts only. Source repositories use
the pinned Git routes and commits in `configs/sources.json`; the downloader
validates the revision and required files before use.

## Manual download

Models may be downloaded manually from the official links above, but their
files must be placed in the exact local directories shown in the table.
Official source links and checkout destinations are listed in
[`third-party-sources.md`](third-party-sources.md).

After a manual download, validate the complete model tree without network
access:

```bash
python scripts/download_models.py \
  --model all --check-only --skip-source-fetch
python scripts/validate_layout_model.py --model-dir models/layout
```

The validation checks required paths, file sizes, hashes where configured, and
the Fysiverse-3D-Vision package structure. The downloader checks free disk
space before transfer and does not delete existing files.
