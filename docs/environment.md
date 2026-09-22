# Inference environments

The default inference pipeline uses five isolated Conda environments. They
are separated because the backends require different Python, PyTorch, and CUDA
extension versions.

| Environment | Role | Reference runtime |
| --- | --- | --- |
| `fysiverse-mask` | Grounded-SAM2 automatic masks | Python 3.10, PyTorch 2.5.1/cu121 |
| `fysiverse-flux` | Object cutout completion | Python 3.12, PyTorch 2.5.1/cu124 |
| `fysiverse-trellis2` | Single-object GLB generation | Python 3.10, PyTorch 2.6.0/cu124 |
| `fysiverse-layout` | Fysiverse-3D-Vision placement and assembly | Python 3.10, PyTorch 2.5.1/cu121 |
| `fysiverse-refine` | Post-refinement and metrics | Python 3.10, PyTorch 2.5.0/cu121 |

## Host prerequisites

Install these system dependencies before running the installer:

- Linux x86_64, Conda or Mamba, and Git.
- `gcc`, `g++`, and `make`.
- An NVIDIA driver and CUDA toolkit (`nvcc`) compatible with the selected
  PyTorch wheels.
- Blender 4.5 or newer when post-refinement is enabled.

The scripts check the host, CUDA visibility, compiler tools, Blender, and free
space. They do not install or modify system packages, drivers, CUDA, or
Blender. See the official [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/),
[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and
[Blender](https://www.blender.org/download/lts/4-5/) instructions.

## One-command setup

Create all five environments, fetch the pinned source repositories, install
the role dependencies, and build the required CUDA extensions:

```bash
bash scripts/bootstrap_envs.sh
```

The installer creates new environments only. If a target environment already
exists, it stops before making changes. Use different names when necessary:

```bash
FYSIVERSE_LAYOUT_ENV=fysiverse-layout-new \
FYSIVERSE_MASK_ENV=fysiverse-mask-new \
FYSIVERSE_FLUX_ENV=fysiverse-flux-new \
FYSIVERSE_TRELLIS_ENV=fysiverse-trellis2-new \
FYSIVERSE_REFINE_ENV=fysiverse-refine-new \
bash scripts/bootstrap_envs.sh
```

To install environments and download all configured model artifacts in one
command:

```bash
bash scripts/bootstrap_envs.sh --download-models --model-source auto
```

The default `auto` policy tries the configured upstream provider first and
uses a configured ModelScope route after a retryable network failure. Model
provider details are in [models.md](models.md).

## Install selected roles

Use `--only` when a smaller workflow is sufficient:

```bash
bash scripts/bootstrap_envs.sh --only layout
bash scripts/bootstrap_envs.sh --only mask
bash scripts/bootstrap_envs.sh --only flux
bash scripts/bootstrap_envs.sh --only trellis
bash scripts/bootstrap_envs.sh --only refine
```

A supplied `--mask` skips the automatic-mask role. `--skip-refine` skips the
refinement role and Blender check during inference.

## Dependency manifests

Ordinary Python dependencies are split by process boundary:

| Role | Manifest | Additional installation |
| --- | --- | --- |
| Layout | `requirements/layout.txt` | `utils3d-moge` and the project package |
| Mask | `requirements/mask.txt` | Grounded-SAM2 and GroundingDINO operators |
| FLUX | `requirements/flux.txt` | Pinned Diffusers checkout |
| TRELLIS.2 | `requirements/trellis.txt` | `utils3d-trellis` and CUDA operators |
| Refine | `requirements/refine.txt` | nvdiffrast and refinement tools |

The CUDA helper handles ABI-sensitive packages such as PyTorch3D, gsplat,
flash-attn, CuMesh, FlexGEMM, o-voxel, nvdiffrast, and related operators. For
manual installation, run it only in a newly created target environment and
pass `--target-is-new`.

```bash
bash scripts/install_cuda_extensions.sh \
  --role layout --env fysiverse-layout --target-is-new
```

## Validation

Run the full preflight after installation and model download:

```bash
python scripts/preflight.py
```

For a supplied mask without post-refinement, the mask and refinement roles
can be omitted from the check:

```bash
python scripts/preflight.py --provided-mask --skip-refine
```

Use `--allow-no-gpu` or `--skip-blender` only for dependency preparation or
diagnostics. Those modes do not make the resulting setup ready for the full
GPU inference pipeline.
