# Third-party source dependencies

The inference pipeline uses external source repositories. They are downloaded
into `third_party/src/`; model weights are stored separately under `models/`.
These components keep their upstream licenses; the top-level Apache-2.0
license does not change those terms. Preserve upstream notices before
redistributing a build.

## Fetch pinned sources

Fetch every source required by the configured model groups:

```bash
bash scripts/fetch_sources.sh
```

Fetch only the sources needed by one group, or validate existing checkouts
without network access:

```bash
bash scripts/fetch_sources.sh --model layout
bash scripts/fetch_sources.sh --check-only
```

The script verifies each pinned commit, required file, and configured recursive
submodule. It tries the official GitHub URL, then `ghproxy.net`, then
`ghfast.top`; Gitee and other mirrors are not used. The authoritative URLs,
commits, required files, and destinations are in
[`configs/sources.json`](../configs/sources.json). Model download commands
prepare the same source dependencies automatically; see
[`docs/models.md`](models.md).

## Manual source download

To install manually, clone the repository in the table below, check out the
`revision` in `configs/sources.json`, and place it at the listed destination.
`TRELLIS.2`, `gsplat`, `CuMesh`, and `FlexGEMM` require recursive submodules.

| Runtime component | Official repository | Destination |
| --- | --- | --- |
| Automatic masks | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | `third_party/src/Grounded-SAM-2` |
| FLUX runtime | [diffusers](https://github.com/huggingface/diffusers) | `third_party/src/diffusers` |
| Object generation (o-voxel included) | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | `third_party/src/TRELLIS.2` |
| Scene placement runtime | [G2VLM](https://github.com/InternRobotics/G2VLM) | `third_party/src/G2VLM` |
| Object conditioning | [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | `third_party/src/sam-3d-objects` |
| Monocular geometry | [MoGe](https://github.com/microsoft/MoGe) | `third_party/src/MoGe` |
| Visual features | [DINOv2](https://github.com/facebookresearch/dinov2) | `third_party/src/dinov2` |
| 3D operators | [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | `third_party/src/pytorch3d` |
| Gaussian rendering | [gsplat](https://github.com/nerfstudio-project/gsplat) | `third_party/src/gsplat` |
| MoGe utilities | [utils3d](https://github.com/EasternJournalist/utils3d) | `third_party/src/utils3d-moge` |
| TRELLIS.2 utilities | [utils3d](https://github.com/EasternJournalist/utils3d) | `third_party/src/utils3d-trellis` |
| Differentiable rasterization | [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | `third_party/src/nvdiffrast` |
| Rendering utilities | [nvdiffrec](https://github.com/JeffreyXiang/nvdiffrec) | `third_party/src/nvdiffrec` |
| CUDA mesh processing | [CuMesh](https://github.com/JeffreyXiang/CuMesh) | `third_party/src/CuMesh` |
| Sparse CUDA kernels | [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) | `third_party/src/FlexGEMM` |

Source code and model terms are independent. Check every upstream license,
including the SAM 3D Objects terms and licenses of CUDA extensions, before
redistribution or commercial use.
