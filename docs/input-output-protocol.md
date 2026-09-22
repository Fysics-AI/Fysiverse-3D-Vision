# Input and output protocol

## Inputs

The public runner accepts one indoor RGB image. A mask is optional:

```text
input/
├── room.png
├── mask.png       # optional instance mask
└── camera.json    # required when refinement is enabled
```

Run with a supplied mask:

```bash
bash scripts/run_inference.sh \
  --image input/room.png \
  --mask input/mask.png \
  --camera input/camera.json \
  --output result/room
```

If `--mask` is omitted, Grounded-SAM2 creates the instance masks. A supplied
mask must have the same width and height as the RGB image. The input may be a
color-coded instance mask or a grayscale/alpha foreground mask; the prepare
stage writes one-channel masks as `masks/0.png`, `masks/1.png`, and so on.

## Camera calibration

Post-refinement is enabled by default and requires the camera corresponding to
the input photograph. The JSON contains a 4 x 4 `camera_to_world` matrix, a
3 x 3 `intrinsic` matrix, positive image dimensions, and the Y-up scene
convention:

```json
{
  "source": "input_calibration",
  "camera": {
    "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "intrinsic": [[700, 0, 320], [0, 700, 240], [0, 0, 1]],
    "width": 640,
    "height": 480
  },
  "coordinate_system": {"scene_up_axis": "+Y"}
}
```

Use `--skip-refine` when no calibrated camera is available. A generated
preview camera is not an input-image calibration and must not be used for
quantitative alignment.

## Outputs

```text
result/room/
├── manifest.json
├── mask.png
├── masks/0.png
├── object_assets/0/model.glb
├── poses.json
├── scene.glb
├── scene_refined.glb     # when post-refinement is enabled
└── post_refine/          # refinement reports and intermediate files
```

`poses.json` records one transform for each object:

```json
{
  "id": "0",
  "category": "chair",
  "mask": "masks/0.png",
  "asset": "object_assets/0/model.glb",
  "translation": [0.0, 0.0, 0.0],
  "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "scaling": 1.0
}
```

The layout uses normalized scene units rather than verified metric units. The
exported GLB is a visual mesh assembly; it does not by itself define validated
rigid bodies, collision shapes, mass, inertia, joints, or simulator behavior.

The machine-readable manifest definition is in
[`manifest.schema.json`](manifest.schema.json).
