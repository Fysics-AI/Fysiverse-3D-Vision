# Fysiverse-3D-Vision model package

The Hugging Face artifact is an inference package. It contains the
scene-placement weights and metadata; the runtime architecture and feature
providers are installed separately by the public inference repository.

## Package contents

```text
models/layout/
├── config.json
├── backbone.safetensors.index.json
├── backbone-00001-of-00004.safetensors
├── backbone-00002-of-00004.safetensors
├── backbone-00003-of-00004.safetensors
├── backbone-00004-of-00004.safetensors
├── layout.safetensors
├── manifest.json
├── checksums.json
├── README.md
├── LICENSE
└── NOTICE
```

Do not replace this package with an external `model.pt` file. The public
loader reads the SafeTensors shards and the separate `layout.safetensors`
file, then applies the declared scene-placement tensors to the runtime
architecture.

## Configuration

`config.json` is the runtime contract. It declares:

- `model_type` and `format_version`;
- the base architecture package, immutable revision, and weight filename;
- the backbone shard index and tensor count;
- the layout weight filename, tensor count, and sections;
- output heads (`translation`, `rotation`, `scaling`), rotation dimension, and
  scene-coordinate conventions.

The package is loaded from `models/layout` by default. Keep the files at the
package root so the index and checksum paths remain valid.

## Validation

Run the validator after downloading or uploading a package:

```bash
python scripts/validate_layout_model.py \
  --model-dir models/layout
```

The validator checks required files, JSON metadata, shard/index mappings,
tensor counts, scene-placement keys, and forbidden absolute or private paths.
It reads SafeTensors headers without loading the full model into memory.
