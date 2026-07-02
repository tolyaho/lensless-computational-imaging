# Third-party notices

This repository is distributed under the GNU AGPL v3.0 (see [`LICENSE`](LICENSE)). It contains code from the projects below; their notices are kept here as their licences require. Overlap figures are the share of non-trivial source lines also found in the upstream file, measured on 2026-10-29.

## LenslessPiCam (GNU AGPL v3.0)

<https://github.com/LCAV/LenslessPiCam>, LCAV, EPFL. Authors of the adapted files: Eric Bezzam, Julien Sahli, Yohann Perron.

`lensless_helpers/sensor.py`, `slm.py`, `utils.py` and `psf.py` are adapted from LenslessPiCam. Each file states its upstream source, the original authors and what was modified (June 2026). The per-file table is in [`lensless_helpers/README.md`](lensless_helpers/README.md). The AGPL v3.0 text is in [`LICENSE`](LICENSE).

## pytorch_project_template (MIT)

<https://github.com/Blinorot/pytorch_project_template>. The training and logging scaffold follows this template:

| File in this repository | Template file | Overlap |
|---|---|---|
| `src/logger/logger.py` | `src/logger/logger.py` | 100% |
| `src/utils/io_utils.py` | `src/utils/io_utils.py` | 100% |
| `src/utils/init_utils.py` | `src/utils/init_utils.py` | 98% |
| `src/datasets/base_dataset.py` | `src/datasets/base_dataset.py` | 91% |
| `src/logger/utils.py` | `src/logger/utils.py` | 90% |
| `src/trainer/base_trainer.py` | `src/trainer/base_trainer.py` | 81% |
| `src/logger/noop.py` | `src/logger/wandb.py` (structure) | 80% |
| `src/datasets/data_utils.py` | `src/datasets/data_utils.py` | 74% |
| `train.py` | `train.py` | 58% |
| `src/metrics/tracker.py`, `src/trainer/trainer.py`, `src/datasets/collate.py` | same paths | 17–27% (partly) |

```
MIT License

Copyright (c) 2024 Petr Grinberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data (not copied into the repository)

- [`waveprop`](https://github.com/ebezzam/waveprop), MIT, Copyright (c) 2022 Eric Bezzam. It is installed as a dependency; no code is copied (overlap below 15% for every file).
- Dataset [`bezzam/DigiCam-Mirflickr-MultiMask-10K`](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K), MIT according to its dataset card. It is downloaded at run time. The scenes are MIRFLICKR photographs, which were distributed under Creative Commons licences; five of them appear in the comparison grids under `reports/assets/`.
