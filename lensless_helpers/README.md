# lensless_helpers

Sensor, SLM and image utilities for simulating DigiCam point spread functions, adapted from [LenslessPiCam](https://github.com/LCAV/LenslessPiCam) (LCAV, EPFL). LenslessPiCam is licensed under the GNU Affero General Public License v3.0; commercial licences are offered by EPFL's technology transfer office.

| File | Origin | What changed (June 2026) |
|---|---|---|
| `sensor.py` | `lensless/hardware/sensor.py` (upstream commit `9e1e8f1`, 2024-02-23), by Eric Bezzam | Module docstring removed, imports re-pointed. Otherwise unchanged. |
| `slm.py` | `lensless/hardware/slm.py`, by Eric Bezzam | Four functions kept with unchanged bodies; hardware helpers removed; imports re-pointed. |
| `utils.py` | `lensless/utils/image.py` (Eric Bezzam, Julien Sahli) and `lensless/utils/io.py` (Eric Bezzam) | Seven functions merged into one module with unchanged bodies. |
| `psf.py` | `get_psf` adapted from `AdafruitLCD.get_psf` in `lensless/hardware/trainable_mask.py`, by Yohann Perron and Eric Bezzam | Rewritten as a standalone function; `simulate_psf_from_mask` is new (its parameter descriptions follow the upstream docstring). |
| `preprocessor.py` | Written for this repository | Uses the DigiCam dataset's published alignment (top-left `(80, 100)`, height 200). |

Each file names its upstream source, its original authors and the modification in its header, as AGPL-3.0 section 5(a) requires. Wave propagation uses [`waveprop`](https://github.com/ebezzam/waveprop) (MIT, Copyright (c) 2022 Eric Bezzam), installed as a dependency, not copied.

Because this code is AGPL-3.0 and the rest of the repository imports it, the repository as a whole is distributed under the AGPL-3.0; see [`LICENSE`](../LICENSE) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
