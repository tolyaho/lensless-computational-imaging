import torch
import numpy as np
from waveprop.devices import slm_dict

from lensless_helpers.sensor import VirtualSensor
from lensless_helpers.slm import get_programmable_mask, get_intensity_psf


def get_psf(
    vals,
    sensor,
    slm,
    downsample,
    rotate=None,
    flipud=False,
    use_waveprop=False,
    vertical_shift=None,
    horizontal_shift=None,
    scene2mask=None,
    mask2sensor=None,
    deadspace=True,
):
    sensor = VirtualSensor.from_name(sensor, downsample=downsample)
    mask = get_programmable_mask(
        vals=vals,
        sensor=sensor,
        slm_param=slm_dict[slm],
        rotate=rotate,
        flipud=flipud,
        color_filter=None,
        deadspace=deadspace,
    )

    if downsample is not None and vertical_shift is not None:
        vertical_shift = vertical_shift // downsample
    if downsample is not None and horizontal_shift is not None:
        horizontal_shift = horizontal_shift // downsample

    if vertical_shift is not None:
        mask = torch.roll(mask, vertical_shift, dims=1)

    if horizontal_shift is not None:
        mask = torch.roll(mask, horizontal_shift, dims=2)

    psf_in = get_intensity_psf(
        mask=mask,
        sensor=sensor,
        waveprop=use_waveprop,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )

    # add first dimension (depth)
    psf_in = psf_in.unsqueeze(0)

    # move channels to last dimension
    psf_in = psf_in.permute(0, 2, 3, 1)

    # flip mask
    psf_in = torch.flip(psf_in, dims=[-3, -2])

    # normalize
    psf_in = psf_in / psf_in.norm()

    return psf_in


def simulate_psf_from_mask(
    mask_vals,
    sensor="rpi_hq",
    slm="adafruit",
    downsample=8,
    rotate=None,
    flipud=True,
    use_waveprop=True,
    vertical_shift=None,
    horizontal_shift=None,
    scene2mask=0.3,
    mask2sensor=0.002,
    deadspace=True,
    revert_flip=True,
):
    """
    Simulate PSF given mask values.

    Args:
        mask_vals (Tensor): Initial mask parameters.
        sensor (lensless.hardware.sensor.VirtualSensor): Sensor object.
        slm_param : (lensless.hardware.slm.SLMParam): SLM parameters.
        rotate (float): Rotation angle in degrees.
        flipud (bool): Whether to flip the mask vertically.
        use_waveprop (bool): Whether to use wave propagation for simulating
            PSF. If False, PSF will simply be intensity of mask pattern.
        vertical_shift (int): Vertical shift of the mask.
        horizontal_shift (int): Horizontal shift of the mask.
        scene2mask (float): Distance from scene to mask. Used for
            wave propagation.
        mask2sensor (float): Distance from mask to sensor. Used for
            wave propagation.
        downsample (int): Downsample factor.
        deadspace (bool): whether to use deadspace for simulating.
        revert_flip (bool): undo flip from AdafruitLCD.
    """
    psf = get_psf(
        vals=torch.from_numpy(mask_vals.astype(np.float32)),
        sensor=sensor,
        slm=slm,
        downsample=downsample,
        rotate=rotate,
        flipud=flipud,
        use_waveprop=use_waveprop,
        vertical_shift=vertical_shift,
        horizontal_shift=horizontal_shift,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        deadspace=deadspace,
    ).detach()

    # if revert_flip:
    #     psf = torch.flip(psf, dims=[-3, -2])

    return psf
