import torch
import numpy as np
import cv2
import os

try:
    import torch
    import torchvision.transforms as tf
    from torchvision.transforms.functional import rgb_to_grayscale, rotate

    torch_available = True
except ImportError:
    torch_available = False


RPI_HQ_CAMERA_CCM_MATRIX = np.array(
    [
        [2.0659, -0.93119, -0.13421],
        [-0.11615, 1.5593, -0.44314],
        [0.073694, -0.4368, 1.3636],
    ]
)
RPI_HQ_CAMERA_BLACK_LEVEL = 256.3
SUPPORTED_BIT_DEPTH = np.array([8, 10, 12, 16])
FLOAT_DTYPES = [np.float32, np.float64]


def rgb2gray(rgb, weights=None, keepchanneldim=True):
    """
    Convert RGB array to grayscale.

    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
        ([Depth,] Height, Width, Channel) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale. Only used for NumPy arrays.
    keepchanneldim : bool
        Whether to keep the channel dimension. Default is True.

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Grayscale image of dimension ([depth,] height, width [, 1]).

    """

    use_torch = False
    if torch_available:
        if torch.is_tensor(rgb):
            use_torch = True

    if use_torch:

        # move channel dimension to third to last
        if len(rgb.shape) == 4:
            rgb = rgb.permute(0, 3, 1, 2)
        elif len(rgb.shape) == 3:
            rgb = rgb.permute(2, 0, 1)
        else:
            raise ValueError("Input must be at least 3D.")

        image = rgb_to_grayscale(rgb)

        # move channel dimension to last
        if len(rgb.shape) == 4:
            image = image.permute(0, 2, 3, 1)
        elif len(rgb.shape) == 3:
            image = image.permute(1, 2, 0)

        if not keepchanneldim:
            image = image.squeeze(-1)

        return image

    else:

        if weights is None:
            weights = np.array([0.299, 0.587, 0.114])
        assert len(weights) == 3

        if len(rgb.shape) == 4:
            image = np.tensordot(rgb, weights, axes=((3,), 0))
        elif len(rgb.shape) == 3:
            image = np.tensordot(rgb, weights, axes=((2,), 0))
        else:
            raise ValueError("Input must be at least 3D.")

        if keepchanneldim:
            return image[..., np.newaxis]
        else:
            return image


def load_image(
    fp,
    verbose=False,
    flip=False,
    flip_ud=False,
    flip_lr=False,
    bayer=False,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    blue_gain=None,
    red_gain=None,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    back=None,
    nbits_out=None,
    as_4d=False,
    downsample=None,
    bg=None,
    return_float=False,
    shape=None,
    dtype=None,
    normalize=True,
    bgr_input=True,
):
    """
    Load image as numpy array.

    Parameters
    ----------
    fp : str
        Full path to file.
    verbose : bool, optional
        Whether to plot into about file.
    flip : bool
        Whether to flip data (vertical and horizontal).
    bayer : bool
        Whether input data is Bayer.
    blue_gain : float
        Blue gain for color correction.
    red_gain : float
        Red gain for color correction.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    back : array_like
        Background level to subtract.
    nbits_out : int
        Output bit depth. Default is to use that of input.
    as_4d : bool
        Add depth and color dimensions if necessary so that image is 4D: (depth,
        height, width, color).
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    bg : array_like
        Background level to subtract.
    return_float : bool
        Whether to return image as float array, or unsigned int.
    shape : tuple, optional
        Shape (H, W, C) to resize to.
    dtype : str, optional
        Data type of returned data. Default is to use that of input.
    normalize : bool, default True
        If ``return_float``, whether to normalize data to maximum value of 1.

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)

    nbits = None  # input bit depth
    if "dng" in fp:
        import rawpy

        assert bayer
        raw = rawpy.imread(fp)
        img = raw.raw_image
        # # # TODO : use raw.postprocess? to much unknown processing...
        # img = raw.postprocess(
        #     adjust_maximum_thr=0,  # default 0.75
        #     no_auto_scale=False,
        #     # no_auto_scale=True,
        #     gamma=(1, 1),
        #     bright=1,  # default 1
        #     exp_shift=1,
        #     no_auto_bright=True,
        #     # use_camera_wb=True,
        #     # use_auto_wb=False,
        #     # -- gives better balance for PSF measurement
        #     use_camera_wb=False,
        #     use_auto_wb=True,  # default is False? f both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        # )

        # if red_gain is None or blue_gain is None:
        #     camera_wb = raw.camera_whitebalance
        #     red_gain = camera_wb[0]
        #     blue_gain = camera_wb[1]

        nbits = int(np.ceil(np.log2(raw.white_level)))
        ccm = raw.color_matrix[:, :3]
        black_level = np.array(raw.black_level_per_channel[:3]).astype(np.float32)
    elif "npy" in fp or "npz" in fp:
        img = np.load(fp)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if bayer:
        assert len(img.shape) == 2, img.shape
        if nbits is None:
            if img.max() > 255:
                # HQ camera
                nbits = 12
            else:
                nbits = 8

        if back:
            back_img = cv2.imread(back, cv2.IMREAD_UNCHANGED)
            dtype = img.dtype
            img = img.astype(np.float32) - back_img.astype(np.float32)
            img = np.clip(img, a_min=0, a_max=img.max())
            img = img.astype(dtype)
        if nbits_out is None:
            nbits_out = nbits

        img = bayer2rgb_cc(
            img,
            nbits=nbits,
            blue_gain=blue_gain,
            red_gain=red_gain,
            black_level=black_level,
            ccm=ccm,
            nbits_out=nbits_out,
        )

    else:
        if len(img.shape) == 3 and bgr_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_dtype = img.dtype

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)

    if bg is not None:

        # if bg is float vector, turn into int-valued vector
        if bg.max() <= 1 and img.dtype not in [np.float32, np.float64]:
            bg = bg * get_max_val(img)

        img = img - bg
        img = np.clip(img, a_min=0, a_max=img.max())

    if as_4d:
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        elif len(img.shape) == 2:
            img = img[np.newaxis, :, :, np.newaxis]

    if downsample is not None or shape is not None:
        if downsample is not None:
            factor = 1 / downsample
        else:
            factor = None
        img = resize(img, factor=factor, shape=shape)

    if return_float:
        if dtype is None:
            dtype = np.float32
        assert dtype == np.float32 or dtype == np.float64
        img = img.astype(dtype)
        if normalize:
            img /= img.max()

    else:
        if dtype is None:
            dtype = original_dtype
        img = img.astype(dtype)

    if verbose:
        print_image_info(img)

    return img


def get_max_val(img, nbits=None):
    """
    For uint image.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image array.
    nbits : int, optional
        Number of bits per pixel. Detect if not provided.

    Returns
    -------
    max_val : int
        Maximum pixel value.
    """
    assert img.dtype not in FLOAT_DTYPES
    if nbits is None:
        nbits = int(np.ceil(np.log2(img.max())))

    if nbits not in SUPPORTED_BIT_DEPTH:
        nbits = SUPPORTED_BIT_DEPTH[nbits < SUPPORTED_BIT_DEPTH][0]
    max_val = 2**nbits - 1
    if img.max() > max_val:
        new_nbit = int(np.ceil(np.log2(img.max())))
        print(f"Detected pixel value larger than {nbits}-bit range, using {new_nbit}-bit range.")
        max_val = 2**new_nbit - 1
    return max_val


def bayer2rgb_cc(
    img,
    nbits,
    down=None,
    blue_gain=None,
    red_gain=None,
    black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
    ccm=RPI_HQ_CAMERA_CCM_MATRIX,
    nbits_out=None,
):
    """
    Convert raw Bayer data to RGB with the following steps:

    #. Demosaic with bi-linear interpolation, mapping the Bayer array to RGB.
    #. Black level removal.
    #. White balancing, applying gains to red and blue channels.
    #. Color correction matrix.
    #. Clip.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        2D Bayer data to convert to RGB.
    nbits : int
        Bit depth of input data.
    blue_gain : float
        Blue gain.
    red_gain : float
        Red gain.
    black_level : float
        Black level. Default is to use that of Raspberry Pi HQ camera.
    ccm : :py:class:`~numpy.ndarray`
        Color correction matrix. Default is to use that of Raspberry Pi HQ camera.
    nbits_out : int
        Output bit depth. Default is to use that of input.

    Returns
    -------
    rgb : :py:class:`~numpy.ndarray`
        RGB data.
    """
    assert len(img.shape) == 2, img.shape
    if nbits_out is None:
        nbits_out = nbits
    if nbits_out > 8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    # demosaic Bayer data
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

    # downsample
    if down is not None:
        img = resize(img[None, ...], factor=1 / down, interpolation=cv2.INTER_CUBIC)[0]

    # correction
    img = img - black_level
    if red_gain:
        img[:, :, 0] *= red_gain
    if blue_gain:
        img[:, :, 2] *= blue_gain
    img = img / (2**nbits - 1 - black_level)
    img[img > 1] = 1

    img = (img.reshape(-1, 3, order="F") @ ccm.T).reshape(img.shape, order="F")
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * (2**nbits_out - 1)).astype(dtype)


def print_image_info(img):
    """
    Print dimensions, data type, max, min, mean.
    """
    print("dimensions : {}".format(img.shape))
    print("data type : {}".format(img.dtype))
    print("max  : {}".format(img.max()))
    print("min  : {}".format(img.min()))
    print("mean : {}".format(img.mean()))


def resize(img, factor=None, shape=None, interpolation=cv2.INTER_CUBIC):
    """
    Resize by given factor.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Image to downsample
    factor : int or float
        Resizing factor.
    shape : tuple
        Shape to copy ([depth,] height, width, color). If provided, (height, width) is used.
    interpolation : OpenCV interpolation method
        See https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    img_shape = np.array(img.shape)[-3:-1]

    assert not ((factor is None) and (shape is None)), "Must specify either factor or shape"
    new_shape = tuple(img_shape * factor) if shape is None else shape[-3:-1]
    new_shape = [int(i) for i in new_shape]

    if np.array_equal(img_shape, new_shape):
        return img

    if torch_available:
        # torch resize expects an input of form [color, depth, width, height]
        tmp = np.moveaxis(img, -1, 0)
        tmp = torch.from_numpy(tmp.copy())
        resized = tf.Resize(size=new_shape, antialias=True)(tmp).numpy()
        resized = np.moveaxis(resized, 0, -1)

    else:
        resized = np.array(
            [
                cv2.resize(img[i], dsize=tuple(new_shape[::-1]), interpolation=interpolation)
                for i in range(img.shape[-4])
            ]
        )
        # OpenCV discards channel dimension if it is 1, put it back
        if len(resized.shape) == 3:
            # resized = resized[:, :, :, np.newaxis]
            resized = np.expand_dims(resized, axis=-1)

    return np.clip(resized, min_val, max_val)


def get_ctypes(dtype, is_torch):
    if not is_torch:
        if dtype == np.float32 or dtype == np.complex64:
            return np.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return np.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
    else:
        import torch

        if dtype == np.float32 or dtype == np.complex64:
            return torch.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return torch.complex128, np.complex128
        elif dtype == torch.float32 or dtype == torch.complex64:
            return torch.complex64, np.complex64
        elif dtype == torch.float64 or dtype == torch.complex128:
            return torch.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
