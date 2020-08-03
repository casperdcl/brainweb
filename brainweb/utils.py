"""General helper functions."""
from __future__ import division
import numpy as np
from numpy.random import seed
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from tqdm.auto import tqdm
import functools
import requests
import re
import os
import gzip
from os import path
import logging

__author__ = "Casper O. da Costa-Luis <casper.dcl@physics.org>"
__date__ = "2017-20"
__licence__ = __license__ = "[MPLv2.0](https://www.mozilla.org/MPL/2.0)"
__all__ = [
    # necessary
    "volshow", "get_files", "get_mmr_fromfile",
    # useful utils
    "get_file", "load_file", "gunzip_array", "ellipsoid", "add_lesions",
    # nothing to do with BrainWeb but still useful
    "register",
    # intensities
    "FDG", "Amyloid", "T1", "T2",
    # scanner params
    "Res", "Shape",
    # probably not needed
    "noise", "seed", "toPetMmr", "LINKS"]

LINKS = "04 05 06 18 20 38 41 42 43 44 45 46 47 48 49 50 51 52 53 54"
LINKS = [
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1' +
    '?do_download_alias=subject' + i + '_crisp&format_value=raw_short' +
    '&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D'
    for i in LINKS.split()]
RE_SUBJ = re.compile('.*(subject)([0-9]+).*')
LINKS = dict((RE_SUBJ.sub(r'\1_\2.bin.gz', i), i) for i in LINKS)


class Act(object):
  """careful: occasionally other bits may be set"""
  background, csf, greyMatter, whiteMatter, fat, muscle, skin, skull, vessels,\
      aroundFat, dura, marrow\
      = [i << 4 for i in range(12)]
  bone = skull | marrow | dura

  @classmethod
  def indices(cls, im, attr):
    matter = None
    if attr == "bone":
      matter = ["skull", "marrow", "dura"]
    elif attr == "tissue":
      matter = ["whiteMatter", "greyMatter", "skin", "csf", "fat", "muscle",
                "vessels", "aroundFat"]

    if matter:
      return sum((cls.indices(im, i) for i in matter[1:]),
                 cls.indices(im, matter[0])) > 0

    return abs(im - getattr(cls, attr)) < 1


class FDG(Act):
  whiteMatter = 32
  greyMatter = whiteMatter * 4
  skin = whiteMatter // 2
  hot = greyMatter * 1.5
  cold = whiteMatter * 0.5
  attrs = ["whiteMatter", "greyMatter", "skin"]


Pet = FDG  # backward-compat


class Amyloid(Act):
  whiteMatter = 29
  greyMatter = 66
  skin = 35
  hot = greyMatter * 1.5
  cold = whiteMatter * 0.5
  attrs = ["whiteMatter", "greyMatter", "skin"]


class T1(Act):
  whiteMatter = 154
  greyMatter = 106
  skin = 92
  skull = 48
  marrow = 180
  bone = 48
  csf = 48
  hot = greyMatter * 1.5
  cold = whiteMatter * 0.8
  attrs = ["whiteMatter", "greyMatter", "skin", "skull", "marrow", "bone",
           "csf"]


class T2(T1):
  whiteMatter = 70
  greyMatter = 100
  skin = 70
  skull = 100
  marrow = 250
  csf = 250
  bone = 200
  hot = greyMatter * 1.5
  cold = whiteMatter * 0.8


class Mu(Act):
  bone = 0.13
  tissue = 0.0975
  attrs = ["bone", "tissue"]


mu_bone_1_cm = Mu.bone  # backward-compat
mu_tissue_1_cm = Mu.tissue  # backward-compat


class Res(object):
  """in mm"""
  mMR = np.array([2.0312, 2.0863, 2.0863])
  MR = np.array([1.0, 1.0, 1.0])
  brainweb = np.array([0.5, 0.5, 0.5])


class Shape(object):
  """in voxels"""
  mMR = np.array([127, 344, 344])
  MR = mMR * Res.mMR / Res.MR
  brainweb = mMR * Res.mMR / Res.brainweb


def get_file(fname, origin, cache_dir=None):
    """
    Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.brainweb`, and given the filename `fname`.
    The final location of a file
    `subject_04.bin.gz` would therefore be `~/.brainweb/subject_04.bin.gz`.
    Vaguely based on:
    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py

    @param fname  : Name of the file. If an absolute path
        `/path/to/file.txt` is specified the file will be saved at that
        location.
    @param origin  : Original URL of the file.
    @param cache_dir  : Location to store cached files, when None it
        defaults to `~/.brainweb`.
    @return  : Path to the downloaded file
    """
    log = logging.getLogger(__name__)

    if cache_dir is None:
        cache_dir = path.join('~', '.brainweb')
    cache_dir = path.expanduser(cache_dir)
    if not path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except Exception:
            log.warn("cannot create:" + cache_dir)
    if not os.access(cache_dir, os.W_OK):
        cache_dir = path.join('/tmp', '.brainweb')
        if not path.exists(cache_dir):
            os.makedirs(cache_dir)

    fpath = path.join(cache_dir, fname)

    if not path.exists(fpath):
        log.debug("Downloading %s from %s" % (fpath, origin))
        try:
            d = requests.get(origin, stream=True)
            with tqdm(total=d.headers.get('Content-length', None), desc=fname,
                      unit="B", unit_scale=True, unit_divisor=1024,
                      leave=False) as fprog:
                with open(fpath, 'wb') as fo:
                    for chunk in d.iter_content(chunk_size=None):
                        fo.write(chunk)
                        fprog.update(len(chunk))
                fprog.total = fprog.n
                fprog.refresh()
        except (Exception, KeyboardInterrupt):
            if path.exists(fpath):
                os.remove(fpath)
            raise

    return fpath


def gunzip_array(fpath, shape=None, dtype=None):
    """
    Uncompress the specified file and read the binary output as an array.
    """
    with gzip.open(fpath) as fi:
        data = np.frombuffer(fi.read(), dtype=dtype)
    if shape is not None:
        return data.reshape(shape)


load_file = functools.partial(
    gunzip_array, shape=(362, 434, 362), dtype=np.uint16)


def get_files(cache_dir=None, progress=True):
    """
    Returns list of files which can be `numpy.load`ed
    """
    files = []
    for f, url in tqdm(LINKS.items(), unit="file", desc="BrainWeb Subjects",
                       disable=not progress):
        files.append(get_file(f, url, cache_dir=cache_dir))
    return sorted(files)


def get_mmr(cache_file, raw_data,
            petNoise=1.0, t1Noise=0.75, t2Noise=0.75,
            petSigma=1.0, t1Sigma=1.0, t2Sigma=1.0,
            PetClass=FDG):
    """
    Return contents of specified `*.npz` file,
    creating it from BrainWeb `raw_data` 3darray if it doesn't exist.
    """
    if not path.exists(cache_file):
        pet, uMap, t1, t2 = toPetMmr(raw_data, PetClass=PetClass)
        pet = noise(pet, petNoise, sigma=petSigma)[:, ::-1]
        t1 = noise(t1, t1Noise, sigma=t1Sigma)[:, ::-1]
        t2 = noise(t2, t2Noise, sigma=t2Sigma)[:, ::-1]
        uMap = uMap[:, ::-1]
        np.savez_compressed(
            cache_file,
            PET=pet, uMap=uMap, T1=t1, T2=t2,
            petNoise=np.float32(petNoise),
            t1Noise=np.float32(t1Noise),
            t2Noise=np.float32(t2Noise),
            petSigma=np.float32(petSigma),
            t1Sigma=np.float32(t1Sigma),
            t2Sigma=np.float32(t2Sigma))

    return np.load(cache_file, allow_pickle=True)


def get_mmr_fromfile(brainweb_file,
                     petNoise=1.0, t1Noise=0.75, t2Noise=0.75,
                     petSigma=1.0, t1Sigma=1.0, t2Sigma=1.0,
                     PetClass=FDG):
    """
    mMR resolution ground truths from a cached `np.load`able file generated
    from `brainweb_file`.
    """
    dat = load_file(brainweb_file)  # read raw data
    return get_mmr(
        brainweb_file.replace(
            '.bin.gz', '.npz' if PetClass == FDG else
            '.{}.npz'.format(PetClass.__name__)),
        dat,
        petNoise=1.0, t1Noise=0.75, t2Noise=0.75,
        petSigma=1.0, t1Sigma=1.0, t2Sigma=1.0,
        PetClass=PetClass)


def volshow(vol,
            cmaps=None, colorbars=None,
            xlabels=None, ylabels=None, titles=None,
            vmins=None, vmaxs=None,
            sharex=True, sharey=True,
            ncols=None, nrows=None,
            figsize=None, frameon=True, tight_layout=1,
            fontproperties=None):
    """
    Interactively slice through 3D array(s) in Jupyter

    @param vol  : imarray or [imarray, ...] or {'title': imarray, ...}
      Note that imarray may be 3D (mono) or 4D (last channel rgb(a))
    @param cmaps  : list of cmap [default: ["Greys_r", ...]]
    @param xlabels, ylabels, titles  : list of strings (default blank)
    @param vmins, vmaxs  : list of numbers [default: [None, ...]]
    @param colorbars  : list of bool [default: [False, ...]]
    @param sharex, sharey, ncols, nrows  : passed to
      `matplotlib.pyplot.subplots`
    @param figsize, frameon  : passed to `matplotlib.pyplot.figure`
    @param tight_layout  : number of times to run `tight_layout(0, 0, 0)`
      [default: 1]
    """
    import matplotlib.pyplot as plt
    import ipywidgets as ipyw
    log = logging.getLogger(__name__)

    if hasattr(vol, "keys") and hasattr(vol, "values"):
        if titles is not None:
            log.warn("ignoring `vol.keys()` in favour of specified `titles`")
        else:
            titles = vol.keys()
            vol = list(vol.values())

    if vol[0].ndim == 2:  # single 3darray
        vol = [vol]
    else:
        for v in vol:
            if v.ndim not in [3, 4]:
                raise IndexError("Input should be (one or a list of)" +
                                 " 3D and/or 4D array(s)")

    if cmaps is None:
        cmaps = ["Greys_r"] * len(vol)
    if colorbars is None:
        colorbars = [False] * len(vol)
    if xlabels is None:
        xlabels = [""] * len(vol)
    if ylabels is None:
        ylabels = [""] * len(vol)
    if titles is None:
        titles = [""] * len(vol)
    if vmins is None:
        vmins = [None] * len(vol)
    if vmaxs is None:
        vmaxs = [None] * len(vol)
    if tight_layout in (True, False):
        tight_layout = 1 if tight_layout else 0

    # automatically square-ish grid, slightly favouring more rows
    if nrows:
        rows = nrows
        cols = ncols or int(np.ceil(len(vol) / rows))
    else:
        cols = ncols or max(1, int(len(vol) ** 0.5))
        rows = int(np.ceil(len(vol) / cols))
    # special case
    if not (nrows or ncols) and len(vol) == 4:
        nrows = ncols = 2

    zSize = min(len(i) for i in vol)
    fig = plt.figure(figsize=figsize, frameon=frameon)

    @ipyw.interact(z=ipyw.IntSlider(zSize // 2, 0, zSize - 1, 1))
    def plot_slice(z):
        """z  : int, slice index"""
        plt.figure(fig.number, clear=True)
        axs = fig.subplots(rows, cols, sharex=sharex, sharey=sharey)
        axs = list(getattr(axs, 'flat', [axs]))
        for ax, v, cmap, cbar, xlab, ylab, tit, vmin, vmax in zip(
                axs, vol, cmaps, colorbars,
                xlabels, ylabels, titles,
                vmins, vmaxs):
            plt.sca(ax)
            plt.imshow(v[z], cmap=cmap, vmin=vmin, vmax=vmax)
            if cbar:
                plt.colorbar()
            textargs = {}
            if fontproperties is not None:
                textargs.update(fontproperties=fontproperties)
            if xlab:
                plt.xlabel(xlab, **textargs)
            if ylab:
                plt.ylabel(ylab, **textargs)
            if tit:
                plt.title(tit, **textargs)
            plt.show()
            if not frameon:
                plt.setp(ax.spines.values(), color='white')
                #  don't need all axes if sharex=sharey=True
                ax.set_xticks(())
                ax.set_yticks(())
        for _ in range(tight_layout):
            plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # make sure to clear extra axes
        for ax in axs[axs.index(ax) + 1:]:
            ax.axis('off')
        #return fig, axs

    return plot_slice


def getRaw(fname):
  """z, y, x"""
  return np.fromfile(fname, dtype=np.uint16).reshape((362, 434, 362))


def noise(im, n, warn_zero=True, sigma=1):
  """
  @param n  : float, noise fraction (0, inf)
  @param sigma  : float, smoothing of noise component
  @return[out] im  : array like im with +-n *100%im noise added
  """
  log = logging.getLogger(__name__)
  if n < 0:
    raise ValueError("Noise must be positive")
  elif n == 0:
    if warn_zero:
      log.warn("zero noise")
    return im
  r = gaussian_filter(np.random.random(im.shape), sigma=sigma, mode='constant')
  return im * (1 + n * (2 * r - 1))


def toPetMmr(im, pad=True, dtype=np.float32, outres="mMR", modes=None,
             PetClass=FDG):
  """
  @param modes  : [default: [PetClass, Mu, T1, T2]]
  @return out  : list of `modes`, each shape [127, 344, 344]
  """
  log = logging.getLogger(__name__)

  out_res = getattr(Res, outres)
  out_shape = getattr(Shape, outres)

  if modes is None:
    modes = [PetClass, Mu, T1, T2]

  # PET
  # res = np.zeros(im.shape, dtype=dtype)
  res = np.zeros_like(im)
  for attr in PetClass.attrs:
    log.debug("PET:%s:%d" % (attr, getattr(PetClass, attr)))
    res[Act.indices(im, attr)] = getattr(PetClass, attr)

  # uMap
  uMap = np.zeros(im.shape, dtype=dtype)
  for attr in Mu.attrs:
    log.debug("uMap:%s:%d" % (attr, getattr(Mu, attr)))
    uMap[Act.indices(im, attr)] = getattr(Mu, attr)

  # MR
  # t1 = np.zeros(im.shape, dtype=dtype)
  t1 = np.zeros_like(im)
  for attr in T1.attrs:
    log.debug("T1:%s:%d" % (attr, getattr(T1, attr)))
    t1[Act.indices(im, attr)] = getattr(T1, attr)
  # t2 = np.zeros(im.shape, dtype=dtype)
  t2 = np.zeros_like(im)
  for attr in T2.attrs:
    log.debug("T2:%s:%d" % (attr, getattr(T2, attr)))
    t2[Act.indices(im, attr)] = getattr(T2, attr)

  # resize
  new_shape = np.rint(np.asarray(im.shape) * Res.brainweb / out_res)
  padLR, padR = divmod((np.array(out_shape) - new_shape), 2)

  def resizeToMmr(arr):
    # oldMax = arr.max()
    # arr = arr.astype(np.float32)
    # arr /= arr.max()
    arr = resize(arr, new_shape,
                 order=1, mode='constant', anti_aliasing=False,
                 preserve_range=True).astype(dtype)
    if pad:
      arr = np.pad(arr, [(p, p + r) for (p, r)
                         in zip(padLR.astype(int), padR.astype(int))],
                   mode="constant")
    return arr

  return [resizeToMmr(i) for i in [res, uMap, t1, t2]]


def ellipsoid(out_shape, radii, position, dtype=np.float32):
    """
    out_shape  : 3-tuple
    radii  : 3-tuple, radii of ellipsoid
    position  : 3-tuple, centre within `out_shape` to place 3d ellipsoid
    """
    # grid for support points at centre `position`
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, out_shape)]
    position = np.ogrid[grid]
    # distance of all points from `position` scaled by radius
    out = np.zeros(out_shape, dtype=np.float32)
    for x_i, semisize in zip(position, radii):
        out += np.abs(x_i / semisize) ** 2
    return (out <= 1).astype(dtype)


def add_lesions(im3d, dim=Res.mMR * Shape.mMR, diam=None, intensity=None,
                blur=None, thresh=None, PetClass=FDG):
    """
    im3d  : 3darray
    dim  : `im3d` dimensions [default: Res.mMR * Shape.mMR]mm
    diam  : [default: [15, 7, 8, 5]]
    intensity  : [default: [
        PetClass.hot, -PetClass.cold, PetClass.hot, PetClass.hot]]
    blur  : [default: [0, 2, 2.5, 0]]
    thresh  : Minimum `im3d` value on which a tumour can be overlaid
      [default: 0.9 * PetClass.whiteMatter]
    """
    diam = diam or [15, 7, 8, 5]
    if thresh is None:
        thresh = 0.9 * PetClass.whiteMatter
    if intensity is None:
        intensity = [PetClass.hot, -PetClass.cold, PetClass.hot, PetClass.hot]
    blur = blur or [0, 2, 2.5, 0]

    rad = max(diam) / (2 * dim) * im3d.shape
    # locations not too close to edges for centre of tumours
    msk = gaussian_filter((im3d > thresh).astype(np.float32), rad) > 0.7
    # only central slice
    msk[:im3d.shape[0] // 2] = 0
    msk[im3d.shape[0] // 2 + 1:] = 0
    # make different quadrants
    msk = msk.astype(np.uint8)
    #msk[:, :im3d.shape[1] // 2, :im3d.shape[2] // 2] *= 1
    msk[:, :im3d.shape[1] // 2, im3d.shape[2] // 2:] *= 2
    msk[:, im3d.shape[1] // 2:, :im3d.shape[2] // 2] *= 3
    msk[:, im3d.shape[1] // 2:, im3d.shape[2] // 2:] *= 4
    im3d = im3d.copy()  # be safe

    params = list(zip(diam, intensity, blur))
    np.random.shuffle(params)
    for i, (d, peak, fwhm) in enumerate(params):
        quadrant = i % 4
        ZYX = np.asarray(np.where(msk == quadrant + 1))
        position = ZYX[:, np.random.choice(ZYX.shape[1])]
        radii = d / (2 * dim) * im3d.shape
        tumour = peak * ellipsoid(
            im3d.shape, radii, position, dtype=im3d.dtype)
        if fwhm:
            sigma = fwhm / (np.sqrt(8 * np.log(2)) * dim) * im3d.shape
            tumour = gaussian_filter(tumour, sigma, mode='constant')
            tumour = tumour.astype(tumour.dtype)
        if peak > 0:
            im3d = np.max([im3d, tumour], axis=0)
        else:
            #tROI = tumour > 0
            #im3d[tROI] = tumour[tROI]
            im3d += tumour
            #im3d[tROI] = np.min([im3d[tROI], tumour[tROI]], axis=0)

    return im3d


def matify(mat, dtype=np.float32, transpose=None):
  """@param transpose  : tuple<int>, (default: range(mat.ndim)[::-1])"""
  if transpose is None:
    transpose = tuple(range(mat.ndim)[::-1])
  return mat.transpose(transpose).astype(dtype)


def register(src, target=None, ROI=None, target_shape=Shape.mMR,
             src_resolution=Res.MR, target_resolution=Res.mMR,
             method="CoM",
             src_offset=None, dtype=np.float32):
    """
    Transforms `src` into `target` space.

    @param src  : ndarray. Volume to transform.
    @param target  : ndarray, optional.
      If (default: None), perform a basic transform using the other params.
      If ndarray, use as reference static image for registration.
    @param ROI  : tuple, optional.
      Ignored if `target` is unspecified.
      Region within `target` to use for registration.
      [default: ((0, None),)] for whole volume. Use e.g.:
      ((0, None), (100, -120), (110, -110)) to mean [0:, 100:-120, 110:-110].
    @param target_shape  : tuple, optional.
      Ignored if `target` is specified.
    @param src_offset  : tuple, optional.
      Static initial translation [default: (0, 0, 0)].
      Useful when no `target` is specified.
    @param method  : str, optional.
      [default: "CoM"]  : centre of mass.
    """
    from dipy.align.imaffine import AffineMap, transform_centers_of_mass
    log = logging.getLogger(__name__)

    assert src.ndim == 3
    if target is not None:
        assert target.ndim == 3
    assert len(target_shape) == 3
    assert len(src_resolution) == 3
    assert len(target_resolution) == 3

    if ROI is None:
        ROI = ((0, None),)
    ROI = tuple(slice(i, j) for i, j in ROI)
    if src_offset is None:
        src_offset = (0, 0, 0)
    method = method.lower()

    moving = src
    # scale
    affine_init = np.diag((src_resolution / target_resolution).tolist() + [1])
    # centre offset
    affine_init[:3, -1] = target.shape if target is not None else target_shape
    affine_init[:3, -1] -= moving.shape * src_resolution / target_resolution
    affine_init[:3, -1] /= 2
    affine_init[:3, -1] += src_offset
    affine_map = AffineMap(
        np.eye(4),
        target_shape, np.eye(4),  # unmoved target
        moving.shape, affine_init)
    src = affine_map.transform(moving)

    if target is not None:
        static = target
        if np.isnan(static).sum():
            log.warn("NaNs in target reference - skipping")
        else:
            # remove noise outside ROI
            msk = np.zeros_like(static)
            msk[ROI] = 1
            msk = affine_map.transform_inverse(msk)
            moving = np.array(moving)
            moving[msk == 0] = 0

            if method == "com":
                method = transform_centers_of_mass(
                    static, np.eye(4), moving, affine_init
                )
            else:
                raise KeyError("Unknown method:" + method)
            src = method.transform(moving)

    if dtype is not None:
        src = src.astype(dtype)
    return src
