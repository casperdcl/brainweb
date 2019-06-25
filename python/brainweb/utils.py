from __future__ import division
import numpy as np
from numpy.random import seed
from skimage.transform import resize
from skimage.filters import gaussian
from tqdm.auto import tqdm
import functools
import requests
import re
import os
import gzip
from os import path
from glob import glob
import logging

__author__ = "Casper O. da Costa-Luis <casper.dcl@physics.org>"
__date__ = "2017-19"
__licence__ = __license__ = "[MPLv2.0](https://www.mozilla.org/MPL/2.0)"
__all__ = [
    "volshow", "get_files", "load_file", "get_mmr", # necessary
    "get_file", "gunzip_array",  # useful utils
    "noise", "seed", "toPetMmr", "LINKS", #  probably not needed
]

LINKS = "04 05 06 18 20 38 41 42 43 44 45 46 47 48 49 50 51 52 53 54"
LINKS = [
    'http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1' +
    '?do_download_alias=subject' + i + '_crisp&format_value=raw_short' +
    '&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D'
    for i in LINKS.split()]
RE_SUBJ = re.compile('.*(subject)([0-9]+).*')
LINKS = dict((RE_SUBJ.sub(r'\1_\2.bin.gz', i), i) for i in LINKS)


def get_file(fname, origin, cache_dir=None):
    """
    Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.brainweb`, and given the filename `fname`.
    The final location of a file
    `subject_04.bin.gz` would therefore be `~/.brainweb/subject_04.bin.gz`.
    Vaguely based on:
    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py

    @param fname  : Name of the file. If an absolute path `/path/to/file.txt` is
        specified the file will be saved at that location.
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
        except:
            pass
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


load_file = functools.partial(gunzip_array,
    shape=(362, 434, 362), dtype=np.uint16)


def get_files(cache_dir=None):
    """
    Returns list of files which can be `numpy.load`ed
    """
    files = []
    for f, url in tqdm(LINKS.items(), unit="file", desc="BrainWeb Subjects"):
        files.append(get_file(f, url, cache_dir=cache_dir))
    return sorted(files)


def get_mmr(cache_file, raw_data,
            petNoise=1.0, t1Noise=0.75, t2Noise=0.75,
            petSigma=1.0, t1Sigma=1.0, t2Sigma=1.0):
    """
    Return contents of specified `*.npz` file,
    creating it from BrainWeb `raw_data` 3darray if it doesn't exist.
    """
    if not path.exists(cache_file):
        pet, uMap, t1, t2 = toPetMmr(raw_data)
        pet = noise(pet, petNoise, sigma=petSigma)[:, ::-1]
        t1 = noise(t1, t1Noise, sigma=t1Sigma)[:, ::-1]
        t2 = noise(t2, t2Noise, sigma=t2Sigma)[:, ::-1]
        uMap = uMap[:, ::-1]
        np.savez_compressed(cache_file, PET=pet, uMap=uMap, T1=t1, T2=t2,
            petNoise=petNoise, t1Noise=t1Noise, t2Noise=t2Noise,
            petSigma=petSigma, t1Sigma=t1Sigma, t2Sigma=t2Sigma)

    return np.load(cache_file, allow_pickle=True)


def volshow(vol,
            cmaps=None, colorbars=None,
            xlabels=None, ylabels=None, titles=None,
            sharex=True, sharey=True):
    """
    Interactively slice through 3D array(s) in Jupyter

    @param vol  : 3darray or [3darray, ...] or {'title': 3darray, ...}
    @param cmaps  : list of cmap [default: ["Greys_r", ...]]
    @param xlabels, ylabels, titles  : list of strings (default blank)
    @param sharex, sharey  : passed to `matplotlib.pyplot.subplots`
    """
    import matplotlib.pyplot as plt
    import ipywidgets as ipyw

    if hasattr(vol, "keys") and hasattr(vol, "values"):
        if titles is not None:
            log.warn("ignoring `vol.keys()` in favour of specified `titles`")
        else:
            titles = vol.keys()
            vol = vol.values()

    if vol[0].ndim == 2:
        vol = [vol]
    else:
        assert vol[0].ndim == 3, "Input should be (one or a list of) 3D array(s)"

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

    cols = max(1, int(len(vol) ** 0.5))
    rows = int(np.ceil(len(vol) / cols))
    zSize = min(len(i) for i in vol)
    fig = plt.figure()

    @ipyw.interact(z=ipyw.IntSlider(zSize // 2, 0, zSize - 1, 1))
    def plot_slice(z):
        """z  : int, slice index"""
        plt.figure(fig.number)
        plt.clf()
        axs = fig.subplots(rows, cols, sharex=sharex, sharey=sharey)
        axs = getattr(axs, 'flat', [axs])
        for ax, v, cmap, cbar, xlab, ylab, tit in zip(axs, vol, cmaps, colorbars, xlabels, ylabels, titles):
            plt.sca(ax)
            plt.imshow(v[z], cmap=cmap)
            if cbar:
                plt.colorbar()
            if xlab:
                plt.xlabel(xlab)
            if ylab:
                plt.ylabel(ylab)
            if tit:
                plt.title(tit)
            plt.show()
        plt.tight_layout(0, 0, 0)
        #return fig, axs

    return plot_slice


class Act(object):
  """careful: occasionally other bits may be set"""
  background, csf, greyMatter, whiteMatter, fat, muscle, skin, skull, vessels,\
      aroundFat, dura, marrow\
      = [i << 4 for i in range(12)]
  bone = skull | marrow | dura

  @classmethod
  def indices(cls, im, attr):
    if attr == "bone":
      return (cls.indices(im, "skull") +
              cls.indices(im, "marrow") +
              cls.indices(im, "dura") > 0)
    return abs(im - getattr(cls, attr)) < 1


class Pet(Act):
  whiteMatter = 32
  greyMatter = whiteMatter * 4
  skin = whiteMatter // 2
  hot = greyMatter * 1.5
  cold = whiteMatter * 0.5
  attrs = ["whiteMatter", "greyMatter", "skin", "hot", "cold"]


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
           "csf", "hot", "cold"]


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


mu_bone_1_cm = 0.13
mu_tissue_1_cm = 0.0975


class Res(object):
  mMR = np.array([2.0312, 2.0863, 2.0863])
  MR = np.array([1.0, 1.0, 1.0])
  brainweb = np.array([0.5, 0.5, 0.5])


class Shape(object):
  mMR = np.array([127, 344, 344])
  MR = mMR * Res.mMR / Res.MR
  brainweb = mMR * Res.mMR / Res.brainweb


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
  r = gaussian(np.random.rand(*im.shape), sigma=sigma, multichannel=False)
  return im * (1 + n * (2 * r - 1))


def toPetMmr(im, pad=True, dtype=np.float32, outres="mMR"):
  """
  @return out  : [[PET, uMap, T1, T2], 127, 344, 344]
  """
  log = logging.getLogger(__name__)

  out_res = getattr(Res, outres)
  out_shape = getattr(Shape, outres)

  # PET
  # res = np.zeros(im.shape, dtype=dtype)
  res = np.zeros_like(im)
  for attr in Pet.attrs:
    log.debug("PET:%s:%d" % (attr, getattr(Pet, attr)))
    res[Act.indices(im, attr)] = getattr(Pet, attr)

  # muMap
  muMap = np.zeros(im.shape, dtype=dtype)
  muMap[im != 0] = mu_tissue_1_cm
  muMap[Act.indices(im, "bone")] = mu_bone_1_cm

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
                 order=1, mode="constant", anti_aliasing=False)
    if pad:
      arr = np.pad(arr, [(p, p + r) for (p, r)
                         in zip(padLR.astype(int), padR.astype(int))],
                   mode="constant")
    if arr.dtype == np.uint16:
      return np.asarray(arr, dtype=np.float32) * np.float32(2 ** 16)
    return arr.astype(dtype)

  return [resizeToMmr(i) for i in [res, muMap, t1, t2]]


def lesionMmr(im3d, wmm=Shape.mMR[-1] * Res.mMR[-1], diam=5,
      intensity=Pet.hot, num=1, blur=0):
    """
    im3d  : 3darray
    wmm  : `im3d` width in mm
    diam  : minimum tumour diameter [default: 5mm]
    intensity  : tumour intensity [default: Pet.hot]
    num  : number of tumours [default: 1]
    blur  : minimum Gaussian FWHM for tumours [default: 0mm]
    """
    # tumours specified by fractional positions d, h, w, radius/[w], sigma/[w]
    rad = diam / (2 * wmm)
    sigma = blur / (np.sqrt(8 * np.log(2)) * wmm)

    # coords: depth, height, width
    d, h, w = im3d.shape

    pad = 6  #1.5  # place tumours in circle of diameter = 2 / pad
    if num > 0:
      z = 0.5
      for r, t in np.random.random((2, num))
        x, y = pol2cart(r * 2 * np.pi, t)
        # centre about 0.5
        y = (y + 1) / 2
        x = (x + 1) / 2
        im3d = np.max([im3d,
            tumour((d, h, w), (z, y, x), rad, sigma=sigma, intensity=intensity, scale=1/pad)],
            axis=0)
    else:
      raise NotImplementedError
    # im3d = im3d / im3d.max()
    return im3d


def tumour(out_shape, loc, rad, sigma=0, intensity=Pet.hot, scale=1):
    """
    out_shape  : image depth, height, width
    loc  : z, y, x of centre of tumour
    rad  : radius of tumour
    scale (default 1) towards centre for tumour z/y/x
    """
    d, h, w = out_shape

    z, y, x = np.array(loc) - 0.5
    phi, theta, r = cart2sph(x, y, z)
    x, y, z = sph2cart(phi, theta, r * scale) + 0.5

    T = [z, y, x, rSmall, maxScale, sigma] .* [d, h, w, w, intensity, w]

    % [D, H, W, R, V, S] = T;
    S = T(6);

    [X, Y, Z] = meshgrid(1:w, 1:h, 1:d);
    im3d = ((X-T(3)).^2 + (Y-T(2)).^2 + (Z-T(1)).^2 <= T(4)^2) .* T(5);
    if S > 0
      im3d = imgaussfilt3(im3d, S);
    end

    im3d = permute(im3d, [3 2 1]);

    return im3d


def matify(mat, dtype=np.float32, transpose=None):
  """@param transpose  : tuple<int>, (default: range(mat.ndim)[::-1])"""
  if transpose is None:
    transpose = tuple(range(mat.ndim)[::-1])
  return mat.transpose(transpose).astype(dtype)