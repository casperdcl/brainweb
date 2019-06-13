try:  # py2
    from StringIO import StringIO
except ImportError:  # py3
    from io import StringIO

__author__ = "Casper O. da Costa-Luis <casper.dcl@physics.org>"
__date__ = "2017-19"
__license__ = "[MPLv2.0](https://www.mozilla.org/MPL/2.0)"
__all__ = ["volshow", "StringIO"]


def volshow(vol, cmap="Greys_r"):
    """Interactively slice through a 3D array in Jupyter"""
    import matplotlib.pyplot as plt
    import ipywidgets as ipyw

    f = plt.figure()
    def plot_slice(z=len(vol) // 2):
        """z  : int, slice index"""
        plt.figure(f.number)
        plt.clf()
        plt.imshow(vol[z], cmap=cmap)
        plt.show()

    return ipyw.interact(plot_slice,
        z=ipyw.IntSlider(min=0, max=len(vol) - 1, step=1, value=len(vol) // 2))