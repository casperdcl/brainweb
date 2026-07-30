import pytest

import brainweb as bw


@pytest.mark.quick
def test_import():
    assert bw.__version__


@pytest.fixture(scope="session")
def subject_file(tmp_path_factory):
    f, url = next(iter(bw.LINKS.items()))
    return bw.get_file(f, url, cache_dir=tmp_path_factory.mktemp("brainweb"))


def test_download(subject_file):
    from pathlib import Path
    assert Path(subject_file).exists()


def test_load(subject_file):
    data = bw.load_file(subject_file)
    assert data.shape == (362, 434, 362)


def test_mMR(subject_file):
    vol = bw.get_mmr_fromfile(subject_file)
    assert not {'T1', 'T2', 'uMap', 'PET'} - set(vol.keys())
    assert vol['PET'].shape == (127, 344, 344)


def test_lesions(subject_file):
    vol = bw.get_mmr_fromfile(subject_file)
    im3d = bw.add_lesions(vol['PET'])
    assert im3d.shape == vol['PET'].shape
    assert (im3d != vol['PET']).any()


def test_registration(subject_file):
    pytest.importorskip("dipy")
    data = bw.load_file(subject_file)
    vol = bw.get_mmr_fromfile(subject_file)
    reg = bw.register(data[:, ::-1], target=vol['PET'], src_resolution=bw.Res.brainweb, target_resolution=bw.Res.mMR)
    assert reg.shape == vol['PET'].shape
