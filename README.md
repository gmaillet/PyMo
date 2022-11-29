# PyMo
Version Python du mosaiquage par maxflow

## environnement Python
Ce programme nécessite l'installation des modules suivants:
- pymaxflow (GPL-3.0): `` conda install -c conda-forge pymaxflow``
- scikit-image (licence BSD-3): ``conda install -c anaconda scikit-image``
- rasterio (licence BSD-3): ``conda install -c conda-forge rasterio``

## utilisation/exemple

``python PyMoSP.py -i data\L1_opi_ss10.tif -s data\L1_quality_ss10.tif -o data\temp.tif -c 10 -a 256 -w True``


