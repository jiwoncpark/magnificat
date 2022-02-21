==================================================
Magnificat - Simulating LSST-like AGN light curves
==================================================

.. image:: https://readthedocs.org/projects/magnificat/badge/?version=latest
    :target: https://magnificat.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/jiwoncpark/magnificat/actions/workflows/package_install.yml/badge.svg
    :target: https://github.com/jiwoncpark/magnificat/actions/workflows/package_install.yml/badge.svg
    :alt: CI

Installation
============

0. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n magnificat python=3.8 -y
$conda activate magnificat

1. Clone the repo and install.

::

$git clone https://github.com/jiwoncpark/magnificat.git
$cd magnificat
$pip install -e . -r requirements.txt

2. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name magnificat --display-name "Python (magnificat)"

