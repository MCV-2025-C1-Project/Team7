# Setting up local development environment
This repository uses [uv](https://docs.astral.sh/uv/) for project management.
You will need to install uv to develop this project.

First clone the repository:
```shell
git clone https://github.com/MCV-2025-C1-Project/Team7.git
```

Then we can ask uv to set up the local environment.
```shell
cd Team7
uv sync
```

If you wanna use jupyter notebooks + uv, follow this [guide](https://docs.astral.sh/uv/guides/integration/jupyter/).

# BBDD Structure
The BBDD folder contains the following structure:
```
BBDD
├── bbdd_00000.jpg
├── bbdd_00000.png
├── bbdd_00000.txt
├── bbdd_00001.jpg
├── bbdd_00001.png
├── bbdd_00001.txt
├── ...
├── bbdd_00286.txt
├── relationships.pkl
```
The images are in jpg format, the masks in png format (I think they are binary masks used to segment the paintings?) and the txt files contain the metadata of each painting (painting title).
The `relationships.pkl` file contains a dictionary with the following structure:
```python
{
    'IMG_20180926_114224': 0,
    'IMG_20190803_134913': 1,
    'IMG_20180926_115603_4': 2,
    'IMG_20190803_145308': 3,
    'IMG_20180926_120525': 4,
    ...
    'IMG_20190803_132712': 286
}
```
So it's basically a mapping between the original image filename and the index of the image in the BBDD folder. So for now, we can ignore it.

# QSD1_W1 Structure
The QSD1_W1 folder contains the following structure:
```
qsd1_w1
├── 00000.jpg
├── 00001.jpg
├── 00002.jpg
├── ...
├── 00029.jpg
├── gt_corresps.pkl
```
The images are in jpg format and the `gt_corresps.pkl` file contains the following:
```python
[[120], [170], [277], [227], [251], [274], [285], [258], [117], [203], [192], [22], [113], [101], [174], [155], [270], [47], [286], [215], [262], [245], [257], [182], [262], [38], [238], [67], [86], [133]]
```
So it's a list of lists where each inner list contains the index of the corresponding painting in the BBDD folder. So for example, the first query image (00000.jpg) corresponds to the painting with index 120 in the BBDD folder (bbdd_00120.jpg).