# ATeX - [ATLANTIS](https://github.com/smhassanerfani/atlantis) TeXture Dataset
This is the respository for the ATeX Dataset. All labels are comprehensively described in [ATeX Wiki](https://github.com/smhassanerfani/atex/wiki).
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_samples.svg">
  Figure 1. ATeX: ATLANTIS TeXture dataset.
</p>

## Overview
ATeX is a new benchmark for classification and texture analysis of water in different waterbodies. This dataset has covered a wide range of waterbodies such as sea, lake, river, swamp, glacier. ATeX includes patches with 32 x 32 pixels of 15 waterbodies. ATeX consists of 12,503 patches split into 8,753 for training, 1,252 for validation and 2,498 for testing.

## Dataset Description
Water does not preserve same texture and visual features in all forms and situations. Some physical and chemical properties of water have effect on water appearance in different waterbodies. Turbidity, color, temperature, suspended living matter, mineral particles and dissolved chemical substances are those water characteristics playing role in water appearance, while water depth and flowrate are those dictated from flow regime having direct effect on water turbulence. Water is also a reflective surface. In laminar flow or still water, depending on ambient light, the reflection effect can be dominant while in turbulent flow because of existing coherent flow structures such as eddies, turbulent bursting, and unsteady vortices of many sizes the reflection becomes distorted. Moreover, turbulent regime plays a critical role in terms of accretion and transport of sediment as well as contaminant mixing and dispersion in rivers having direct effect on water turbidity and visual appearance.

Considering the combination of the aforementioned water properties, water can appear in completely different forms in various waterbodies. The ATeX dataset is designed and developed with the goal of representing texture appearance which water usually bears in different waterbodies. ATeX images are derived from ATLANTIS (ArTificiaL And Natural waTer-bodIes dataSet). ATLANTIS is a semantic segmentation dataset including 5,195 pixel-wise annotated images which covers a wide range of natural and artificial waterbodies such as sea, lake, river, reservoir, canal, pier, peline. Figure 2 shows the pipeline through which the ATeX images are cropped from ATLANTIS images. As it is showed in Figure 2 (STEP 3) there is no partial overlap between any two patches.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_pipeline.png">
  Figure 2. ATeX patches are derived from [ATLANTIS](https://github.com/smhassanerfani/atlantis). The boundary waterbodies are determined from images using correspoing ground-truth masks (STEP 1), then the irrelevant pixels are cut based on waterbodies' coordination (STEP 2), finally the outputs are cropped 32 x 32 to create ATeX patches (STEP 3).
</p>

## Dataset Statistics
Figure 3 shows the frequency distribution of the number of images for waterbody labels.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/frequency_distribution.svg">
  Figure 3. Frequency distribution of the number of images assigned to each waterbody label.
</p>




## Unsupervised Analysis
<p float="left">
    <img src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_tsne.gif" width="49%" />
    <img src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_ae.gif" width="49%" /> 
</p>

<p float="left">
    <img src="https://github.com/smhassanerfani/atex/blob/main/wiki/LDAv3.svg" width="49%" />
    <img src="https://github.com/smhassanerfani/atex/blob/main/wiki/PCAv3.svg" width="49.2%" /> 
</p>

## ATeX Related Projects
* [ATLANTIS](https://github.com/smhassanerfani/atlantis) is a code used for downloading images from [Flickr](https://www.flickr.com) 

### Citations
Mohammad

