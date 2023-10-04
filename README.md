# ATeX - [ATLANTIS](https://github.com/smhassanerfani/atlantis) TeXture Dataset
This is the repository for the ATeX Dataset. All labels are comprehensively described in [ATeX Wiki](https://github.com/smhassanerfani/atex/wiki).
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_samples.svg">
  Figure 1. ATeX: ATLANTIS TeXture dataset.
</p>

## Overview
ATeX is a new benchmark for classification and texture analysis of water in different waterbodies. This dataset has covered a wide range of waterbodies such as sea, lake, river, swamp, glacier. ATeX includes patches with 32 x 32 pixels of 15 waterbodies. ATeX consists of 12,503 patches split into 8,753 for training, 1,252 for validation, and 2,498 for testing.

## Dataset Description
Water does not preserve the same texture and visual features in all forms and situations. Some physical and chemical properties of water have an effect on water's appearance in different waterbodies. Turbidity, color, temperature, suspended living matter, mineral particles, and dissolved chemical substances are those water characteristics playing a role in water appearance, while water depth and flowrate are those dictated by the flow regime having a direct effect on water turbulence. Water is also a reflective surface. In laminar flow or still water, depending on ambient light, the reflection effect can be dominant while in turbulent flow because of existing coherent flow structures such as eddies, turbulent bursting, and unsteady vortices of many sizes, the reflection becomes distorted. Moreover, turbulent regime plays a critical role in terms of accretion and transport of sediment as well as contaminant mixing and dispersion in rivers having a direct effect on water turbidity and visual appearance.

Considering the combination of the aforementioned water properties, water can appear in completely different forms in various waterbodies. The ATeX dataset is designed and developed with the goal of representing the texture appearance that water usually bears in different waterbodies. ATeX images are derived from ATLANTIS (ArTificiaL And Natural waTer-bodIes dataSet). ATLANTIS is a semantic segmentation dataset including 5,195 pixel-wise annotated images that covers a wide range of natural and artificial waterbodies such as sea, lake, river, reservoir, canal, pier, and pipeline. Figure 2 shows the pipeline through which the ATeX images are cropped from ATLANTIS images. As shown in Figure 2 (STEP 3) there is no partial overlap between any two patches.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_pipeline.png">
  Figure 2. ATeX patches are derived from ATLANTIS. The boundary waterbodies are determined from images using corresponding ground-truth masks (STEP 1), then the irrelevant pixels are cut based on waterbodies' coordination (STEP 2), and finally, the outputs are cropped 32 x 32 to create ATeX patches (STEP 3).
</p>

## Dataset Statistics
Figure 3 shows the frequency distribution of the number of images for waterbody labels.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/frequency_distribution.svg">
  Figure 3. Frequency distribution of the number of images assigned to each waterbody label.
</p>

## ATeX Visualization
We used t-Distributed Stochastic Neighbor Embedding ([t-SNE](https://lvdmaaten.github.io/tsne/)), Linear Auto Encoder (AE), Linear Discriminant Analysis ([LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)) and Principal Component Analysis ([PCA](https://www.youtube.com/watch?v=52d7ha-GdV8&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=11)) for dimensionality reduction. All of these methods are well-suited for the visualization of high-dimensional datasets. In order to achieve better results, images of the train set are first passed into the feature layers of [ShuffleNet v2](https://arxiv.org/abs/1807.11164) (pre-trained on ImageNet and fine-tuned on ATeX), then the output results were fed into the features reduction models. Figure 4 (a) shows animation for the t-SNE result on features extracted from ShuffleNet V2x1.0, and Figure 4 (b) shows animation for the AE result on those. Figure 4 (c) and (d) are the results for LDA and PCA, respectively.

<TABLE>
  <TR>
     <TD><img src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_tsne.gif" width="100%" /></TD>
     <TD><img src="https://github.com/smhassanerfani/atex/blob/main/wiki/atex_ae.gif" width="100%" /></TD>
  </TR>
  <TR>
     <TD align="center">(a)</TD>
     <TD align="center">(b)</TD>
  </TR>
  <TR>
     <TD><img src="https://github.com/smhassanerfani/atex/blob/main/wiki/LDAv3.svg" width="100%" /></TD>
     <TD><img src="https://github.com/smhassanerfani/atex/blob/main/wiki/PCAv3.svg" width="100%" /></TD>
  </TR>
  <TR>
     <TD align="center">(c)</TD>
     <TD align="center">(d)</TD>
  </TR>
</TABLE>

## Experimental Results
Three common performance metrics including Precision, Recall, and F1-score are reported to evaluate the performance of the models on ATeX. Table 1 shows the weighted average (averaging the support-weighted mean per label) of these three metrics on the test set. Accordingly, EffNet-B7, EffNet-B0, and ShuffleNet V2x1.0 provide the best results. Considering training time, ShuffleNet V2x1.0 can be presented as the most efficient network.

Table 1. The performance result on ATeX test set by well-known classification models.
| Networks           | Training Time [h:mm:ss] | Learning Rate | Epochs | Accuracy  (Val) | Precision | Recall | F1-score |
|--------------------|-------------------------|---------------|--------|-----------------|-----------|--------|----------|
| Wide ResNet-50-2   | 0:06:56                 | 2.50E-04      | 30     | 91              | 77        | 75     | 75       |
| VGG-16             | 0:04:38                 | 2.50E-04      | 30     | 90              | 75        | 72     | 72       |
| SqueezeNet 1.0     | 0:00:47                 | 7.50E-04      | 30     | 82              | 81        | 81     | 81       |
| ShuffleNet V2 x1.0 | 0:01:46                 | 1.00E-02      | 30     | 90              | 90        | 90     | 90       |
| ResNeXt-50-32x4d   | 0:03:15                 | 2.50E-04      | 30     | 90              | 77        | 75     | 75       |
| ResNet-18          | 0:01:28                 | 2.50E-04      | 30     | 87              | 74        | 72     | 72       |
| MobileNet V2       | 0:01:35                 | 2.50E-04      | 30     | 88              | 74        | 72     | 72       |
| GoogleNet          | 0:01:51                 | 5.00E-03      | 30     | 89              | 88        | 88     | 88       |
| EfficientNet-B7    | 0:12:42                 | 1.00E-02      | 30     | 90              | 91        | 91     | 91       |
| EfficientNet-B0    | 0:02:38                 | 7.50E-03      | 30     | 91              | 90        | 90     | 90       |
| Densenet-161       | 0:06:15                 | 2.50E-04      | 30     | 91              | 81        | 79     | 79       |

### Convolutional Autoencoder

Autoencoders are an unsupervised learning technique that we can use to learn efficient data encodings. Basically, autoencoders can learn to map input data to the output data. While doing so, they learn to encode the data. And the output is the compressed representation of the input data ([Autoencoders in Deep Learning](https://debuggercafe.com/autoencoders-in-deep-learning/)). We developed a Convolutional Autoencoder including three encoder and three decoder 2D convolution layers. 

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atex/blob/main/wiki/on_img_ae_results.svg">
  Figure 4. Resutls for 1000 epochs. Each row (from bottom to top) shows the sample results from epoch 100 to 1000. 
</p>


# Reference
If you use this data, please cite the following paper which can be downloaded through this [link](https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29WR.1943-5452.0001615):
```
@article{erfani2022atex,
  title={ATeX: A Benchmark for Image Classification of Water in Different Waterbodies Using Deep Learning Approaches},
  author={Erfani, Seyed Mohammad Hassan and Goharian, Erfan},
  journal={Journal of Water Resources Planning and Management},
  volume={148},
  number={11},
  pages={04022063},
  year={2022},
  publisher={American Society of Civil Engineers}
}
@article{erfani2023vision,
  title={Vision-based texture and color analysis of waterbody images using computer vision and deep learning techniques},
  author={Erfani, Seyed Mohammad Hassan and Goharian, Erfan},
  journal={Journal of Hydroinformatics},
  volume={25},
  number={3},
  pages={835--850},
  year={2023},
  publisher={IWA Publishing}
}
```
