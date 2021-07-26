# Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images-and-a-NovelDeep-Convolutional-Neural-Network
The recent emergence of a highly infectious and contagious respiratory viral disease known as COVID-19 has vastly impacted human lives and greatly burdened the health care system. Therefore, it is indispensable to develop a fast and accurate diagnostic system for timely identification of COVID-19 infected patients and to control its spread. This work proposes a new X-ray based COVID-19 classification framework consisting of (i) end-to-end classification module and (ii) deep feature-space based Machine Learning classification module. In this regard, two new custom CNN architectures, namely COVID-RENet-1 and COVID-RENet-2 are developed for COVID-19 specific pneumonia analysis by systematically employing Region and Edge base operations along with convolution operations. The synergistic use of Region and Edge based operations explores the region homogeneity, textural variations and region boundary; it thus helps in capturing the pneumonia specific pattern. In the first module, the proposed COVID-RENets are used for end-to-end classification. In the second module, the discrimination power is enhanced by jointly providing the deep feature hierarchies of the COVID-RENet-1 and COVID-RENet-2 to SVM for classification. The discrimination capacity of the proposed classification framework is assessed by comparing it against the standard state-of-the-art CNNs using radiologist’s authenticated chest X-ray dataset. The proposed classification framework shows good generalization (accuracy: 98.53%, F-score: 0.98, MCC: 0.97) with considerable high sensitivity (0.99) and precision (0.98). The exemplary performance of the classification framework suggests its potential use in other X-ray imagery based infectious disease analysis.

In this repository, we provide the MATLAB GUI and Testing Code for the Coronavirus Disease Analysis using Chest X-ray Images for the research community to use our research work.

# Overview of the workflow for the proposed COVID-19 Classification Framework
In this work, a new classification framework is developed based on deep learning and classical ML techniques for automatic discrimination of COVID-19 infected patients from healthy individuals based on chest X-ray images. The proposed classification framework is constituted of two modules: (i) Proposed COVID-RENet based end-to-end Classification, and (ii) Deep Concatenated Feature-space based ML classification. In the experimental setup, initially, training samples were augmented to improve the generalization. These augmented samples were used to train the two proposed modules. Fig. 1. (A) shows the modules of the proposed COVID-19 classification framework, whereas (B) gives the detailed overview of the workflow.

![workflow](https://user-images.githubusercontent.com/45933925/126980404-568c6cdf-0d4b-44bd-abbf-7429e245a910.png)

# Models Architectures

### Architectural details of the proposed COVID-RENet-1

![architecture-covid-renet-1](https://user-images.githubusercontent.com/45933925/126981075-07ae9db5-bcdf-4a9d-9a65-35abae7322cf.png)

### Architectural details of the proposed COVID-RENet-2

![architecture-covid-renet-2](https://user-images.githubusercontent.com/45933925/126981849-ec810d2c-cfca-4c54-9a8a-70bda34eebcd.png)

### Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification module

[Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification module](https://github.com/PRLAB21/Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images/blob/main/repo-images/architecture-covid-19-classification-module.png)

#### Trained Model is available at [COVID-RENet-1](https://drive.google.com/file/d/1IY8Di0Jqlmb7pjw6OmKdmc2QasnLQ3sA/view) and [COVID-RENet-2](https://drive.google.com/file/d/1ctjUFQLtNgMcKbQCYdPaPEsWXiBqhujM/view) links.

# Dataset

We built a new dataset consisting of X-ray images of COVID-19 pneumonia and healthy individuals in this work. X-ray images were collected from Open Source GitHub repository and Kaggle repository called “pneumonia”

#### Dataset will be available on request, contact at [hengrshkhan822@gmail.com]

### Dataset Samples

Panel (A) and (B) show COVID-19 infected and healthy images, respectively.

<img width="338" alt="dataset" src="https://user-images.githubusercontent.com/45933925/126985731-49ca6028-30a8-41af-aaa0-5f5741fc6bdf.png">

# Training plot of the proposed COVID-RENet-1 and COVID-RENet-2

<img width="623" alt="training-plot-RENet-2" src="https://user-images.githubusercontent.com/45933925/126986130-a5ac49ac-63b3-44b0-9831-239e275c9e39.png">

# Results

Performance comparison of the proposed COVID-RENet-1, COVID-RENet-2 and DCFS-MLC with standard existing CNNs.

![performance-comparison-01](https://user-images.githubusercontent.com/45933925/126986414-4d5a4fac-6573-4395-853b-72570138e0af.png)

Detection and misclassification rate analysis of the proposed COVID-RENet-1, COVID-RENet-2, DCFS-MLC and ResNet.

![performance-comparison-02](https://user-images.githubusercontent.com/45933925/126986466-78a35dcb-7bb2-4f21-a056-3f715d3f077b.png)

Performance metrics for the state-of-the-art CNN models that are trained from scratch and TL-based fine-tuned pre-trained on the augmented dataset.

![results-01](https://user-images.githubusercontent.com/45933925/126987021-4775f68c-12d5-4e6c-9596-efc927eca556.png)

Performance metrics for the deep feature extraction from custom layers of state-of-the-art training from scratch and TL-based fine-tuned pre-trained CNN on the augmented dataset.

![results-02](https://user-images.githubusercontent.com/45933925/126987106-54601a5f-2446-425b-912b-02a23b41136f.png)

COVID-19 (panel a & b) and Healthy (panel c & d) images, which are misclassified.

![output](https://user-images.githubusercontent.com/45933925/126990084-79cfdae9-106e-4de7-8cb3-b97c9a38c52e.png)

# PCA Visualization

Feature visualization of the proposed COVID-RENet-1, COVID-RENet-2, DCFS-MLC and the best performing standard existing CNN (ResNet) on test dataset.

<img width="355" alt="PCA-COVID-RENet-1" src="https://user-images.githubusercontent.com/45933925/126988128-b0487d4d-adf4-46d0-abdb-64945da11a0f.png">

# Heatmaps

Panel (a) shows the original chest X-ray image. Panel (b) shows the radiologist defined COVID-19 infected regions highlighted by yellow circle or black arrow. The resulted heat map of the proposed COVID-RENet-2 and COVID-RENet-1 is shown in panels (c & d), respectively. Panel (e) shows the heat map of the standard existing ResNet model.

![heatmap](https://user-images.githubusercontent.com/45933925/126989628-62b26d01-e233-4bf5-b7e9-27ddb433bf9a.png)

# ROC Curves

ROC curve for the proposed approach (DCFS-MLCS), proposed models (COVID-RENet-1 and COVID-RENet-2), and standard existing CNN models. The values in square bracket show a standard error at the 95% confidence interval.

<img width="367" alt="ROC-TL-based-fine-tuned-CNNs" src="https://user-images.githubusercontent.com/45933925/126989035-f08d9bf6-412f-4e39-869b-b14082692ca6.png">

# Requirements

1. Matlab 2019b.
2. Deep Learning library.
3. NVIDIA GeForce GTX Titan X Computer.

# Setup

```git bash
git clone https://github.com/PRLAB21/Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images.git
```

2. Download model and place it in following structure.

```text
Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images
|__ models
   |__ net_RENet_VGG_Modified1.mat
   |__ net_RENet_VGG_Modified2.mat
```
3. Testing images are downloaded along with this repo and are present inside "test-dataset" directory.

4. Run testing code using below mentioned methods.

# Inference Code

1. Open MATLAB.
2. Change MATLAB Working Directory to this repository's folder from top panel.
3. Now add each folder to MATLAB path from Current Folder panel by right clicking on each folder and selecting.
   Add to Path > Selected Folder and Subfolders.
4. Run any of the two test model using following files.

-  **test_code_RENet_VGG_Modifier1.m**: Use this file for testing the model "net_RENet_VGG_Modified1".
-  **test_code_RENet_VGG_Modifier2.m**: Use this file for testing the model "net_RENet_VGG_Modified2".

# Co-Author

Prof. Asifullah Khan,

Department of Computer and Information Sciences (DCIS),

Pakistan Institute of Engineering and Applied Sciences (PIEAS).

Email: [asif@pieas.edu.pk]

faculty.pieas.edu.pk/asifullah/

# How to cite / More information

Khan, Saddam Hussain, Anabia Sohail, Muhammad Mohsin Zafar, and Asifullah Khan. "Coronavirus Disease Analysis using Chest X-ray Images and a Novel Deep Convolutional Neural Network." (2020), 10.13140/Rg. 2.2. 35868.64646 April (2020): 1-31.


