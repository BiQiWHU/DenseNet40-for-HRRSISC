An implementation of DenseNet40 for HRRSISC
High Resolution Remotely Sensed Imagery Scene Classification, short for HRRSISC, is a challenge task because of the high intra-class dissimilarity and external class similarity.
Since 2012, deep learning based method has been widely used in HRRSISC and outperforms almost all low and middle level feature based methods.
This project implements DenseNet40 for HRRSISC
Dataset:UC Merced Dataset   
Training samples：test samples=8：2
Different from the original implementation in computer vision, Adam optimizer is used in this project.

tfdata.py: Transfer the original dataset images into standard tensorflow format,   
and generate pipelines for training.
training.py: Train DenseNet40 
testacc.py:Using trained model in training.py to test on the test samples, output the overall accuracy and each class's classification accuracy.

result:
unfortunately, the structure of Dense40 is not suitable for large scale high resolution remote sensing image. The input size of 32*32 means we have to down-scale 8 times on UC dataset(256*256). Too much spatial information is lost in this process. 
After training,the OA is about 0.8.
For better results, researchers have to change the network structure to make it fit the image size of 256*256, instead of downsampling simply.


