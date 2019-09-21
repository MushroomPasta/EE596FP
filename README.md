# EE596 Final Project
# Youtube-8M video semantic multi-label classification 
  Semantic video classification is providing semantic tags for videos, where each video can be
associated with multiple categories. For example, a video can be categorized as sport, and further to
be basketball. In our project, we aim at predicting the probability of each video belongs to each
category and choosing the top 5 as our labels for the result. Our data comes from YouTube8M
dataset, including 3862 total categories with 6.1million videos. And we treat the problem as multiple
binary classification problem.
# Data
[Youtube-8M link](https://research.google.com/youtube8m/)<br/>
Youtube8M dataset contains over 6 million videos with 3862 labels. The average number of label is 3
per video. Two types of features are available, frame-level and video-level. All features are obtained
from a GoogLeNet pre-trained on ImageNet. For our semantic video classification need, we uses
the frame-level feature. To obtain frame-level feature, the original video is down sampled into 300
frames. For each frame, a 1024-dimension feature is generated. 
# Approaches
## Logistic Regression
Logistic regression is the baseline model which does not capture temporal features among frames. We implement a fully connected network with one hidden layer. The each cell in output
layer uses the sigmoid function as activation function. Each label is treated as a binary classification
problem and sigmoid cross entropy is used as loss function.
## LSTM
We uses LSTM model to treat the input data as a sequence to obtain the feature in temporal feature. We use two layers of stacked LSTM, each layer contains 512 cells. For a 300-frame input feature, each cell takes a frame as input and output labels (N to One). The inbalance between positive and negative samples is significant in this data set. We implement focal loss to balance the weight of labels.
## NetVLAD
The idea of NetVLAD comes from the traditional computer vision descriptor VLAD. The
VLAD descriptor stands for "vector of local aggregated descriptor". Generally, we have a vector
representation of each local region in an image, i.e, HOG descriptor. However, simply concatenating
all local feature vectors within an image to form a global descriptor is not always a good idea becuase
the dimension of global feature would be very high in this case. Furthermore, the relationship
between each local descriptor cannot be addressed by simply concatenating descriptors one by
one. To solve this problem, VLAD first uses kmeans clustering to provide cluster representation
of features in local descriptor space. And then each local descriptor is subtracted from the cluster
center with which the descriptor is associated with. In this case, each local descriptor forms a vector
that points from its cluster center to itself. Aggregating all these vectors within a cluster will give a
compact representation in exact the same dimension as each local descriptor, and concatenating all
cluster representation gives us the vlad vector. 

# Result
We use 600000 videos to train our LSTM model and 10000 videos to test. Since the dataset is massive, we only train our model for 5 epochs with the learning rate of 0.001, batch size of 128, 0.05 of learning rate decay every 400000 examples and Adam optimizer. The total time used for training 5 epoch is over 12 hours. We uses three different indices to evaluate the performance of the model. GAP is the average precision on the top 20 labels per example. Hit@1 is the probability of the top 1 label being the ground truth label. PERR is the annotation precision when the same number of labels per video are retrieved as there are in the ground-truth.<br/> 
### Logistic Regression
![Logistic Regression](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/logistic.png)<br/>
### LSTM
![LSTM](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/lstm.png)<br/>
### NetVLAD
![NetVLAD](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/vald_4.png)<br/>
[More details](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/FinalReport.pdf) <br/>

# Reference
Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.<br/>
Abu-El-Haija, Sami, et al. "Youtube-8m: A large-scale video classification benchmark." arXiv preprint arXiv:1609.08675(2016).
Kim, Hyun Sik, and Ryan Wong. "Google Cloud and YouTube-8M Video Understanding Challenge.“<br/>
Miech, Antoine, Ivan Laptev, and Josef Sivic. "Learnable pooling with context gating for video classification." arXiv preprint arXiv:1706.06905 (2017).<br/>
Arandjelovic, Relja, et al. "NetVLAD: CNN architecture for weakly supervised place recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.<br/>

