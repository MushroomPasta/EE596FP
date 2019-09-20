# EE596 Final Project
# Youtube-8M video semantic multi-label classification 
  by Tianyi Zhang and Xiulong Liu
# Purpose
  Video semantic classification means to extract the situation or context from videos by processing image features. Each video may contain multiple labels.
# Data
[Youtube-8M link](https://research.google.com/youtube8m/)<br/>
Youtube8M dataset contains over 6 million videos with 3862 labels. The average number of label is 3 per video. Two types of features are available, frame-level and video-level. 
# Approaches
## Logistic Regression
LR is our baseline model which does not capture temporal features among frames.
## LSTM
We uses LSTM model to treat the input data as a sequence to obtain the feature in temporal feature. We use two layers of stacked LSTM, each layer contains 512 cells. For a 300-frame input feature, each cell takes a frame as input and output labels (N to One). The inbalance between positive and negative samples is significant in this data set. We implement focal loss to balance the weight of labels.
## NetVLAD


# Result
We use 600000 videos to train our LSTM model and 10000 videos to test. Since the dataset is massive, we only train our model for 5 epochs with the learning rate of 0.001, batch size of 128, 0.05 of learning rate decay every 400000 examples and Adam optimizer. The total time used for training 5 epoch is over 12 hours. We uses three different indices to evaluate the performance of the model. GAP is the average precision on the top 20 labels per example. Hit@1 is the probability of the top 1 label being the ground truth label. PERR is the annotation precision when the same number of labels per video are retrieved as there are in the ground-truth.<br/> 
[Logistic Regression](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/logistic.png)
[LSTM](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/lstm.png)
[NetVLAD](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/vald_4.png)
[More details](https://github.com/TianyiZhang0315/EE596FinalProject/blob/master/FinalReport.pdf) 

# Reference
Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.<br/>
Abu-El-Haija, Sami, et al. "Youtube-8m: A large-scale video classification benchmark." arXiv preprint arXiv:1609.08675(2016).
Kim, Hyun Sik, and Ryan Wong. "Google Cloud and YouTube-8M Video Understanding Challenge.“<br/>
Miech, Antoine, Ivan Laptev, and Josef Sivic. "Learnable pooling with context gating for video classification." arXiv preprint arXiv:1706.06905 (2017).<br/>

