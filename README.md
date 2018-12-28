# kaggle_projects
Repository for Kaggle Projects and Competitions


**1. Submitted for Quora Insincere Questions Classification Competition: <br />
An Implementation of 'Hierarchical Attention Networks for Document Classification' by Yang et al. (2016)**

A Tensorflow implementation for the Hierarchical Attention Network proposed by Yang et al. (2016) was used in the Kaggle competition 'Quora Insincere Questions Classification':https://www.kaggle.com/c/quora-insincere-questions-classification/data <br /> <br />
The highest F1 score achieved in the competition currently by this model is 0.652 (ongoing effort!). <br /> Please note that this score was achieved by using only ~500,000 training records in mini-batches of 128, out of the 1.3 million provided. <br/> The constraint is in the GPU time allocated by the competition (only 2 hours). <br/> It may be possible to get a higher score in the competition if it were possible to train longer! 
<br /> <br /> 
For comparison, the highest score in the competition currently is 0.711.
<br /> <br />
The F1 score progression on a holdout dataset below. Note that the dataset is highly imbalanced (6% positive class only) <br/>
![alt text](https://github.com/nitinvwaran/kaggle_projects/blob/master/f1_score_valid.PNG) <br /> <br />

The tensorboard graph is as below: <br/>
There are two components to the graph: <br/>
1) The Hierarchical Attention Network component, implemented as proposed by Yang et al. (2016). <br/>
2) To extract more features, a 'hierarchical CNN' component was used, which first extracts n-grams of window sized 3,4,5 from each sentence in a  document, and does a global maxpooling over time for each window extraction, before concatenating the features. This approach is similar to the approach followed in 'Convolutional Neural Networks for Sentence Classification' by Yoon Kim (2014). <br/>
Following this extraction, the concatenated features which are at sentence level, are rolled up into their document, and a reduction by max is applied across all the sentences in the document, essentially preserving all the features extracte earlier, but only for the most prominent sentence. <br /> <br/>

Features from both components are concatenated, and passed through a dense layer before a logistic classifier <br/> <br/>

The graph as described above is visualized below on tensorboard <br />
![alt text](https://github.com/nitinvwaran/kaggle_projects/blob/master/tensorboard_graph.PNG) <br /> <br />

To find out more about the technical implementation of this project, please visit my Linkedin post: https://www.linkedin.com/pulse/implementation-hierarchical-attention-network-nitin-venkateswaran/
