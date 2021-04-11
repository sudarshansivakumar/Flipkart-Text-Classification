# MIDAS-IIITD-Flipkart

Repository for submitting the task required for the selection process for a summer internship at the MIDAS lab in IIIT Delhi. The given dataset has been scraped from the Flipkart Website, and consists of product descriptions and categories. The goal is to build a model which classifies a given  description into a particular category. For instance, some of the categories that are present in the dataset include Clothing, Jewellery, etc.

## Requirements 

* PyTorch 1.8.1
* TorchText 0.9.0
* Python 3 
* NLTK 

There are two models that we train in this (both are CNN based text classification models but the difference is in the datasets used). 
For the first model, which in the notebook is reffered to as CNNmodel() the datasets are present in the 'Model 1' folder under the names 'flipkart_dataset_train.csv', 'flipkart_dataset_valid.csv', and 'flipkart_dataset_test.csv'. The process of how the dataset has been created from the original dataset which was given has been detailed in the notebook. 
For the second model which in the notebook is referred to as CNNmodel2() the datasets are present in the 'Model 2' folder under the names 'train_final.csv','valid_final.csv', and 'test_final.csv'. Again the process of how we form these datasets from the datasets used for Model 1 is described in the notebook. 
In order to reproduce the results the models can be trained using the datasets present in these folders


The approach followed for the problem overall is as follows :

  * First the primary category of each product description is extracted. This is done by applying regex commands on the dataframe column 'product_category_tree'. If a given product belongs to multiple categories, that too is extracted and the dataframe is modified in such a way that for every row in the dataframe we have one primary category that the description of the row belongs to. If a given product belongs to two categories, it would occupy two rows
  * Once we manage to extract the primary category for each product, we pre-process the product description. Stopwords, punctuations, digits, and some common words which feature across categories are removed from the product description. This would allow for easier predictions, and would make the task of classification easier. This processed dataframe is then divided into train,test, and validation sets
  * We then proceed to use TorchText to solve the rest of the task. We create TorchText datasets and iterators from the processed dataset described above
  * The model accuracy is measured using F1 scores. Since we have an imbalanced dataset, F1 scores would be a better metric than just overall accuracy to gauge how well our model is doing.
  * In order to classify the descriptions we use a CNN architecture - convolutions over the product descriptions are done using filters of sizes 2,3,4,5, and 6, and the convolutional layer is followed by a fully connected layer to give us the final class predictions 
  * After training the first model, we look at the F1 scores of categories, and justify why certain categories can be removed. So a new dataset is saved, from which these categories have been removed
  * The second and final model is trained on this dataset, with reasonably accurate predictions on the test set 

Some of the tutorials/resources that helped me complete the task:
* [Github Repo on Sentiment Analysis with PyTorch](https://github.com/bentrevett/pytorch-sentiment-analysis)
* [Kaggle Notebook on Text Preprocessing](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
