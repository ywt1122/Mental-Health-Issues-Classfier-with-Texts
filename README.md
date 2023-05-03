# Mental-Health-Issues-Classfier-with-Texts
We use some basic machine learning and NLP technique to classify Mental Health Issues.


## Abstract

This machine learning project focuses on analyzing a mental health corpus dataset to classify the existence of mental health issues. The dataset contains a binary label, indicating the presence or absence of mental health issues. We mainly explored two classification methods, Random Forest and Neural Network. This project explores different text tokenization methods including CountVectorizer, Word2Vec, BERT (Bidirectional Encoder Representations from Transformers), and DistilBERT. We used 1500 instances in the dataset to generate BERT and DistilBERT embeddings due to computational constraints. In combination with Feed-Forward Neural Networks, our results show a similar accuracy across simple and complex models, achieving an accuracy around 90%. The project demonstrates the potential of machine learning in identifying mental health issues through text, which can aid in early detection and timely intervention.


##  Data Collection and Preparation
The Mental Health Corpus dataset was collected from Kaggle. This is a dataset of text related to mental health issues with one column for text and another for binary labels indicating toxicity. Label = 1 stands for having poisonous mental health issues. Data exploration was performed by checking the shape of the dataset, which contained 27,977 instances, and checking for missing values, which were not found. The dataset is well-balanced with 14139 entries with label = 0 and 13838 entries with label = 1. We used a bar chart to plot the top 20 most frequently used words. The 'text' column was transformed into numerical values using CountVectorizer to create a vocabulary of unique integers representing each word. The resulting features were transformed into an array and was split into an 80:20 training and testing set using train_test_split.

link: https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus?resource=download

## Methodology
The main algorithms used were logistic regression, random forest classifier, and Neural Networks. PCA was used to perform feature selection to accelerate model training efficiency. Word2Vec was another way to transform text into feature vectors other than CountVectorizer. We also used BERT embeddings to conserve more contextual relationships between texts. Grid search with cross validation was used to identify the optimal hyperparameters for random forest, and hyperparameter tuning and regularization optimized neural network performance.

## Baseline Model - Logistic Regression with CSR Matrix
To optimize memory usage, we converted the training and testing matrices into compressed sparse matrices using Scipy's csr_matrix class. This format stores only non-zero elements and their corresponding indices, allowing for efficient indexing and arithmetic operations. The logistic regression model was implemented using the LogisticRegression function from the sklearn.linear_model module. The model was trained using the reduced size feature obtained from CSR Matrix. The accuracy of the model was 91.69%.

## Feature selection with PCA (Principal Component Analysis)
PCA (Principal Component Analysis) was used to reduce the number of features in the dataset. Specifically, the number of components to keep was set to 1000. This was done to address the issue of limited computing resources, as reducing the number of features can significantly decrease the time and resources required for model training and prediction. The PCA function from the sklearn.decomposition module was used for feature reduction. The logistic regression model using the PCA features resulted in 91.17% of accuracy.

## Random Forest
Random Forest combines multiple decision trees to make a final prediction. It is useful for handling large and complex datasets with many features, and can capture intricate relationships between the features and the target variable. Grid search and cross validation were used to identify the optimal hyperparameters for random forest. The accuracy of the model was 82.15% using the following hyperparameters {'max_depth': 200, 'min_samples_split': 50, 'n_estimators': 300}.

## Neural Network
Neural networks excel at capturing complex patterns in data, including text. They can learn important features and relationships, and therefore, they are a good choice for this task. Keras from tensorflow was used for implementation. The model consisted of three fully connected layers with 64, 32, and 1 neurons. The activation function for the first two layers was relu, while the output layer was sigmoid. The model was compiled with the binary cross-entropy loss function and the stochastic gradient descent optimizer with a learning rate of 0.001. The model was trained for 50 epochs with a batch size of 32, and the accuracy of the model on the test set was 90.48%.

## Word2Vec Embedding with Random Forest and NN
The Word2Vec algorithm was used to generate feature vectors for each text in the text data. A function was defined to generate feature vectors for each comment. The feature vectors for the training and testing data were then generated and used to train a random forest classifier. The accuracy of the model was 90.14% with n_estimators=100. The same NN was used for the feature vectors with increased epochs (100) and reached accuracy of 90.49%

## BERT and DistilBERT embeddings with NN
BERT is based on the Transformer architecture, a type of deep learning model that is designed to handle sequential data. It learns contextual embeddings through a process called pre-training, which involves masked language modeling and next sentence prediction tasks. The key differences between BERT and Word2Vec are listed below.

## Contextual Understanding
While Word2Vec captures context through a fixed window size, BERT is capable of understanding context bidirectionally, resulting in a more accurate representation of word meanings.

## Pre-training
BERT uses pre-training to learn language representations, allowing it to be fine-tuned for  specific tasks with minimal training data. Word2Vec, on the other hand, requires the entire dataset to learn embeddings.

## Model Complexity
BERT is a more complex model with a deeper architecture, enabling it to capture intricate relationships between words. Word2Vec's shallow neural network, while effective for simpler tasks, may not be as robust in capturing complex language features.

## Neural Network Fine-Tuning
In order to prevent overfitting in our Neural Network models, we employed two regularization methods: Dropout and L2 Regularization. Dropout involves randomly dropping neurons (i.e. setting the value to zero) during training, which helps prevent co-adaptation and subsequently increases efficiency. L2 regularization, on the other hand, works by adding a penalty term to the loss function that discourages large weight values. The decision to combine both techniques was further supported by the findings of "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014), which demonstrated that the L2+dropout method had the second lowest error rate, as presented in the table below. We attempted to implement various combinations of regularization methods, including dropout, L2, and dropout + max_norm, but they resulted in lower accuracy compared to dropout + L2.



To find the optimal set of hyperparameters that yield the highest accuracy, different combinations were adjusted, such as the number of neurons in each layer, the regularization strength, the learning rate, and changing the number of epochs.
The Neural Network with DistilBERT embedding has four hidden layers with varying numbers of neurons, dropout rates, regularization strength for each layer. The learning rate of the Adam optimizer is set to 0.0001. The model is trained on the training set for 100 epochs with a batch size of 16 and evaluated on the test set. The accuracy of the model ranges from 88% - 90%.
The Neural Network with BERT embedding has eight hidden layers, a learning rate of 0.0001, and was trained on the training set for 100 epochs with a batch size of 16. The accuracy of the model ranges from 88% - 90%.

## Results

Model

Logistic (Baseline)
91.69%
Random Forest (CountVectorizer + PCA)
82.15%
Neural Network (CountVectorizer + PCA)
90.48%
Random Forest (Word2Vec)
90.14%
Neural Network (Word2Vec)
90.49%
Neural Network (DistilBERT)
88% - 90%
Neural Network (BERT)
88% - 90%

## Discussion
The results of this project show that logistic regression is a surprisingly strong baseline model for classifying the presence of mental issues using tokenized text, achieving an accuracy of 91.69%. While random forest performed worse than the neural networks, with an accuracy of 89%, the neural networks achieved an accuracy of 90%, which is only slightly better than the random forest model.

Surprisingly, the use of embedding methods such as Word2Vec and BERT did not improve the performance significantly. However, it is important to note that the BERT model was only trained on a small subset of the full dataset (around 5%), and it is possible that performance could improve if trained on the full dataset.

The fact that simple models like logistic regression can perform so well on this task may be due to the relatively simple nature of the dataset with only binary labels. While more complex models like neural networks and random forest have the ability to capture more complex nonlinear relationships between the input features and the target variable, in this case, the simple models were able to achieve high accuracy.

The limited computing resources used for training the models were the greatest constraints. In future work, it would be interesting to explore other embedding methods such as ELMo and GPT, and to train models on larger datasets with more varied labels. Additionally, exploring the use of transfer learning and pre-trained models could also be an interesting avenue for improving model performance.

## Conclusion
This project outlines the findings of the effectiveness of various machine learning techniques for classifying the presence of mental health issues using a mental health corpus dataset. A range of text tokenization and processing methods, including CountVectorizer, Word2Vec, BERT, and DistilBERT, were analyzed and subsequently applied in conjunction with both Random Forest and Neural Networks.




