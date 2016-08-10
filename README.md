# Detecting Person of Interest in Enron Dataset

Enron is the biggest corporate scandal in the U.S. history. This report documents how I apply different machine learning algorithms to identify persons of interest in the Enron case based on financial and email data made public as a result of the scandal. With the testing data of 146 observations and 21 features, I have completed following steps to identify the best algorithms

- Explored the data and exclude the outliers;
- Excluded variables with large proportion of NAs; Created new features and demonstrate the importance of new features; Using SelectK to further exclude some unrelated features;
- Applied three models (Gaussian Naive Bayes, Support Vector Classifier, and RamdonForest) and apply stratified cross validate 1000 times and grid search to select the models.
- Applied the metrics of precision and recall since the data is largely skewed, and the best performing model is Gaussian Naive Bayes



The training dataset is stored in my_dataset.pkl. The training script is poi.id.py and the detail description can be find in Enron_email_final_project_document.
