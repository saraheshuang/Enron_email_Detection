# Detecting Person of Interest in Enron Dataset

Enron is the biggest corporate scandal in thei U.S. history. This report documents how I apply different machine learning algorithms to identify persons of interest in the enron case based on financial and email data made public as a result of the scandal. With the testing data of 146 observations and 21 features, I have completed following steps to identify the best algorithms

- Explored the data and exclude the outliers;
- Excluse variables with large propotion of NAs; Create new features and domenstrate the importance of new features;  Using SelectK to further exclude some unrelated features;
- Apply three models (Guassian Naive Bayes, Support Vector Classifier, and RamdonForest) and apply stratified cross validate 1000 times and grid search to select the models.
- The metrics of precision and recall are applied as the data is largly skewed, and the best performing model is Guassian Naive Bayes

The training dataset is stored in my_dataset.pkl. The training script is poi.id.py and the detail description can be find in Enron_email_final_project_document.
