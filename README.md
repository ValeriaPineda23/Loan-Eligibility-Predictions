# Loan Eligibility Predictions
Applying CRISP-DM methodology for predicting Loan Elegibility

## Business Understanding
The company Dream Housing Finance (DHF) deals in all home loans. They have a presence across all urban, semi-urban and rural areas. Customers first apply for a home loan after that company validates the customer's eligibility for a loan. The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling out an online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify the customer segments that are eligible for loan amounts to specifically target these customers that seem safe to borrow money from DHF. DHF owns data from its past operation that is regarded as one of its most valuable business assets. We can find the data in the *loan.csv* dataset.

The following report presents the CRISP-DM phases developed in a Jupyter Notebook (Python 3) using the libraries: *NumPy, pandas, matplotlib, seaborn, stats model, sklearn,* among others. We stored the *loan.csv* dataset in a data frame which we thoroughly manipulated to obtain the correct predictions of the prediction.csv.

## Data Understanding and Preparation


The dataset initially contained information on 618 customers who have asked for a loan to DHF. The information in these clients appears across 12 different independent features, both categorial and numerical.

Dataset Key Information
Key Name|Description
---|---
Loan_ID|	Unique Loan ID
Gender|	Male/ Female
Married|	Applicant married (Y/N)
Dependents|	Number of dependents
Education|	Applicant Education (Graduate/ Under Graduate)
Self_Employed	|Self-employed (Y/N)
ApplicantIncome	|Applicant income
CoapplicantIncome|	Coapplicant income
LoanAmount|	Loan amount in thousands
Loan_Amount_Term|	Term of a loan in months
Credit_History|	credit history meets guidelines
Property_Area|	Urban/ Semi-Urban/ Rural
Loan_Status|	Loan approved (Y/N)


First of all, we deleted the variable `Loan_ID` since it provided no meaningful information. Next, we changed the data type of `Credit_History` from numerical to categorical since we found out that its values only lay among 0 and 1. Where 0 meant that they did not have a credit history; otherwise, its value was one. Then, we divided the data frame into two datasets, one that contained the information of categorical variables and the other with the numerical ones. 

### Missing Data
#### Categorical Data
From here, we analyzed the percentage of missing data in each dataset. Five of the categorical variables had missing data (`Gender`, `Married`, `Dependents`, `Self_Employed`, and `Credit_History`). To correct this, we used the `SimpleImputer` functions from the *sklearn* library, where we imputed the missing values with the most frequent values from each column. 
#### Numerical Data
Then, we analyzed the numerical variables, two of which contained missing data. We use the `KNNImputer` that fills missing values through the implementation of the k-Nearest Neighbor approach to treat this issue.

### Balance Dataset
Additionally, we analyzed the data balance through the target variable Loan_Status, where we noticed that 422 customers were granted a loan and only 192 did not. The unbalanced target variable shows that the proportion of the category “Y” is 2.2 concerning category “N.” Thus, we proceeded into developing a random oversampling in the type “N,” which results in 422 customers for each category.

![Oversampling](https://user-images.githubusercontent.com/90649106/183491910-bcf14b04-c059-4351-a5b2-bb3b24751676.png)

### Label and one-hot encoding
Furthermore, we transformed the categorical variables to numerical by implementing label and one-hot Encoding (to those with more than two unique variables). After this transformation, the categorical data lied between 0 and 1; thus, we needed to apply a min-max scaling transformation to numerical data. With this transformation, the magnitude of the numerical values is equal to the categorical.

### Exploratory Data Analysis

#### Categorical Data
Next, we performed the Exploratory Data Analysis (EDA) for the categorical data. For this analysis, we developed a series of bar plots that convey the frequency of the unique values of each variable. Additionally, we also developed a plot with the proportions regarding if the loans had been accepted or not. From these visualizations, we concluded that men ask more for loans than women; however, both genders have the same probability of being granted a loan. Therefore, gender is not a determining factor in predicting if a loan should be granted or not.

Furthermore, more married clients ask for loans, and they have a higher probability of being accepted. Customers who have no dependents have a significantly higher likelihood of asking for a loan. Still, curiously, if the client has two dependents, they have a higher probability of being granted the loan. Moreover, if the applicant is not graduated from college, they have a lower likelihood of being accepted for a loan. The customer's credit history is the most significant variable since it shows that the customers who have a credit history appear to have an almost 80% probability of being granted a loan. However, if the customer does not have a credit history, they have only a 7.9% chance of getting a loan. Finally, suppose the property is in a suburban area. In that case, the client has a higher probability of being granted a loan than if the property was in an urban or rural area.

#### Numerical Data
Furthermore, for the numerical data, we first visualized the distribution of each variable through a histogram, where we see that most of them are skewed to the right, except for the `Loan_Amount_Term` variable. Additionally, we developed a series of boxplots that describe the distribution of the data of each numerical variable and its loan status. From these plots, we can see that all of them convey a very similar distribution between the data of the accepted and rejected loans. Thus, they appear to be non-meaningful for predicting the target variable. For more in-depth information, we developed a series of ANOVA F-tests to prove that the mean of each numerical variable for accepted loans is the same as that for unaccepted. The results of these tests prove that it seems that the numerical data does not influence the decision of whether the loan is accepted or not.


Variable|p-value|Null Hypothesis
---|---|---
ApplicantIncome |0.390 |Not rejected 
CoapplicantIncome |0.072 |Not rejected 
LoanAmount |0.66 |Not rejected 
Loan_Amount_Term|0.328|Not rejected


#### Correlation Analysis
To finalize EDA, we developed a correlation analysis heatmap, where we convey the linear relationship in a correlation matrix among all variables. This heatmap shows that the categorical variables resulting from the one-hot encoding (`Dependents_0`, `Dependents_1`, `Dependents_2`, `Dependents_3`, `Property_Area_0`, `Property_Area_1`, `Property_Area_2`) have a pronounced negative linear correlation among them. However, we can see that married status has a 37% correlation with gender. If the person is male, they have a higher probability of also being married. Additionally, there is a 25% correlation between being married and having two dependents, which makes sense since most of the time, married couples who have a growing family, tend to ask for loans to buy a bigger house. Furthermore, the amount of money being loaned has a 57% correlation with the applicant's income. This conclusion makes sense since the more an applicant gains, the more they would be able to pay; thus, the more they are loaned.

Finally, we analyzed the variables that appeared to correlate significantly with our target variable, `Loan_Status`. We see that people who are asking for loans in the urban area have an 18% probability of being granted the loan. Additionally, if the applicant is married, they have a 13% probability of having an accepted loan. Also, we can conclude that having a credit history has the most influence on whether the loan is accepted or rejected, with a 51% correlation.

![Correlation Matrix](https://user-images.githubusercontent.com/90649106/183494711-e259ee89-f7aa-44b0-8099-1c129a30b8d6.png)

Furthermore, we performed *Kendall’s Rank Correlation* hypothesis test to determine whether the correlation between the categorical target variable `Loan_Status` and the independent numerical variables is significant. For these tests, the null hypothesis would be that both variables are independent. Therefore, this analysis concludes that the numerical variables seem to be independent of the target variable.

Variable|correlation|p-value|Null Hypothesis
---|---|---|---
ApplicantIncome |0.004 |0.890 |Not rejected 
CoapplicantIncome |0.057 |0.059 |Not rejected 
LoanAmount |-0.007 |0.796 |Not rejected 
Loan_Amount_Term|-0.020|0.549|Not rejected 

Moreover, we developed the $Xi^2$ hypothesis tests to evaluate the correlation between the target and categorical variables (see Table 5). For which we obtained the results that appear in Table 5. According to the results, we can see that the variables `Married`, `Education`, `Credit_History`, `Dependents_2`, `Property_Area_1`, and `Property_Area_2` seem to have a significant correlation with the target variable.

Variable|stat|p-value|Null Hypothesis
---|---|---|---
Gender |1.279 |0.258 |Not reject 
Married |13.594 |0.000 |Reject 
Education |8.027 |0.005 |Reject 
Self_Employed |0.476 |0.490 |Not reject 
Credit_History |214.454 |0.000 |Reject 
Dependendents_0 |0.000 |1.000| Not reject 
Dependendents_1 |1.013| 0.314| Not reject 
Dependendents_2 |3.242| 0.072| Reject 
Dependendents_3 |0.726| 0.394| Not reject 
Property_Area_0 |2.102 |0.147| Not reject 
Property_Area_1 |25.870 |0.000| Reject 
Property_Area_2 |12.630 |0.000 |Reject

We developed an RFE feature selection strategy to finalize the data preparation phase. Through this technique we found the most significant variables that predict the `Loan_Status`. We first got the optimum number of features according to a Logistic Regression model. This analysis concluded that we required only six variables to achieve the highest accuracy possible. Next, we applied for this number again to the logistic regression. As a result, we obtained that `Married`, `Credit_History`, `Property_Area_1`, `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount` were the most valuable variables to predict `Loan_Status`. 

### Multicollinearity
Now that we have this information, we developed a VIF analysis of the significant variables, from which we obtained the results below. 

Variable|stat
---|---
Married|2.3288
Credit_History|4.2510
Property_Area_1|1.4974
ApplicantIncome|3.1519
CoapplicantIncome|1.4250
LoanAmount|5.8079

This analysis tells us that the essential variables in the dataset do not contain multicollinearity problems, so all of these may be part of the final data to feed the model.

## Modeling
For this phase, we applied decision trees, logistic regression, support vector machines, Naïve Bayes, and ensemble methods bagging, boosting, and XGB models to predict the target variable Loan_Status. We first divided the data into the dependent and independent variables, called y and X, respectively, to evaluate these models. Then, we divided them into training and test set, with an 80%-20% ratio.

## Evaluation
For the evaluation of the algorithms, we developed a K Fold Cross-Validation such as in the figure below. This technique consists of the training set partitioning into equal-sized K parts. One of these partitions is used as a validation set and the rest as a sub-training set. For this strategy, we defined a total of 10 partitions. 

![k-fold](https://user-images.githubusercontent.com/90649106/183498743-d77cbf70-dfff-40d2-90f5-f3669d6b8598.png)


Now we show the mean of the metrics for each algorithm in terms of ROC/AUC, accuracy, precision, recall, and f1 score.

Algorithm|ROC AUC Mean|Accuracy Mean|Precision Mean| Recall Mean| F1 Score Mean
---|---|---|---|---|---
Random Forest|	92.65|	85.37|	87.00|	83.60|	84.98
XGB|	90.82	|84.07|	85.57	|81.99|	83.51|
GBM|	87.69|	80.93|	77.17	|86.98	|81.66|
Decision Tree Classifier|	84.50	|84.44|	86.52|	81.36	|83.73
Logistic Regression|	81.00|	72.41	|66.97	|88.72	|75.91
SVM|	79.10|	71.11	|66.13	|87.99	|74.66
Naïve Bayes|	78.81|	73.33	|65.91|	96.97	|78.31

Next, we convey the boxplots that contain the distribution of the accuracy score, and ROC/AUC score for each algorithm. From these plots, we can conclude that some of the best models for fitting these data are GBM, XGB, Decision Tree, and Random Forest. These models achieve significantly higher accuracy and ROC/AUC scores than others.

![ROC AUC](https://user-images.githubusercontent.com/90649106/183499620-143287df-4619-49ab-9db3-849ddd997331.png)
![Accuracy](https://user-images.githubusercontent.com/90649106/183499629-78eed8dd-b64a-4010-81f9-48ba8ff19984.png)

### Hyperparameter Tuning
For the next iteration, we developed some hyperparameter tuning for the algorithms that attained the best accuracy and AUC scores (GBM, XGB, Decision Tree, and Random Forest) so that they may further improve their metrics. The hyperparameter tuning was developed through the use of GridSearchCV, this function allowed us to try different values for several parameter. Where the results were the following:
- The best parameters for Random Forest are {'criterion': 'gini', 'max_depth': 13, 'min_samples_leaf': 1, 'n_estimators': 45}
- The best parameters for GBM are {'learning_rate': 0.1, 'n_estimators': 250, 'max_depth': 4}
- The best parameters for Decision Tress are {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 5}
- The best parameters for XGB are {'gamma': 0.1, 'max_depth': 15, 'min_child_weight': 1}

The resulting metrics of these changes appear below:
Model|	ROC/AUC	|Accuracy	|Precision|	Recall|	F1 Score
---|---|---|---|---|---
Random Forest|0.928445|	0.866667|	0.878378|	0.878378|	0.878378
Gradient Boosting|	0.919805|	0.844444	|0.884058|	0.824324	|0.853147
XGB	|0.888569	|0.829630	|0.892308	|0.783784	|0.834532
Decision Tree|	0.849801	|0.785185	|0.816901|	0.783784	|0.800000
Support Vector Machine	|0.763181	|0.718519	|0.695652	|0.864865	|0.771084
Logistic Regression	|0.753434	|0.711111	|0.701149|	0.824324|	0.757764
Naive Bayes	|0.745680	|0.718519	|0.680000	|0.918919	|0.781609

This table shows that the Random Forest became the optimal model to make predictions for the given dataset, as it has relatively the highest combination of AUC, accuracy, precision, recall, and F1 scores. Hence, we used this model to evaluate its performance when training in the entire train dataset and testing in the true testing dataset.

Model|	ROC/AUC	|Accuracy	|Precision|	Recall|	F1 Score
---|---|---|---|---|---
Random Forest|	0.920244|	0.87574|	0.857143	|0.888889	|0.872727


![Cross Validation](https://user-images.githubusercontent.com/90649106/183500834-ba6e385c-13ab-47a3-81c2-cbd5d5878119.png)

## Deployment
We applied the best model obtained to make predictions using the attached *predict.csv* as the test set and *loan.csv* as training set. We created a new Jupyter Notebook (CRISP-DM (Deployment Phase).ipynb) that cleans the *predict.csv* dataset by imputing missing values, labeling and applying one-hot encoding on categorical data, transforming continuous values into integers, and selecting the essential features. The predictions for each user appear in *Model Predictions.xlsx*.

# Conclusion
In conclusion, we used the Dream Housing Finance Loan dataset to build a machine learning classifier to automate the loan eligibility process. This model attained a reasonable accuracy score of 87.5%. Additionally, according to this analysis, we can conclude that the customer segments that DHF should target are applicants that appear to be married and are looking for a property in the suburban area. This situation could mean that they may be planning to grow a family; thus, they have a higher probability of being responsible for avoiding debts. Furthermore, these applicants and their co-applicants, should count on a high amount of income. If DHF targets people who follow these characteristics, they can ensure that customers will be capable of paying back; hence, DHF will be more secure in lending a higher amount of money to them. Finally, and most importantly, ensure that the person has a credit history because applicants who have repaid their previous debts have a significantly higher probability of repaying this one.
