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




