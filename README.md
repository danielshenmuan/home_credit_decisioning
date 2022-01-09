# home_credit_decisioning
This project tackles the decision of which recipients to give out home credit loan using classification machine learning methods. It also identifies important attributes to applicants' default probability

## Business Problem
Many people struggle to get loans due to insufficiency in their credit history. And, unfortunately, this population is often taken advantage of by untrustworthy lenders. We strive to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience.

## Data Cleaning
- Dropped columns with more than 60% data missing
- Impute values: Mode for categorical columns and mean for numerical ones
- Convert numerical columns with less than 5 unique values to categorical columns
- Drop columns which have near 0 variation in data
- Train-validation splits (70/30) validation set used as out of sample

## Models
- Logistic Regression
- K Nearest Neighbor 
- Bagging
- Random Forest 
- Boosting

## Conclusion
- Best model is Boosting with specificity of 70% and sensitivity of 69% meaning we can successfully predict with 70% accuracy if a person will default his/her loan

