# Assignment 8 - Logistic regression

#This document performs Logistic Regression on the 
#Heart Failure Prediction dataset. We iterate over multiple
#models to come up with the most robust model.

# Let us load libraries and data

# clear environment
rm(list = ls())

# defining libraries

library(ggplot2)
library(dplyr)
library(PerformanceAnalytics)
library(data.table)
library(sqldf)
library(nortest)
library(MASS)
library(rpart)
library(class)
library(ISLR)
library(scales)
library(ClustOfVar)
library(GGally)
library(reticulate)
library(ggthemes)
library(RColorBrewer) 
library(gridExtra)
library(kableExtra)
library(Hmisc) 
library(corrplot)
library(energy)
library(nnet)
library(Hotelling)
library(car)
library(devtools)
library(ggbiplot)
library(factoextra)
library(rgl)
library(FactoMineR)
library(psych)
library(nFactors)
library(scatterplot3d)
library(lmtest)
library(mctest)
library(aod)
library(InformationValue)
library(pROC)
library(tidyverse)
library(caret)
library(Information)

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)

# Fitting a logistic regression model

#We recall three key points from third assignment (EDA) -
  
#1. Our data has no missing values \
#2. We saw 4 observations as outliers \
#3. We saw no multicollinearity as our VIF values were all below 2 \

#Hence we have a very small pre-processing step of removing outliers.

# Data Cleaning - Let's remove these outliers
data <- data[data$ejection_fraction <70,]
data <- data[data$creatinine_phosphokinase <7000,]
str(data)

#We remove the 4 outliers before proceeding to modeling exercise.

### Converting categorical features and dependent variable to factor

data$DEATH_EVENT <- factor(data$DEATH_EVENT)
data$anaemia <- factor(data$anaemia)
data$diabetes <- factor(data$diabetes)
data$high_blood_pressure <- factor(data$high_blood_pressure)
data$sex <- factor(data$sex)
data$smoking <- factor(data$smoking)
str(data)


### Two-way contingency table of categorical outcome and predictors 
### Since we want to make sure there are not 0 cells


xtabs(~DEATH_EVENT + anaemia, data = data)
xtabs(~DEATH_EVENT + diabetes, data = data)
xtabs(~DEATH_EVENT + high_blood_pressure, data = data)
xtabs(~DEATH_EVENT + sex, data = data)
xtabs(~DEATH_EVENT + smoking, data = data)

#\
#We note no 0 or low cells in any of the categorical variables.

# Checking event rate in data - this will help determine prob. value for thresholds

table(data$DEATH_EVENT)
#We note a 31.5% event rate in the data.

# Fitting a model

# Iteration - 1 All variables
# Model 1
mylogit <- glm(DEATH_EVENT ~ age+anaemia+creatinine_phosphokinase+
                 diabetes+ejection_fraction+high_blood_pressure+platelets+
                 serum_creatinine+serum_sodium+sex+smoking+time , data = data, family = "binomial")
summary(mylogit)

### Key observations  

#1. We note age, ejection_fraction, serum_creatinine and time as 
#significant variables in this iteration 
#2. The above iteration has an AIC of 241.95
#3. Interpreting the coefficient - For every one unit change in age, 
#the log odds of death (versus survival) increases by 5.197e-02
#4. None of the categorical variables are significant in
#predicting the death event

### Predicting the outcome

glm.probs <- predict(mylogit,type = "response")
glm.probs[1:5]

#The first five probabilities in this case are all close to 1 as evidenced in the
#data as well.

### Let's use a base calculation to figure out accuracy

# Here we try the case of using default 0.5 as threshold
glm.pred <- ifelse(glm.probs > 0.5, "1", "0")
table(data$DEATH_EVENT,glm.pred)


#\
#Looking at the diagonal, we're not that bad.
#We have an overall accuracy of ~85% (Diagonals summed over overall sum)
#But note this is called a base model because we didn't do
#splitting into train and test so the model trained on the entire data
#and predicted on the entire data. Without out of sample testing
#one cannot claim robustness as there may be overfitting here.

### Deciding on optimal cutoff
optCutOff <- optimalCutoff(data$DEATH_EVENT, glm.probs)[1] 
optCutOff

#\
#We used 0.5 above to classify however we want the cut-off where
#model is balanced in accuracy measures. We note this as 0.466.
#This tells us that we can use this prob cutoff to classify an 
#observation as 0 or 1. If the prob-value is below this threshold we
#can classify it as survival else death event.

### Mis-classification error
misClassError(data$DEATH_EVENT, glm.probs, threshold = optCutOff)

#\
#We note a misclassification error of 14.58%

### ROC curve
plotROC(data$DEATH_EVENT, glm.probs)


#\
#We see an AUC of 0.894 which is decent.

### KS plot
ks_stat(data$DEATH_EVENT, glm.probs)
ks_plot(data$DEATH_EVENT, glm.probs)

#\
#A KS plot answers the question how many responders/ deaths can 
#we capture if we target x% of the population. Here, we see
#we can capture 73% responders if we target first 30% of the population.

### Confusion Matrix and all accuracy measures
confusionMatrix(data = as.factor(glm.pred),
                reference = as.factor(data$DEATH_EVENT), mode = "prec_recall")

#\
#We have to be careful here. From above, we can see
#accuracy of the model is 85.0% as before. Precision is 0.87
#and Recall is 0.91 while F1- score is 0.89 but this is for
#positive class taken as '0'. However, we want to understand
#precision, recall for positive class as predicting death
#events is more important than survival events.

### Confusion Matrix and all accuracy measures for positive class chosen as death=1
confusionMatrix(data = as.factor(glm.pred), 
                reference = as.factor(data$DEATH_EVENT),positive='1', mode = "prec_recall")

#\
#We give positive class as '1' as we want to understand precision
#and recall of death events more than survival events so we know
#what to maximize for. Our accuracy of the model is 85.0% as before
#however precision is 0.79 and recall is 0.70 
#while F1- score is 0.75. This is a more true picture of our
#model than before and we know that the overall
#accuracy is slightly dominant towards predicting survival
#events better than death events.

### Using optimal cutoff to determine accuracy measures
threshold=optCutOff
predicted_values<-ifelse(predict(mylogit,type="response")>threshold,1,0)
actual_values<-data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), mode = "prec_recall")

#\
#Our new accuracy has gone up from 85% to 85.4% all by optimizing the 
#prob. cutoff threshold. 

### Using optimal cutoff to determine accuracy measures with positive class as 1
threshold=optCutOff
predicted_values<-ifelse(predict(mylogit,type="response")>threshold,1,0)
actual_values<-data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values),
                reference = as.factor(actual_values), 
                positive='1',mode = "prec_recall")

#\
#Here, we see that we have improved our recall while keeping precision same
#and hence consequently our F1 score. 

#\
#However we may be over-fitting here as we haven't kept a hold out set.
#This is something we will explore in future iterations.

# Iteration - 2 Using only significant variables from Iteration 1
# Model 2
mylogit_2 <- glm(DEATH_EVENT ~ age
    +ejection_fraction+
    serum_creatinine+time , data = data, family = "binomial")
summary(mylogit_2)

#\ 
#We note the lower AIC value in this iteration of 232.4

### Predicting the outcome
glm.probs_2 <- predict(mylogit_2,type = "response")
glm.probs_2[1:5]

#The first five probabilities in this case are all close to 1 as evidenced in the
#data as well.

### Let's use a base calculation to figure out accuracy
# Here we try the case of using default 0.5 as threshold
glm.pred_2 <- ifelse(glm.probs_2 < 0.5, "0", "1")
table(data$DEATH_EVENT,glm.pred_2)

#\
#Looking at the diagonal, we have an overall accuracy of ~83% 
#(Diagonals summed over overall sum). This is lower than before
#which makes sense since we removed the unnecessary independent variables
#But how do we know this is more robust than the model before ?
  
  ### Deciding on optimal cutoff
  
optCutOff <- optimalCutoff(data$DEATH_EVENT, glm.probs_2)[1] 
optCutOff

### Mis-classification error
misClassError(data$DEATH_EVENT, glm.probs_2, threshold = optCutOff)

#\
#We note a misclassification error of 14.92%

### ROC curve
plotROC(data$DEATH_EVENT, glm.probs_2)

#\
#We see an AUC of 0.888 which is decent.

### KS plot
ks_stat(data$DEATH_EVENT, glm.probs_2)
ks_plot(data$DEATH_EVENT, glm.probs_2)

### Confusion Matrix and all accuracy measures
confusionMatrix(data = as.factor(glm.pred_2), 
                reference = as.factor(data$DEATH_EVENT), mode = "prec_recall")

#\
#Our new accuracy is 83.0%. This is slightly lower than before
#which makes sense since we have eliminated some independent variables
#in this iteration. 

### Confusion Matrix and all accuracy measures for positive class chosen as death=1
confusionMatrix(data = as.factor(glm.pred_2),
                reference = as.factor(data$DEATH_EVENT),positive='1', mode = "prec_recall")

#\
#Our accuracy of the model is 83.0% as before
#however precision is 0.75 and recall is 0.67
#while F1- score is 0.71. This is a more true picture of our
#model than before and we know that the overall
#accuracy is slightly dominant towards predicting survival
#events better than death events. We can see that all precision,
#recall, F1 score, accuracy and AUC are slightly lower in
#this iteration.

### Using optimal cutoff to determine accuracy measures
threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_2,type="response")>threshold,1,0)
actual_values<-data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values),
                reference = as.factor(actual_values), mode = "prec_recall")

#\
#On using the optimal cutoff however, Our new accuracy 
#has gone up from 83% to 85.0% all by optimizing the 
#prob. cutoff threshold. 

### Using optimal cutoff to determine accuracy measures with positive class as 1
threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_2,type="response")>threshold,1,0)
actual_values<-data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), 
                positive='1',mode = "prec_recall")

#\
#Here, we see that we have improved our pecision however
#recall is much worse. 

#The problem in the first two iterations however is that we
#haven't kept a test set and may have over-fitted the model
#unknowingly.

# Iteration - 3 All variables but splitting into train and test

### Splitting into train and test - 70%, 30% split
set.seed(123)
trainIndex <- createDataPartition(data$DEATH_EVENT, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train_data<-data[trainIndex,]
test_data<-data[-trainIndex,]
table(train_data$DEATH_EVENT)
table(test_data$DEATH_EVENT)

#We see our train data has event rate of 31.7% and our test data
#has event rate of 31.0%. This can happen in reality as well and 
#hence a good accuracy on test will ensure we have built a robust 
#model.

#We will now train the model on training set and test on test set.

# Model 3
mylogit_3 <- glm(DEATH_EVENT ~ age+anaemia+creatinine_phosphokinase+
    diabetes+ejection_fraction+high_blood_pressure+platelets+
    serum_creatinine+serum_sodium+sex+smoking+time ,
    data = train_data, family = "binomial")
summary(mylogit_3)
predicted <- predict(mylogit_3, test_data, type="response")

#We note a key difference here - we do not see
#the serum_creatinine as a significant variable in this iteration.
#We see only age, time and ejection_fraction as significant variables.

### Deciding on optimal cutoff

optCutOff <- optimalCutoff(test_data$DEATH_EVENT, predicted)[1] 
optCutOff


#\
#This tells us that we can use this prob cutoff to classify an 
#observation as 0 or 1. If the prob-value is below this threshold we
#can classify it as survival else death event.

### Mis-classification error

misClassError(test_data$DEATH_EVENT, predicted, threshold = optCutOff)


#\
#We note a misclassification error on test set of 17.24%

### ROC curve

plotROC(test_data$DEATH_EVENT, predicted)


#\
#We see an AUC of 0.868 which is decent.

### KS plot

ks_stat(test_data$DEATH_EVENT, predicted)
ks_plot(test_data$DEATH_EVENT, predicted)



### Concordance check

Concordance(test_data$DEATH_EVENT, predicted)

#\
#Usually concordance is in-line with AUC and we see that 86.8% pairs
#are concordant (the model calculated prob scores of 1s being greater than
#model calculated prob scores of 0s)

### Using optimal cutoff to determine accuracy measures

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_3,test_data,type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values),
            reference = as.factor(actual_values), mode = "prec_recall")


#\
#We see that on test set our accuracy is 82.7% 

### Using optimal cutoff to determine accuracy measures with positive class as 1

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_3, test_data, 
                                 type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), 
                positive='1',mode = "prec_recall")

#\
#We however note a key difference in this accuracy.
#Our recall has fallen to 0.59 while the precision is 0.80
#with F1-score of 0.68



# Iteration - 4 Stepwise regression

#\
#We now perform a stepwise regression which computes a null model and a
#full model first and adds variables as long as the added variable's AIC
#is below the previous computation of AIC.


null_model<-glm(DEATH_EVENT~1,data=train_data,family='binomial')
full_model<-glm(DEATH_EVENT~.,data=train_data,family='binomial')
step_model <- step(null_model, 
                   scope = list(lower = null_model,
                                upper = full_model),
                   direction = "forward")


#\
#We see the results of the stepwise regression lowering our AIC to 164.14
#with variables like time, ejection_fraction, serum_creatinine, age,
#sex and serum_sodium


summary(step_model)

#\ 
#We see a lower AIC but unfortunately serum_sodium isn't significant.
#We can remove this variable and re-compute the ideal model.


# Model 4
mylogit_4 <- glm(DEATH_EVENT ~ time + ejection_fraction + serum_creatinine + 
    age + sex , data = train_data, family = "binomial")
summary(mylogit_4)

#\
#We note our lowest AIC yet of 164.78

### Let's predict the outcome for this model

predicted <- predict(mylogit_4, test_data, type="response")

### Deciding on optimal cutoff

optCutOff <- optimalCutoff(test_data$DEATH_EVENT, predicted)[1] 
optCutOff


#\
#This tells us that we can use this prob cutoff to classify an 
#observation as 0 or 1. If the prob-value is below this threshold we
#can classify it as survival else death event.

### Mis-classification error

misClassError(test_data$DEATH_EVENT, predicted, threshold = optCutOff)


#\
#We note a misclassification error on test set of 14.9%

### ROC curve

plotROC(test_data$DEATH_EVENT, predicted)


#\
#We see an AUC of 0.876 which is our best yet on test data.

### KS plot

ks_stat(test_data$DEATH_EVENT, predicted)
ks_plot(test_data$DEATH_EVENT, predicted)


### Concordance check

Concordance(test_data$DEATH_EVENT, predicted)

#\
#Usually concordance is in-line with AUC and we see that 87.6% pairs
#are concordant (the model calculated prob scores of 1s being greater than
#                model calculated prob scores of 0s)

### Using optimal cutoff to determine accuracy measures

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_4,test_data,
                                 type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), 
                mode = "prec_recall")


#\
#We see that on test set our accuracy is 85.0% 

### Using optimal cutoff to determine accuracy measures with positive class as 1

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_4, test_data,
                                 type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), 
                positive='1',mode = "prec_recall")

#\
#We see improved values of precision to 0.79, recall to 0.70
#and F1 score to 0.74

#This clearly is our most balanced and best model yet.



# Iteration 5 - Computing WOE (weight of evidence) and IV (information value) to improve prediction accuracy

### Computing IV

library(Information)
library(gridExtra)
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
data <- data[data$ejection_fraction <70,]
data <- data[data$creatinine_phosphokinase <7000,]
data$anaemia <- factor(data$anaemia)
data$diabetes <- factor(data$diabetes)
data$high_blood_pressure <- factor(data$high_blood_pressure)
data$sex <- factor(data$sex)
data$smoking <- factor(data$smoking)
# this package needs the dependent variable in numeric format
# hence we reload data here
IV <- create_infotables(data=data, y="DEATH_EVENT",
                        bins=10, parallel=FALSE)
IV_Value = data.frame(IV$Summary)
IV$Summary


#Let's analyze IV first -
#Our IV values are significant for time, ejection_fraction,
#serum_creatinine, age, serum_sodium, creatinine_phosphokinase
#and platelets (>0.1). After platelets, IV values are below
#0.02 so we do not need to use these variables. 

### Plotting WOE

library(woe)
# plot woe
plot_infotables(IV, IV$Summary$Variable[1:12], same_scale=FALSE)



### Replacing WOE

library(fuzzyjoin)
woe_replace <- function(df_orig, IV) {
  df <- cbind(df_orig)
  df_clmtyp <- data.frame(clmtyp = sapply(df, class))
  df_col_typ <-
    data.frame(clmnm = colnames(df), clmtyp = df_clmtyp$clmtyp)
  for (rownm in 1:nrow(df_col_typ)) {
    colmn_nm <- toString(df_col_typ[rownm, "clmnm"])    
    if(colmn_nm %in% names(IV$Tables)){
    column_woe_df <- cbind(data.frame(IV$Tables[[toString(df_col_typ[rownm, "clmnm"])]]))
    if (df_col_typ[rownm, "clmtyp"] == "factor" | df_col_typ[rownm, "clmtyp"] == "character") {
      df <-
        dplyr::inner_join(
          df,
          column_woe_df[,c(colmn_nm,"WOE")],
          by = colmn_nm,
          type = "inner",
          match = "all"
        )
      df[colmn_nm]<-NULL
      colnames(df)[colnames(df)=="WOE"]<-colmn_nm
    } else if (df_col_typ[rownm, "clmtyp"] == "numeric" | df_col_typ[rownm, "clmtyp"] == "integer") {
      column_woe_df$lv<-as.numeric(str_sub(
        column_woe_df[,colmn_nm],
        regexpr("\\[", column_woe_df[,colmn_nm]) + 1,
        regexpr(",", column_woe_df[,colmn_nm]) - 1
      ))
      column_woe_df$uv<-as.numeric(str_sub(
        column_woe_df[,colmn_nm],
        regexpr(",", column_woe_df[,colmn_nm]) + 1,
        regexpr("\\]", column_woe_df[,colmn_nm]) - 1
      ))
      column_woe_df[colmn_nm]<-NULL      
      column_woe_df<-column_woe_df[,c("lv","uv","WOE")]      
      colnames(df)[colnames(df)==colmn_nm]<-"WOE_temp2381111111111111697"      
      df <-
        fuzzy_inner_join(
          df,
          column_woe_df[,c("lv","uv","WOE")],
          by = c("WOE_temp2381111111111111697"="lv","WOE_temp2381111111111111697"="uv"),
          match_fun=list(`>=`,`<=`) 
        )      
      df["WOE_temp2381111111111111697"]<-NULL      
      df["lv"]<-NULL      
      df["uv"]<-NULL      
      colnames(df)[colnames(df)=="WOE"]<-colmn_nm      
    }}
  }
  return(df)
}
df_woe <- woe_replace(data, IV)
str(df_woe)


#Let's now use the new dataframe for prediction.

### Splitting into train and test - 70%, 30% split

set.seed(123)
trainIndex <- createDataPartition(df_woe$DEATH_EVENT, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train_data<-df_woe[trainIndex,]
test_data<-df_woe[-trainIndex,]

# Model 5
mylogit_5 <- glm(DEATH_EVENT ~ age+anaemia+
                   creatinine_phosphokinase+
                   diabetes+ejection_fraction+high_blood_pressure+platelets+
                   serum_creatinine+serum_sodium+sex+smoking+time ,
                 data = train_data, family = "binomial")
summary(mylogit_5)
predicted <- predict(mylogit_5, test_data, type="response")


#\
#We see age, time, ejection_fraction and serum_sodium,
#creatinine_phosphpkinase, serum_creatinine as significant variables.


# Model 5
mylogit_5 <- glm(DEATH_EVENT ~ age+
                   ejection_fraction+
                   serum_sodium+time+creatinine_phosphokinase+
                   serum_creatinine, data = train_data, family = "binomial")
summary(mylogit_5)
predicted <- predict(mylogit_5, test_data, type="response")


#
#We note our lowest AIC yet of 97.77

### Deciding on optimal cutoff

optCutOff <- optimalCutoff(test_data$DEATH_EVENT, predicted)[1] 
optCutOff


#\
#This tells us that we can use this prob cutoff to classify an 
#observation as 0 or 1. If the prob-value is below this threshold we
#can classify it as survival else death event.

### Mis-classification error

misClassError(test_data$DEATH_EVENT, predicted, threshold = optCutOff)


#\
#We note a misclassification error on test set of 14.7%

### ROC curve

plotROC(test_data$DEATH_EVENT, predicted)


#\
#We see an AUC of 0.889 which is decent.

### KS plot

ks_stat(test_data$DEATH_EVENT, predicted)
ks_plot(test_data$DEATH_EVENT, predicted)



### Concordance check

Concordance(test_data$DEATH_EVENT, predicted)

#\
#Usually concordance is in-line with AUC and we see that 88.8% pairs
#are concordant (the model calculated prob scores of 1s being greater than
#                model calculated prob scores of 0s)

### Using optimal cutoff to determine accuracy measures

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_5,test_data,
                                 type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), mode = "prec_recall")


#\
#We see that on test set our accuracy is 85.2% 

### Using optimal cutoff to determine accuracy measures with positive class as 1

threshold=optCutOff
predicted_values<-ifelse(predict(mylogit_5, test_data,
                                 type="response")>threshold,1,0)
actual_values<-test_data$DEATH_EVENT
confusionMatrix(data = as.factor(predicted_values), 
                reference = as.factor(actual_values), 
                positive='1',mode = "prec_recall")

#\
#We however note a key difference in this accuracy.
#Our recall is 0.87 while the precision is 0.75
#with F1-score of 0.81

#We see how computing and recoding variables to WOE
#improved our model accuracy even further. We also see our
#highest recall yet of 0.87 which is great for the purpose
#of predicting death events more rigorously.



# Summarizing all model results in a table

#| Model | Variables | AIC | AUC | Accuracy | Precision | Recall | F1-Score | Comments |
#  |------ | ----------| ----|-----|----------|-----------|--------|----------|----------|
#  | Model 1 | All | 241.95 | 0.894 | 0.854 | 0.79 | 0.72 | 0.75 | No test set (overfitting) |
#  | Model 2 | Age, ejection_fraction, serum_creatinine, time  | 232.4 | 0.888 | 0.85 | 0.84 | 0.64 | 0.73 | No test set (overfitting) |
#  | Model 3 | All  | 173.4 | 0.868 | 0.827 | 0.80 | 0.59 | 0.68 | Test set results- First real model |
#  | Model 4 | Age, ejection_fraction, serum_creatinine, time, sex  | 164.7 | 0.87 | 0.85 | 0.79 | 0.70 | 0.74 | Stepwise Forward selection |
#  | Model 5 | Age, ejection_fraction, serum_creatinine, time, serum_sodium, creatinine_phosphokinase  | 97.7 | 0.89 | 0.852 | 0.75 | 0.88 | 0.81 | WoE dataset |
  
  ### We note that in model 4 stepwise method performed well and gave us better model results than model 3
  ### However, we finally have a model (Model 5) which we can use as a best model outcome from our iterations which came from computing woe and iv values for each of our variables in the dataset.
  
  ### This concludes our approach to Logistic regression in our dataset
  
  
  