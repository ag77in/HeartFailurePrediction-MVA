# Assignment 9 - Discriminant Analysis

#This document performs Discriminant Analysis on the 
#Heart Failure Prediction dataset.

#We note there are multiple types of Discriminant Analysis
#and we will explore all of them in this document.

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
library(mda)
library(klaR)
library(ROCR)



# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)


# Data Cleaning - Let's remove the outliers

data <- data[data$ejection_fraction <70,]
data <- data[data$creatinine_phosphokinase <7000,]
str(data)


#We remove the 4 outliers before proceeding to modeling exercise.

# Split into train (70%), test (30%) and normalize data


set.seed(123)
training.samples <- data$DEATH_EVENT %>%
  createDataPartition(p = 0.7, list = FALSE)
train.data <- data[training.samples, ]
test.data <- data[-training.samples, ]
# Estimate preprocessing parameters
preproc.param <- train.data %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)


#We see that our train set has 207 observations, while our test
#set has 88 observations.



# Linear Discriminant Analysis

#LDA like PCA finds maximum separation only this time between classes instead of
#independent variables. The directions are termed as linear discriminants and
#are the linear combination of independent variables.

### Assumptions of LDA

#1. Independent variables are normally distributed
#2. No outliers and scaling

#We saw in EDA exercise when we performed univariate checks that
#not all our variables are normal with exception of
#age, serum_sodium. Most other numeric variables are +vely skewed.
#We can try transformations but first let's perform LDA and analyze results.
#We already did outlier removal and scaling.

### Fitting a model


# Fit the model
set.seed(123)
model <- lda(DEATH_EVENT~., data = train.transformed)
model


#\
#The outcome is easy enough to interpret. Group means depict the 
#centre of gravity for each variable. We get only one LD1 as 
#our dependent variable is binary.

### Let's plot the model

plot(model)


#\
#The first plot is for survival events and the latter
#plot is for death events. We see above some overlap above -1
#till 1.2.

### Let's predict


# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class==test.transformed$DEATH_EVENT)


#This base model is 77.2% accurate. 

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions$class),
                reference = as.factor(test.transformed$DEATH_EVENT),
                positive='1.54333408509054', mode = "prec_recall")


#\
#We see a precision of 0.73 and a recall of 0.59 and an
#F1-score of 0.65. 

### Let's compute ROC/AUC


posteriors <- as.data.frame(predictions$posterior)
pred <- prediction(posteriors[,2], test.transformed$DEATH_EVENT)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train.old <- performance(pred, measure = "auc")
auc.train.old <- auc.train.old@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train.old[[1]],3), sep = ""))


#\
#We get an AUC of 0.843. However, we do note that False positive
#rates also increase as TPR increases which means we would be 
#telling people with heart failiure risk that they are fine.



# Quadratic Discriminant Analysis

#QDA doesn't assume equality of variance/covariance. Let's experiment
#on our data


# Fit the model
set.seed(123)
model <- qda(DEATH_EVENT~., data = train.transformed)
model



# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class == test.transformed$DEATH_EVENT)


#This base model is 72.7% accurate. 

#In general, QDA works better than LDA for large data.

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions$class),
                reference = as.factor(test.transformed$DEATH_EVENT), 
                positive='1.54333408509054', mode = "prec_recall")


#\
#We see a precision of 0.7 and a recall of 0.43 and an
#F1-score of 0.53. 

### Let's compute ROC/AUC


posteriors <- as.data.frame(predictions$posterior)
pred <- prediction(posteriors[,2], test.transformed$DEATH_EVENT)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#\
#We get an AUC of 0.81 however we can get a higher TPR for the same
#increase in FPR from before



# Mixed Discriminant Analysis

#While LDA assumes, each class comes from normal distribution, while
#in MDA each class is assumed to be a mixture of subclasses.


# Fit the model
set.seed(123)
model <- mda(DEATH_EVENT~., data = train.transformed)
model

#\
#The MDA gives percent between group variance for 5 dimensions.
#It is easier to think of this as LDA1-5 performed simultaneously.


# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions == test.transformed$DEATH_EVENT)


#This base model is 78.4% accurate which is better than both 
#LDA and QDA.

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions),
                reference = as.factor(test.transformed$DEATH_EVENT), 
                positive='1.54333408509054', mode = "prec_recall")

#\
#We see a precision score of 0.78, recall of 0.56
#and F1-score of 0.65



# Flexible Discriminant Analysis

#FDA is just an extension of LDA for modeling nonlinearities.


# Fit the model
set.seed(123)
model <- fda(DEATH_EVENT~., data = train.transformed)
model

#\

#FDA produces only one dimension as expected (binary case)


# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions == test.transformed$DEATH_EVENT)


#This base model is 77.2% accurate which is exactly what we got from LDA
#as well.

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions),
                reference = as.factor(test.transformed$DEATH_EVENT), 
                positive='1.54333408509054', mode = "prec_recall")

#\
#We see a precision score of 0.73, recall of 0.59
#and F1-score of 0.65



# Regularized Discriminant Analysis

#RDA builds classification rules by regularizing group covariance matrices.


# Fit the model
set.seed(123)
model <-rda(DEATH_EVENT~., data = train.transformed)
model



# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class == test.transformed$DEATH_EVENT)


#This base model is 78.4% accurate which is exactly what we got from MDA
#as well. 

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions$class),
                reference = as.factor(test.transformed$DEATH_EVENT),
                positive='1.54333408509054', mode = "prec_recall")


#\
#We see a precision of 0.74 and a recall of 0.62 and an
#F1-score of 0.67. 

### We found that for a base model MDA, and RDA gave us same accuracies on the test set however RDA gave better F1-score whereas LDA was second best with slightly lower accuracies. However, QDA which assumes different covariance matrices for each class gave us slightly worse results.

#We now try an iteration with PCA set being used to perform LDA.



# LDA with PCA combined

### conduct PCA on training dataset

pca <- prcomp(train.transformed[,1:12], retx=TRUE, 
              center=TRUE, scale=TRUE)
expl.var <- round(pca$sdev^2/sum(pca$sdev^2)*100) 
# percent explained variance
expl.var

#
#The explained variance in components is same as before \

# prediction of PCs for validation dataset

pred <- predict(pca, newdata=test.transformed[,1:12])
head(pred,5)

### Let's create the same sets but with PCA variables added


new_data.train <- cbind(train.transformed,pca$x)
new_data.test <- cbind(test.transformed,pred)


### Let's compute for 7 components -


model <- lda(DEATH_EVENT ~ PC1 + PC2 + PC3 + PC4 
             + PC5 + PC6 + PC7, data = new_data.train)
model
# Make predictions
predictions <- model %>% predict(new_data.test)
# Model accuracy
mean(predictions$class == new_data.test$DEATH_EVENT)


#\
#We see that with PCA dimensions reduced to 7 from 12, we're able to increase
#our accuracy to 81.8%

### Let's compute accuracies

confusionMatrix(data = as.factor(predictions$class),
                reference = as.factor(test.transformed$DEATH_EVENT),
                positive='1.54333408509054', mode = "prec_recall")


#\
#We see a precision of 0.78 and a recall of 0.68 and an
#F1-score of 0.73. We see this is even better than before. 

### Let's compute ROC/AUC


posteriors <- as.data.frame(predictions$posterior)
pred <- prediction(posteriors[,2], new_data.test$DEATH_EVENT)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train.new <- performance(pred, measure = "auc")
auc.train.new <- auc.train.new@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train.new[[1]],3), sep = ""))

#\
#Our new AUC is 0.847 which is good and marginally better than non-PCA set (0.843).



# Trying LDA on WOE dataset as before

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


### lda on data

model <- lda(DEATH_EVENT ~ ., data = train_data)
model
# Make predictions
predictions <- model %>% predict(test_data)
# Model accuracy
mean(predictions$class == test_data$DEATH_EVENT)


#\
#This model gives us an accuracy of 84.0%


confusionMatrix(data = as.factor(predictions$class),
        reference = as.factor(test_data$DEATH_EVENT), 
        positive='1', mode = "prec_recall")


#\
#We see a precision of 0.84 and a recall of 0.68 and an
#F1-score of 0.75. We see this is again marginally better than
#our previous iteration.

### Let's compute ROC/AUC


posteriors <- as.data.frame(predictions$posterior)
pred <- prediction(posteriors[,2], test_data$DEATH_EVENT)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train.new <- performance(pred, measure = "auc")
auc.train.new <- auc.train.new@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train.new[[1]],3), sep = ""))

#\
#We obtain our highest AUC yet of 0.89



# Summarizing all model results in a table

#| Model | Type  |  Accuracy | Precision | Recall | F1-Score | Comments |
#  |------ | ----------| -----|----------|-----------|--------|----------|----------|
#  | Model 1 | LDA | 0.772 | 0.73 | 0.59 | 0.65 | Linear |
#  | Model 2 | QDA | 0.727 | 0.70 | 0.43 | 0.53 | Quadratic |
#  | Model 3 | MDA | 0.784 | 0.78 | 0.56 | 0.65 | Mixed |
#  | Model 4 | FDA | 0.772 | 0.73 | 0.59 | 0.65 | Flexible |
#  | Model 5 | RDA | 0.784 | 0.74 | 0.62 | 0.67 | Regularized |
#  | Model 6 | LDA with PCA | 0.818 | 0.78 | 0.68 | 0.73 | PCA of 7 components |
#  | Model 7 | LDA on WoE | 0.841 | 0.84 | 0.69 | 0.76 | WoE dataset |
  
  ### Once again, we see that WoE dataset has best accuracies with LDA and even LDA with PCA works well. Among the base models, we saw MDA and RDA outperform LDA and FDA with QDA being the wrong model for this data.
  
  ### This concludes our approach to Discriminant Analysis in our dataset
  
  
  