# Assignment 7 - Linear regression

#This document checks the assumptios of Linear regression on the 
#Heart Failure Prediction dataset. We know we have a classification
#problem at hand and modeling with linear regression would not serve
#our purpose however we perform a theoretical exercise of
#checking assumptions, multi-collinearity as well as some other
#interesting results.

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

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)

# Fitting a linear regression model
mod <-  lm( DEATH_EVENT ~ age+anaemia+creatinine_phosphokinase+
              diabetes+ejection_fraction+high_blood_pressure+platelets+
              serum_creatinine+serum_sodium+sex+smoking+time, data) 
summary(mod)

#\
#We cannot really interpret the results here.

# Testing assumptions of linear regression

#Recall the assumptions of linear regression as \
#1. Linear relationship \
#2. Normality of residuals \
#3. Homoscedasticity \
#4. No auto-correlation \
#5. No or little multicollinearity \
#6. Normality of the dependent variable.

# Linear relationship

#The linearity assumption can be checked by inspecting 
#the Residuals vs Fitted plot.

plot(mod,which=1)

#In this plot, we clearly see a pattern for residuals
#We see them decreasing below 0.5 (fitted values) and increasing
#above 0.5 (fitted values). This indicates we don't have linear
#relationship between our dependent and independent variables.

# Normality of residuals

#The QQ plot of residuals can be used to check the normality 
#assumption. The normal probability plot of residuals 
#should approximately follow a straight line.

plot(mod,which=2)

#Surprisingly we see points falling along reference line 
#however we also see some falling outside so we dig deeper

# Shapiro-Wilk Normality Test

resid <- studres(mod) 
shapiro.test(resid)

#From the p-value = 0.002927 < 0.05, 
#we can see that the residuals are not normally distributed

# High leverage points
plot(mod, which=5)

#Leverage statistic is defined as - \
#$\hat{L}= \dfrac{2(p+1)}{n}$ 
#where $p$ is number of predictors and $n$ is number of observations \
#So for us $\hat{L} = 0.0869$ \

# Cook's distance

#Cook's distance
plot(mod, 4)


#A rule of thumb is that an observation has high influence 
#if Cook's distance exceeds $\dfrac{4}{(n - p - 1)}$

#So from the above plots we see cook's plot shows
#10, 132, 218 as values of extreme nature and we see
#no influential points. All points seem to fall under 
#Cook's distance lines (missing dashed lines in residuals
#vs leverage plot indicates the same)

# Homoskedasticity

plot(mod,which=3)

#The spread-location or scale-location plot helps us assess 
#homoskedasticity. We see clearly what is not a horizontal line 
#indicating residuals are not spread equally around the range of
#fitted values.

###  ncvTest() For Homoscedasticity

ncvTest(mod)

#\
#We see a p-value < .05, indicating that our data is 
#not homoscedastic.

### Breusch-Pagan Test For Homoscedasticity

bptest(mod)

#We see a p-value < .05, indicating that our data is 
#not homoscedastic.

#  Autocorrelation Assumption
#The Durbin Watson examines whether the errors are autocorrelated 
#with themselves. The null states that they are not 
#autocorrelated.

durbinWatsonTest(mod)

#We see that p-value < 0.05, 
#so the errors are autocorrelated.

# Normality of y

#We can check the normality of the dependent variable
#by plotting a histogram.

hist(data$DEATH_EVENT)

#Our histogram doesn't indicate normality of dependent variable.

# Assessing multicollinearity
### VIF method

vif(mod)

#\
#We note that VIF is below 2 for all independent variables 
#indicating there is no multi-collinearity problem in our data.


### Note: We see our dependent variable isn't ideal for Linear regression 
#and requires classification techniques to model the same. 
#We do test for assumptions however above and see most of them 
#failing except for little multi-collinearity in our data.

### This concludes our approach to Linear regression in our dataset


