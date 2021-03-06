#### Aman ####

##############  Assignment 3 - Data Cleaning, EDA, Tests #####################
##############################################################################
##############################################################################

# clear environment
rm(list = ls())

# defining libraries

library(ggplot2)
library(dplyr)
library(PerformanceAnalytics)
library(data.table)
library(sqldf)
library(nortest)
library(tidyverse)
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

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')

# structure of data
str(data)
glimpse(data)

# summary of data
summary(data)

# Observations
# 0. 299 observations for 13 variables
# 1. Age is between 40 and 95 so not much outliers by intuition
# 2. Death_event should be converted to factor variable as they take only 2 values
# 3. Creatinine phosphokinase, platelets clearly has an outlier from max value which
# we will confirm later by univariate analysis

# Let's look for missing/ NAs
data2 <- na.omit(data)

# data2 has same rows as data so there are no missing values in data

# Correlation plot
M<-cor(data)
head(round(M,2))
corrplot(M, method="color")

# We see that age, anameia, creatinine_phosphokinase, 
# high_blood_pressure, serum_creatinine have +ve correlation with death_event

# We see that ejection_fraction, platelets, serum_sodium, 
# and time have -ve correlation with death_event

# But we will need deeper analysis to confirm these relationships

# Converting to factor (dependent variable)
data$DEATH_EVENT <- factor(data$DEATH_EVENT)

# Let's check how many zeros are in dataset
colSums(data==0)

# Let's check their proportion to dataset as well 
round(colSums(data==0)/nrow(data)*100,2)

# Smoking, High BP, Diabetes, Anaemia are over 50% while sex is below 35%
# Also event rate of survival is ~67.9%

# Let's classify variables into -
# 1. Categorical -> Anaemia, Diabetes, High_blood_pressure, Sex, Smoking, Death_event
# 2. Numeric -> Age, Creatinine_phosphokinase, Ejection_fraction, Platelets, 
# serum_creatinine, serum_sodium, time

# We also see that 
# Sex - Gender of patient Male = 1, Female =0
# Diabetes - 0 = No, 1 = Yes
# Anaemia - 0 = No, 1 = Yes
# High_blood_pressure - 0 = No, 1 = Yes
# Smoking - 0 = No, 1 = Yes
# DEATH_EVENT - 0 = No, 1 = Yes

### Analysis ### 

# We note the scale of few variables like creatinine_phosphokinase,platelets,
# ejection_fraction and time. We can normalize the same before modeling
# but for now we will keep them as-is for the EDA

# Since there are no missing values, let's look at outliers

# Outlier Analysis
boxplot(data$age,
        main = "Age Box Plot",
        xlab = "Age",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# No observed outlier in age

boxplot(data$creatinine_phosphokinase,
        main = "creatinine_phosphokinase Box Plot",
        xlab = "creatinine_phosphokinase",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)

# We notice some max outliers in creatinine_phosphokinase with data above median 
# more dispersed

boxplot(data$ejection_fraction,
        main = "ejection_fraction Box Plot",
        xlab = "ejection_fraction",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# We notice 2 data points as outliers in ejection_fraction

boxplot(data$platelets,
        main = "platelets Box Plot",
        xlab = "platelets",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# We notice outliers on both spectrum (high and low) in platelets

boxplot(data$serum_creatinine,
        main = "serum_creatinine Box Plot",
        xlab = "serum_creatinine",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# We notice some outliers in serum_creatinine on higher end (similar to 
# creatinine_phosphokinase)
# However these are in possible ranges medically

boxplot(data$serum_sodium ,
        main = "serum_sodium  Box Plot",
        xlab = "serum_sodium ",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# We notice some outliers in serum_sodium on lower end 

boxplot(data$time ,
        main = "time  Box Plot",
        xlab = "time ",
        ylab = "Spread",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
# We notice no outliers in time (follow up period) however data above median is 
# more dispersed
# Note: While some of these are clear outliers, others must be checked with 
# possible medical range

# Data Cleaning - Let's remove these outliers
data <- data[data$ejection_fraction <70,]
data <- data[data$creatinine_phosphokinase <7000,]

# The new data now has only 295 observations - 4 observations were removed

#### Univariate analysis ###

# Age

# Let's check Age distribution and see it with death event
a<-ggplot(data,aes(x = age))+geom_histogram(binwidth = 5, color = "black", 
                                            fill = "dark blue",alpha = 0.5)+
  labs(title = "Age Distribution", 
      caption = "Age Distribution")+
  theme(plot.caption = element_text(hjust = 0.5,face = "italic"))+
  scale_x_continuous(breaks = seq(40,100,10))

b<-ggplot(data,aes(x = age, fill = DEATH_EVENT))+geom_histogram(binwidth = 5, 
                        position = "identity",
alpha = 0.5,color = "black")+scale_fill_manual(values = c("#999999", "#E69F00"))+
  labs(caption = "Age Distribution with Death Event")+
  theme(plot.caption = element_text(hjust = 0.5,face = "italic"))+
  scale_x_continuous(breaks = seq(40,100,10))

gridExtra::grid.arrange(a,b)

# We see that as age increases, chances of death event go up as well

# Let's create age ranges
data$age_tr[data$age < 50 & data$age >= 40]="40-50"
data$age_tr[data$age < 60 & data$age >= 50]="50-60" 
data$age_tr[data$age < 70 & data$age >= 60]="60-70"
data$age_tr[data$age < 80 & data$age >= 70]="70-80"
data$age_tr[data$age < 90 & data$age >= 80]="80-90"
data$age_tr[data$age < 100 & data$age >= 90]="90-100"

table(data$DEATH_EVENT, data$age_tr)

# Numerically, we can confirm the same observation (Higher death rate in 
# higher ages)

# Creatinine_phosphokinase
# density plot of Creatinine_phosphokinase
ggplot(data,aes(x = creatinine_phosphokinase))+geom_density(fill = "dark blue",
                                          alpha = 0.5)+
  labs(title = "Distribution of creatinine phosphokinase", caption = 
    "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$creatinine_phosphokinase_tr <- cut(data$creatinine_phosphokinase, 10)
table(data$DEATH_EVENT, data$creatinine_phosphokinase_tr)

# Numerically, we can see that for creatinine levels above 4000, death event 
# seems to be higher

aggregate(data[, c('creatinine_phosphokinase')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average creatinine levels are lower in case of 
# death event

# Ejection_fraction
ggplot(data,aes(x = ejection_fraction))+geom_density(fill = "dark blue", 
                                                     alpha = 0.5)+
  labs(title = "Distribution of ejection_fraction", 
       caption = "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$ejection_fraction_tr <- cut(data$ejection_fraction, 10)
table(data$DEATH_EVENT, data$ejection_fraction_tr)

# Numerically, we can see that ejection fraction is low in case of death event

aggregate(data[, c('ejection_fraction')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average ejection fraction is also low in 
# case of death event


# platelets 
ggplot(data,aes(x = platelets ))+geom_density(fill = "dark blue", alpha = 0.5)+
  labs(title = "Distribution of platelets ", caption = "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$platelets_tr  <- cut(data$platelets , 10)
table(data$DEATH_EVENT, data$platelets_tr)

# Numerically, we can see that platelets are low in case of death event

aggregate(data[, c('platelets')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average platelets are marginally lower in 
# case of death event

# serum_creatinine 
ggplot(data,aes(x = serum_creatinine ))+geom_density(fill =
                                          "dark blue", alpha = 0.5)+
  labs(title = "Distribution of serum_creatinine ",
       caption = "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$serum_creatinine_tr  <- cut(data$serum_creatinine , 10)
table(data$DEATH_EVENT, data$serum_creatinine_tr)

# Numerically, we can see that death event is high when serum_creatinine
# levels are above 1.39 and very high above 2.28

aggregate(data[, c('serum_creatinine')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average serum_creatinine is high in case of death event


# serum_sodium 
ggplot(data,aes(x = serum_sodium ))+geom_density(fill = "dark blue",
                                                alpha = 0.5)+
  labs(title = "Distribution of serum_sodium ", 
      caption = "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$serum_sodium_tr  <- cut(data$serum_sodium , 10)
table(data$DEATH_EVENT, data$serum_sodium_tr)

# Numerically, we can see that serum_sodium are low in case of death event

aggregate(data[, c('serum_sodium')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average serum sodium are marginally lower 
# in case of death event


# time 
ggplot(data,aes(x = time ))+geom_density(fill = "dark blue", alpha = 0.5)+
  labs(title = "Distribution of time ", caption = "Density distribution")+
  theme(plot.caption = element_text(hjust = 0.5, face = "italic"))

#let's create 10 splits of this variable
data$time_tr  <- cut(data$time , 10)
table(data$DEATH_EVENT, data$time_tr)

# Numerically, we can see that follow up period was small in case of death event

aggregate(data[, c('time')], list(data$DEATH_EVENT), mean)

# Numerically, we can see that average follow up period is low in case of death event
# This simply may illustrate that once deemed healthy,the patients may have stopped 
# following up whereas diseased patients would undergo more checkups

# Let's do EDA for categorical variables now

# Anaemia, Diabetes, High_blood_pressure, Sex, Smoking

a <- ggplot(data, aes(x = DEATH_EVENT, fill = factor(anaemia)))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Death Event:No","Death Event:Yes"))+
  scale_fill_manual(values = c("#999999", "#E69F00"), name = "Anaemia",
  labels = c("No","Yes"))+labs(subtitle = "Anaemia")

b<-ggplot(data, aes(x = DEATH_EVENT, fill = factor(diabetes)))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Death Event:No","Death Event:Yes"))+
  scale_fill_manual(values = c("#999999", "#E69F00"), name = "Diabetes",
                    labels = c("No","Yes"))+labs(subtitle = "Diabetes")

c<-ggplot(data, aes(x = DEATH_EVENT, fill = factor(high_blood_pressure)))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Death Event:No","Death Event:Yes"))+
  scale_fill_manual(values = c("#999999", "#E69F00"), name = "High BP",
                    labels = c("No","Yes"))+labs(subtitle = "High BP")

d<-ggplot(data, aes(x = DEATH_EVENT, fill = factor(sex)))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Death Event:No","Death Event:Yes"))+
  scale_fill_manual(values = c("#999999", "#E69F00"), name = "Sex",
                    labels = c("Female","Male"))+labs(subtitle = "Sex")

e<-ggplot(data, aes(x = DEATH_EVENT, fill = factor(smoking)))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Death Event:No","Death Event:Yes"))+
  scale_fill_manual(values = c("#999999", "#E69F00"), name = "Smoking",
                    labels = c("No","Yes"))+labs(subtitle = "Smoking")

grid.arrange(a,b,c,d,e)

# We can see that Anaemia and High BP has significant difference for death event 
# whereas others not so much

# Normality tests

# univariate normality
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
cm <- colMeans(data)
S <- cov(data)
d <- apply(data, MARGIN = 1, function(data)t(data - cm) %*% solve(S) %*% (data - cm))

# Age
qqnorm(data[,"age"], main = "age")
qqline(data[,"age"])

# Age looks normally distributed

# Creatinine_phosphokinase
qqnorm(data[,"creatinine_phosphokinase"], main = "Creatinine_phosphokinase")
qqline(data[,"creatinine_phosphokinase"])

# Creatinine_phosphokinase doesn't looks normal but skewed

# ejection_fraction
qqnorm(data[,"ejection_fraction"], main = "ejection_fraction")
qqline(data[,"ejection_fraction"])

# ejection_fraction doeesn't look normal as well

# serum_creatinine
qqnorm(data[,"serum_creatinine"], main = "serum_creatinine")
qqline(data[,"serum_creatinine"])

# serum_creatinine doesn't look normal but skewed

# serum_sodium
qqnorm(data[,"serum_sodium"], main = "serum_sodium")
qqline(data[,"serum_sodium"])

# serum_sodium looks normal but is slightly skewed on lower end

plot(qchisq((1:nrow(data) - 1/2) / nrow(data), df = 13), sort(d),
     xlab = expression(paste(chi[13]^2, " Quantile")),
     ylab = "Ordered distances")
abline(a = 0, b = 1)

# While plotting for multivariate normality, we see that
# data is non normal and has some skewness towards positive side

# t-tests for death events vs not for each variable

# age
with(data,t.test(age[DEATH_EVENT=="1"],age[DEATH_EVENT=="0"],var.equal=TRUE))
# anaemia
with(data,t.test(anaemia[DEATH_EVENT=="1"],anaemia[DEATH_EVENT=="0"],var.equal=TRUE))
# creatinine_phosphokinase
with(data,t.test(creatinine_phosphokinase[DEATH_EVENT=="1"],
                 creatinine_phosphokinase[DEATH_EVENT=="0"],var.equal=TRUE))
# diabetes
with(data,t.test(diabetes[DEATH_EVENT=="1"],
                 diabetes[DEATH_EVENT=="0"],var.equal=TRUE))
# ejection_fraction
with(data,t.test(ejection_fraction[DEATH_EVENT=="1"],
                 ejection_fraction[DEATH_EVENT=="0"],var.equal=TRUE))
# high_blood_pressure
with(data,t.test(high_blood_pressure[DEATH_EVENT=="1"],
                 high_blood_pressure[DEATH_EVENT=="0"],var.equal=TRUE))
# platelets
with(data,t.test(platelets[DEATH_EVENT=="1"],
                 platelets[DEATH_EVENT=="0"],var.equal=TRUE))
# serum_creatinine
with(data,t.test(serum_creatinine[DEATH_EVENT=="1"],
                 serum_creatinine[DEATH_EVENT=="0"],var.equal=TRUE))
# serum_sodium
with(data,t.test(serum_sodium[DEATH_EVENT=="1"],
                 serum_sodium[DEATH_EVENT=="0"],var.equal=TRUE))
# sex
with(data,t.test(sex[DEATH_EVENT=="1"],
                 sex[DEATH_EVENT=="0"],var.equal=TRUE))
# smoking
with(data,t.test(smoking[DEATH_EVENT=="1"],
                 smoking[DEATH_EVENT=="0"],var.equal=TRUE))
# time
with(data,t.test(time[DEATH_EVENT=="1"],
                 time[DEATH_EVENT=="0"],var.equal=TRUE))

# p-value is below 0.05 for - 
# 1. age
# 2. serum_sodium
# 3. serum_creatinine
# 4. ejection_fraction
# 5. time
# so we may conclude that death event does differ by these variables 

# Hotelling's T2 test. Comparing multivariate means between death events and non-death event
t2testdata <- hotelling.test(age + anaemia + creatinine_phosphokinase +
              diabetes + ejection_fraction + high_blood_pressure+
                platelets+serum_creatinine + serum_sodium +
                sex+smoking+time
                ~ DEATH_EVENT, data)

cat("T2 statistic =",t2testdata$stat[[1]],"\n")
print(t2testdata)

# The difference in means in the two groups taken together is
# significant as well

# Homoskedasticity check

data$DEATH_EVENT <- factor(data$DEATH_EVENT)
leveneTest(age ~ DEATH_EVENT, data=data)
leveneTest(anaemia ~ DEATH_EVENT, data=data)
leveneTest(creatinine_phosphokinase ~ DEATH_EVENT, data=data)
leveneTest(diabetes ~ DEATH_EVENT, data=data)
leveneTest(ejection_fraction ~ DEATH_EVENT, data=data)
leveneTest(high_blood_pressure ~ DEATH_EVENT, data=data)
leveneTest(platelets ~ DEATH_EVENT, data=data)
leveneTest(serum_creatinine ~ DEATH_EVENT, data=data)
leveneTest(serum_sodium ~ DEATH_EVENT, data=data)
leveneTest(sex ~ DEATH_EVENT, data=data)
leveneTest(smoking ~ DEATH_EVENT, data=data)
leveneTest(time ~ DEATH_EVENT, data=data)

# p-value is below 0.05 for - 
# 1. age
# 2. serum_creatinine
# 3. ejection_fraction
# 4. time
# 5. serum_sodium
# so we may conclude that variance between the two groups
# differ in them

# One-way ANOVA tests: comparing univariate means
aov_age <- aov(age ~ DEATH_EVENT, data)
summary(aov_age)
aov_anaemia <- aov(anaemia ~ DEATH_EVENT, data)
summary(aov_anaemia)
aov_creatinine_phosphokinase <- aov(creatinine_phosphokinase ~ DEATH_EVENT, data)
summary(aov_creatinine_phosphokinase)
aov_diabetes <- aov(diabetes ~ DEATH_EVENT, data)
summary(aov_diabetes)
aov_ejection_fraction <- aov(ejection_fraction ~ DEATH_EVENT, data)
summary(aov_ejection_fraction)
aov_high_blood_pressure <- aov(high_blood_pressure ~ DEATH_EVENT, data)
summary(aov_high_blood_pressure)
aov_platelets <- aov(platelets ~ DEATH_EVENT, data)
summary(aov_platelets)
aov_serum_creatinine <- aov(serum_creatinine ~ DEATH_EVENT, data)
summary(aov_serum_creatinine)
aov_serum_sodium <- aov(serum_sodium ~ DEATH_EVENT, data)
summary(aov_serum_sodium)
aov_sex <- aov(sex ~ DEATH_EVENT, data)
summary(aov_sex)
aov_smoking <- aov(smoking ~ DEATH_EVENT, data)
summary(aov_smoking)
aov_time <- aov(time ~ DEATH_EVENT, data)
summary(aov_time)


# p-value is below 0.05 for - 
# 1. age
# 2. serum_creatinine
# 3. ejection_fraction
# 4. time
# 5. serum_sodium
# so we may conclude that means between the two groups
# differ in them

# Comparing multivariate means (One-way MANOVA)
mnv <- manova(as.matrix(data[,-13])~ DEATH_EVENT, data)
summary(mnv)

# We observe from MANOVA that estimated effects may be unbalanced
# indicating that mean between groups may be different 

# Let us also look at Multicollinearity check
# Earlier we saw the correlation plot
# We will check VIF for this purpose
# In classification, although linear regression isn't to be used
# For VIF, a rudimentary model lets us know the association
# between continuous and categorical variables

data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
mod <-  lm( DEATH_EVENT ~ age+anaemia+creatinine_phosphokinase+
    diabetes+ejection_fraction+high_blood_pressure+platelets+
    serum_creatinine+serum_sodium+sex+smoking+time, data) 
summary(mod)

vif(mod)

# We see that most VIF values are below 1.5
# This incdicates absence of multi-collinearity in our data


############## This concludes our initial EDA for the data ###########
######################################################################
######################################################################

