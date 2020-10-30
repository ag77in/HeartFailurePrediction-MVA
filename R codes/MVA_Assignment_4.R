#### Aman ####

##############  Assignment 4 - PCA #####################
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
library(devtools)
library(ggbiplot)
library(factoextra)
library(rgl)
library(FactoMineR)

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)

# We check to see if we have categorical variables
# However we see all our variables are numeric
# Even the categorical ones are binary and already encoded into 1/0

# Let's quickly revise our correlation plot
# Correlation plot
M<-cor(data)
head(round(M,2))
corrplot(M, method="color")

# Since most of the correlations are low (Pearson's r < 0.25) ), 
# we don't particularly see a need for PCA
# We use PCA to reduce the dimensionality of the dataset
# PCA accomplishes this by capturing the variance in the dataset
# It get the components such that the are in the direction of the highest variance
# We also saw from EDA in last exercise that our VIF was quite low
# Indicating absence of multi-collinearity
# So reducing dimensionality may lead to loss of variance for our project
# However, for exposition, we will try PCA and analyse results

pca <- prcomp(data[,1:12],scale=TRUE)
summary(pca)
str(pca)
# Here, we see that we need 8 components to get cumulative proportion of
# variance equivalent to 0.78 (which isn't ideal given we have only 12 features)
# For convention, we would consider as many components as required
# to get in the range of 0.75-0.95
# Let us then consider 10 components (Cum prop. ~90%) instead of 12 reducing our dimensions
# from 12 to 10

# Let's plot the Scree diagrams
plot(pca, type="lines",main = "Scree diagram")
fviz_eig(pca)

# We can also see a cumulative plot
std_dev <- pca$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

# This shows that we need atleast 10 components for 90% variance
# and since we don't see a taper down in graph we can note that this isnt ideal

# Plotting PCA

# bi-plot which will use PC1 and PC2
ggbiplot(pca)

# Here, we can tell that ejection_fraction, diabetes, platelets all
# contribute to PC1 with higher values in these features moving the samples
# to the right
# Similarly we can tell that age, serum_creatinine contributes more towards PC2
# In PC1, we can see sex, smoking towards negative side of PC1
# In PC2, we can time towards negative side of PC2

# We can also tell which patients are similar to one other
# by adding rownames
# Let's use each row as a patient identifier, then,

ggbiplot(pca, labels=rownames(data))

# tell us that patient IDs-16,32,56 are similar as they cluster together
# This would ideally be helpful with more meaningful identifiers

# Let's also look at contribution by variables
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
# We see that age, sex, smoking contribute more to PC1 and PC2
# so we can try and visualize this is more detail by bi-plots with these

# Let us try to do that then,
# Let's plot the bi-plot with gender
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE, labels=rownames(data), groups=data$sex)

# A clear indicator that males indicated by 1 have more breadth
# in PC1 as opposed to Females indicated by 0 which are more
# narrow along with that we see +ve indication for females
# along PC1 and negative for males

# Let's plot the bi-plot with smoking
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE, labels=rownames(data), groups=data$smoking)

# A clear indicator that smokers indicated by 1 have less breadth
# in PC1 as opposed to non-smokers indicated by 0 which are more
# wider and to the positive side along with that we see +ve
# indication for non-smokers for PC1 and negative for smokers

# We will create an age range variable and do the same as well
data$age_tr[data$age < 50 & data$age >= 40]="40-50"
data$age_tr[data$age < 60 & data$age >= 50]="50-60" 
data$age_tr[data$age < 70 & data$age >= 60]="60-70"
data$age_tr[data$age < 80 & data$age >= 70]="70-80"
data$age_tr[data$age < 90 & data$age >= 80]="80-90"
data$age_tr[data$age < 100 & data$age >= 90]="90-100"

# And then plot the same result with 
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE, labels=rownames(data), groups=data$age_tr)

# Not much indication here other than higher age groups
# tend to be more spread out in PC2

# We can also look at PC3 and PC4
ggbiplot(pca,ellipse=TRUE,choices=c(3,4), labels=rownames(data))

# serum_creatinine, diabetes more towards PC3
# Platelets, creatinine_phosphokinase, and high bp more towards PC4

# By gender
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE,choices=c(3,4), labels=rownames(data), groups=data$sex)
# We note even spread in PC3, PC4 for gender
# By smoking
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE,choices=c(3,4), labels=rownames(data), groups=data$sex)
# We note even spread in PC3, PC4 for smoking
ggbiplot(pca,ellipse=TRUE, var.axes=FALSE, choices=c(3,4), labels=rownames(data), groups=data$age_tr)
# We note age group 40-50 with most spread in PC4

# Let us do a visualizations to see how much of each variable is present in each component
# We use factoextra and factominer for this

pca_viz <- PCA(data[,1:12], graph = FALSE,ncp =12)
var <- get_pca_var(pca_viz)

# We can now use the contrib function to get contribution of each variable
# to the PCs
var$contrib

corrplot(var$contrib, is.corr=FALSE, method="pie") 

# Key Observations -

# 1. Sex and Smoking are dominant in PC1
# 2. Age and time are dominant in PC2
# 3. Serum_Sodium is dominant in PC3
# 4. Platelets and creatinine_phosphokinase are dominant in PC4
# 5. creatinine_phosphokinase is dominant in PC5
# 6. Platelets and ejection_fraction are dominant in PC6
# 7. Anaemia is dominant in PC7
# 8. Diabetes is dominant in PC8
# 9. Age is dominant in PC9
# 10. Serum_Sodium, Serum_creatinine is dominant in PC10
# 11. Time, anaemia is dominant in PC11
# 12. Sex and smoking are dominant in PC12

# Note -
# We don't see a good combination of variables in any component
# PC12 is redundant as PC1 and gives same information

# Let us now combine the pca with dataset
data_pca <- cbind(data,pca$x)

# The new dataset now has 26 variables with PC1-PC12 added

# Now Let us check the means by death events
meansPC <- aggregate(data_pca[,15:26],by=list(DEATH_EVENT=data$DEATH_EVENT),mean)
meansPC

# Let us check stddev by death events
sdsPC <- aggregate(data_pca[,15:26],by=list(DEATH_EVENT=data$DEATH_EVENT),sd)
sdsPC

# We notice a clear difference in means however not much in variance
# This may indicate that PCs aren't doing a good job in
# segregating the death events from non-death events

# Let us perform t-tests
t.test(PC1~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC2~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC3~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC4~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC5~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC6~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC7~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC8~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC9~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC10~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC11~data_pca$DEATH_EVENT,data=data_pca)
t.test(PC12~data_pca$DEATH_EVENT,data=data_pca)

# We notice signifcant results in PC2, PC3, PC4, and PC11 at alpha=0.5

# Let us also perform F-ratio tests
var.test(PC1~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC2~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC3~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC4~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC5~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC6~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC7~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC8~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC9~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC10~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC11~data_pca$DEATH_EVENT,data=data_pca)
var.test(PC12~data_pca$DEATH_EVENT,data=data_pca)

# We notice signifcant results in PC2, PC4, PC5, PC6, PC8, PC9, and PC10

# Plotting the scores for the first and second components
plot(data_pca$PC1, data_pca$PC2,pch=ifelse(data_pca$DEATH_EVENT == "1",1,16),xlab="PC1", ylab="PC2", main="Heart disease patient against values for PC1 & PC2")
abline(h=0)
abline(v=0)
legend("bottomleft", legend=c("Death_Event=1","Death_Event=0"), pch=c(1,16))

# We do note that survivors seem to be closer to average than those who died
# Also recall the definition of PC1 and PC2
# PC1 was sex, smoking dominant
# PC2 was age, time dominant
# This also tells us that non-survivors were on the extremes of ages and follow-up
# period 

### pca - prediction 

# We can try a prediction with pca by splitting our data into train and test
# Finding the PCs on train and validating on test data

data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')

# Split data into 2 parts for pca training (75%) and prediction (25%)

set.seed(1)
samp <- sample(nrow(data), nrow(data)*0.75)
data.train <- data[samp,]
data.valid <- data[-samp,]

# conduct PCA on training dataset
pca <- prcomp(data.train[,1:12], retx=TRUE, center=TRUE, scale=TRUE)
expl.var <- round(pca$sdev^2/sum(pca$sdev^2)*100) # percent explained variance

# prediction of PCs for validation dataset
pred <- predict(pca, newdata=data.valid[,1:12])
view(pred)

# Let us take first 10 components that explain 90% variance in data
# and do the same

train.data <- data.frame(DEATH_EVENT=data.train$DEATH_EVENT, pca$x)
train.data <- train.data[,1:11]

test.data <- predict(pca, newdata = data.valid)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:10]
view(test.data)

# This finally gives us the test data with PC1-10
# Our final conclusion however remains the same that PCA isn't ideal
# for modeling purpose in our project


###### This concludes our analysis of PCA in our dataset ######
###############################################################

