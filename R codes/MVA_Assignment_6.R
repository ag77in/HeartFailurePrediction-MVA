#### Aman ####

##############  Assignment 6 - Factor Analysis #####################
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

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)


# Let's quickly revise our correlation plot and see if factor analysis is appropriate
# Correlation plot

M<-cor(data)
head(round(M,2))
corrplot(M, method="color")

#Since most of the correlations are low (Pearson's r < 0.25) ), 
#we don't particularly see a need for Factor Analysis since 
#we use Factor Analysis to understand the latent factors in the data 
#However, we can see that given these are patient details, we may
#try and understand factors such as patient demographics (age, sex),
#patient lifestyle (smoking, diabetes, high bp), patient physiological 
#makeup (serum sodidum, creatinine_phosphokinase), patient genetics 
#(bp, anaemia). While this is our intuition  before we begin, 
#only once we see the factor analysis results will we be able to 
#comment more appropriately.

#scale the data
data_fact <- as.data.frame(scale(data[,1:12],center = TRUE, scale = TRUE))

# Tests to see if factor analysis is appropriate on the data

KMO(data_fact)

### Bartlett’s test

#We also perform the Bartlett’s test which allows us to 
#compare the variance of two or more samples to determine 
#whether they are drawn from populations with equal variance.

bartlett.test(data_fact)

# Let us now perform Factor Anaysis on our dataset

# perform factor analysis
data.fa <- factanal(data_fact, factors = 2)
data.fa

#Here, we see high uniqueness (>0.7) for most variables indicating that factors
#don't account well for the variance. But we do note that sex variable
#has the least uniqueness (0.233). 

#We also note that cumulative variance explained is only 15.8% which isn't 
#great and we may have to use more than 2 factors

#squaring the loadings to assess communality
apply(data.fa$loadings^2,1,sum)

# Let's try and interpret the factors

#We perform three factor models - one with no rotation, one with varimax rotation,
#and finally one with promax rotation and see the results

data.fa.none <- factanal(data_fact, factors = 2, rotation = "none")
data.fa.varimax <- factanal(data_fact, factors = 2, rotation = "varimax")
data.fa.promax <- factanal(data_fact, factors = 2, rotation = "promax")

par(mfrow = c(1,3))
plot(data.fa.none$loadings[,1], 
     data.fa.none$loadings[,2],
     xlab = "Factor 1", 
     ylab = "Factor 2", 
     ylim = c(-1,1),
     xlim = c(-1,1),
     main = "No rotation")
abline(h = 0, v = 0)

plot(data.fa.varimax$loadings[,1], 
     data.fa.varimax$loadings[,2],
     xlab = "Factor 1", 
     ylab = "Factor 2", 
     ylim = c(-1,1),
     xlim = c(-1,1),
     main = "Varimax rotation")

text(data.fa.varimax$loadings[,1]-0.08, 
     data.fa.varimax$loadings[,2]+0.08,
     colnames(data),
     col="blue")
abline(h = 0, v = 0)

plot(data.fa.promax$loadings[,1], 
     data.fa.promax$loadings[,2],
     xlab = "Factor 1", 
     ylab = "Factor 2",
     ylim = c(-1,1),
     xlim = c(-1,1),
     main = "Promax rotation")
abline(h = 0, v = 0)

#We can see that factor 1 corresponds to smoking, sex, platelets,
#, ejection_fraction and diabetes whereas factor 2 corresponds to
#age, anaemia, high bp, serum_creatinine and time among others.
#We cannot clearly name the factors at this point in line with 
#our intuition.

# Let's plot the results

### Maximum Likelihood Factor Analysis with 2 factors
# Maximum Likelihood Factor Analysis
# entering raw data and extracting 2 factors,
# with varimax rotation
fit <- factanal(data_fact, 2, rotation="varimax")
# plot factor 1 by factor 2
load <- fit$loadings[,1:2]
plot(load,type="n") # set up plot
text(load,labels=names(data_fact),cex=.7) # add variable names```

# However, there is a better method to first determine number of Factors to Extract
ev <- eigen(cor(data_fact)) # get eigenvalues
ap <- parallel(subject=nrow(data_fact),var=ncol(data_fact),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)

#This is interesting now since our interpretation might be more relevant with 
#3 factors.

# Factor Analysis (n=3 factors)

data.fa.none <- factanal(data_fact, factors = 3, rotation = "none")
data.fa.none
scatterplot3d(as.data.frame(unclass(data.fa.none$loadings)), 
              main="3D factor loadings", color=1:ncol(data_fact), pch=20)
pairs(data.fa.none$loadings, col=1:ncol(data_fact), 
      upper.panel=NULL, main="Factor loadings")
par(xpd=TRUE) 
legend('topright', bty='n', pch='o', col=1:ncol(data_fact), y.intersp=0.5,
       attr(data.fa.none$loadings, 'dimnames')[[1]], title="Variables")

#This is a lot more interesting since now if we try and interpret the 3 factors we see
#that Factor 1 is sex, smoking dominant while factor 2 is ejection_fraction
#and serum component dominant while factor 3 is age, anaemia, high bp
#dominant. While not exactly the same as intuition, we do note
#that Factor 1 can be interpreted as patient demographics/ lifestyle 
#feature as males tend to smoke more, while factor 2 is 
#the physiological makeup we discussed about earlier and 
#factor 3 is the again patient demographics but also genetics
#as variables with blood pressure and anaemia show up along with age.

### Conclusion -

### Factor 1 - Patient Demographics / Lifestyle
### Factor 2 - Patient Physiological Makeup
### Factor 3 - Patient Demographics / Genetics

# Factor Analysis (n=4 factors)

data.fa.none <- factanal(data_fact, factors = 4, rotation = "none")
data.fa.none
pairs(data.fa.none$loadings, col=1:ncol(data_fact), 
      upper.panel=NULL, main="Factor loadings")
par(xpd=TRUE) 
legend('topright', bty='n', pch='o', col=1:ncol(data_fact), y.intersp=0.5,
       attr(data.fa.none$loadings, 'dimnames')[[1]], title="Variables")

#Again an interesting result since if we try and interpret the 4 factors we see
#that Factor 1 is serum_sodium dominant (Physiological makeup), 
#while Factor 2 is sex and smoking dominant (Patient Lifestyle) 
#and Factor 3 is serum_sodium and high bp dominant (Physiological makeup 
# & lifestyle) and Factor 4 is age, anaemia dominant (Patient Demographics 
#& genetics). We notice some overlaps here so perhaps, 3 factors would be 
#the ideal choice, however do note that p-values aren't significant in
# either results.

###  Conclusion -

### Factor 1 - Physiological makeup
### Factor 2 - Patient Lifestyle
### Factor 3 - Physiological makeup & lifestyle
### Factor 4 - Patient demographics & Genetics

### Another method - we can try the psych package as well for n=3 factors
fit.pc <- principal(data_fact, nfactors=3, rotate="varimax")
fit.pc
round(fit.pc$values, 3)
fit.pc$loadings
# Loadings with more digits
for (i in c(1,2,3)) { print(fit.pc$loadings[[1,i]])}
# Communalities
fit.pc$communality

# Play with FA utilities
fa.parallel(data_fact) # See factor recommendation
fa.plot(fit.pc) # See Correlations within Factors
fa.diagram(fit.pc) # Visualize the relationship
vss(data_fact) # See Factor recommendations for a simple structure

#Note: While we see our data isn't perhaps ideal for Factor Analysis,
#we can gauge some interesting results and given this dataset is
##part of a study of only 299 patients, the latent factors 
#may be more prominent in the population distribution.

##### This concludes our approach to Factor Analysis in our dataset ######
###############################################################
