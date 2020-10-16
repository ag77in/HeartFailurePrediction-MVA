#### Aman ####

##############  Assignment 5 - Clustering Methods #####################
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
library(cluster)
library(magrittr)
library(NbClust)

# reading data
data <- read.csv('/Users/mac/Downloads/heart_failure_clinical_records_dataset.csv')
str(data)

# Let's scale the data for the independent variables - we will invoke dplyr now
data_2 <- data[,-13] %>%
  na.omit() %>%          # Remove missing values (NA)
  scale()                # Scale variables

###### Let's assess clustering tendency of the data first #####

# We use the visual and the hopkins statistic approach for this
# With hopkins' statistic, we see how close the value is to 1
# to identify if our data is actually clusterable

gradient.color <- list(low = "steelblue",  high = "white")
data_2 %>% get_clust_tendency(n = 50, gradient = gradient.color)

# Our hopkin's statistic is 0.72
# This value is still above 0.5 but no as close to 1 as we'd like
# In the visual approach we cannot really see dark boxes along
# the diaganol as well

###### Partitioning Cluster methods #######

# Let's compute distance matrix between rows of our heart failure clinical data
res.dist <- get_dist(data_2, stand = TRUE, method = "pearson")

fviz_dist(res.dist, 
          gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# It is very hard to make sense of distances between patients with this method
# We will have to try a logic or come up with groups so the
# Clustering methods we apply on our data make sense

# Let's however try k-means and explore 4 methods of choosing the optimal clusters
# k - means 
# Elbow method
fviz_nbclust(data_2, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(data_2, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# Gap method 
set.seed(123)
fviz_nbclust(data_2, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")

# Nbclust method
res.nbclust <-  data[,-13] %>%
  scale() %>%
  NbClust(distance = "euclidean",
          min.nc = 2, max.nc = 10, 
          method = "complete", index ="all")

fviz_nbclust(res.nbclust, ggtheme = theme_minimal())

# We see elbow gives 4 optimal clusters
# Silhouettes gives 2
# Gap doesn't converge
# Nbclust gives 2 as well

# It is important to understand that our data may not be clusterable

# Note- Some theory behind gap
# The gap statistic method is used in addition to elbow plot
# to determine the optimal number of clusters in a data
# The ideal point in gap is where gap statistic is maximised
# We can see 8 as optimal cluster in graph but we also use
# the 1-standard error rule
# Choosing the cluster size to be the smallest k such that 
# Gap(k) >= Gap (K+1) - s(k+1)
# Choosing this criteria we see that happens at k=1 itself
# This is Gap's way of saying that the data (atleast in this form)
# should not be clustered

# Note - Some theory behind Si
# We use the silhouette coefficient to determine
# how good clustering is
# The silhouette coefficient measures how similar an object i is to 
# the other objects in its own cluster versus those in the neighbor cluster
# It ranges from -1 to 1

# Given the above, it seems clear that the
# methods are unable to give a robust cluster solution
# But robust is defined by validation methods

# For exposition, however let's try the solution
# for 4 clusters
# K-means
set.seed(123)
km.res <- kmeans(data_2, 4, nstart = 25)
fviz_cluster(km.res, data = data_2,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

# Compute PAM as well - pam is more used these days
# as it isn't affected by outliers as much and uses medoids
pam.res <- pam(data_2, 4)
# Visualize
fviz_cluster(pam.res)

###### Heirarchical Cluster methods #######

set.seed(123)
# Enhanced hierarchical clustering, cut in 4 groups
res.hc <- data[, -13] %>%
  scale() %>%
  eclust("hclust", k = 4, graph = FALSE)

# Visualize with factoextra
fviz_dend(res.hc, k = 4, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE, # Add rectangle around groups
          rect_border = "red" # add rect border
)


fviz_silhouette(res.hc)

# We note the average silhouette coefficients in
# Cluster 2 with 6 observation is 0.38
# but none of the clusters have values closer to 1
# indicating that the solution isn't robust
# as distance within cluster isn't as different
# as any other neighboring points

### Other hc methods ####

# Dissimilarity matrix
d <- dist(data_2, method = "euclidean")
# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )
# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)

# methods to assess
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(data_2, method = x)$ac
}
map_dbl(m, ac)

hc3 <- agnes(data_2, method = "ward")
pltree(hc3, cex = 0.6, hang = -1, main = "Dendrogram of agnes")

# We notice ambiguity in results for clustering
# While hopkins suggest data is clusterable, we do not
# actually obtain a good optimal number of clusters
# as both gap statistic and silhouette plots don't really
# give us good info. on what the number is
# and hence we don't have good validation of our
# clustering results

# But of course, we can still profile our clusters and see
# if they make some sense in this dataset

####### Profiling the results #######

# K-Means Cluster Analysis
fit <- kmeans(data_2, 4) # 4 cluster solution
# get cluster means
aggregate(data_2,by=list(fit$cluster),FUN=mean)
# append cluster assignment
data_3 <- data.frame(data_2, fit$cluster)
data_4 <- cbind(data_3, DEATH_EVENT = data$DEATH_EVENT)
# We can also check proprtion of variance explained by 4 cluster solution
perc.var.4 <- round(100*(1 - fit$betweenss/fit$totss),1)
names(perc.var.4) <- "Perc. 4 clus"
perc.var.4
# we create orginal data with assignment
org_data <- as.data.frame(t(apply(data_2, 1, function(r)r*attr(data_2,'scaled:scale') + attr(data_2, 'scaled:center'))))
data_with_clus_assgn <- cbind(org_data,fit.cluster=data_4$fit.cluster,DEATH_EVENT=data$DEATH_EVENT)

attach(data_with_clus_assgn)
# Clusters with death event visualization
ggplot(data_with_clus_assgn,aes(x = fit.cluster,fill=DEATH_EVENT))+geom_histogram(binwidth = 1, color = "black", 
                                            fill = "red",alpha = 0.5)+
  labs(title = "Clusters", 
       caption = "Cluster assignments with DEATH_EVENT")+
  theme(plot.caption = element_text(hjust = 0.5,face = "italic"))

data_5 <- data_with_clus_assgn %>%
  group_by(fit.cluster) %>%
  dplyr::summarise(cnt = n(),cnt_death_event = sum(DEATH_EVENT), 
                   prop_death_event = sum(DEATH_EVENT)/n())

data_5

# We note that the death proportion is highest in cluster 3 ~73%
# (highest percentage of 1s) whereas others
# have pretty much similar death proportion

# Clusters with age 
# Let's create age ranges
data$age_tr[data$age < 50 & data$age >= 40]="40-50"
data$age_tr[data$age < 60 & data$age >= 50]="50-60" 
data$age_tr[data$age < 70 & data$age >= 60]="60-70"
data$age_tr[data$age < 80 & data$age >= 70]="70-80"
data$age_tr[data$age < 90 & data$age >= 80]="80-90"
data$age_tr[data$age < 100 & data$age >= 90]="90-100"

data_with_clus_assgn <- cbind(data_with_clus_assgn,age_tr=data$age_tr)

# Clusters with age and death event 
table(data_with_clus_assgn$fit.cluster,data_with_clus_assgn$age_tr)

# Numerically we see that cluster 3 is dominated by age range 60+
# We also see cluster 1 dominated by individuals with age range < 60
# Hence it makes sense that cluster 3 has a higher death event rate

data_with_clus_assgn$DEATH_EVENT <- factor(data_with_clus_assgn$DEATH_EVENT)
data_with_clus_assgn$fit.cluster <- factor(data_with_clus_assgn$fit.cluster)
str(data_with_clus_assgn)

# Let's check each variable with cluster assignment

# Age
ggplot(data_with_clus_assgn,aes(x = age, fill = fit.cluster))+geom_histogram(binwidth = 5, 
  position = "identity",
  alpha = 0.5,color = "black")+scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"))+
  labs(caption = "Age Distribution with Clusters")+
  theme(plot.caption = element_text(hjust = 0.5,face = "italic"))+
  scale_x_continuous(breaks = seq(40,100,10))

# It validates our point that cluster 3 is more towards higher age groups

# Gender
ggplot(data_with_clus_assgn, aes(x = factor(sex), fill = fit.cluster))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("Sex:Female","Sex:Male"))+
  scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"), name = "Sex")

# We see cluster 4 is more dominated by female population whereas
# We see cluster 1,2 is more dominated by male population (even 3 to some extent)

# Anaemia 
ggplot(data_with_clus_assgn, aes(x = factor(anaemia), fill = fit.cluster))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("No","Yes"))+
  scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"), name = "Anaemia")

# We see cluster 3 and to some extent cluster 4 have higher
# proprotion of anaemic individuals

# Creatinine_phosphokinase
aggregate(data_with_clus_assgn[, c('creatinine_phosphokinase')], list(data_with_clus_assgn$fit.cluster), mean)

# We see the high creatinine levels in cluster 1

# Diabetes
ggplot(data_with_clus_assgn, aes(x = factor(diabetes), fill = fit.cluster))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("No","Yes"))+
  scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"), name = "Diabetes")

# We see high diabetic concentration in Cluster 4

# ejection_fraction
aggregate(data_with_clus_assgn[, c('ejection_fraction')], list(data_with_clus_assgn$fit.cluster), mean)

# We see marginally high ejection fraction in cluster 4

# High_blood_pressure 
ggplot(data_with_clus_assgn, aes(x = factor(high_blood_pressure), fill = fit.cluster))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("No","Yes"))+
  scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"), name = "High BP")

# We see high BP differ in cluster 4 and also less concentration in Cluster 1

# platelets
aggregate(data_with_clus_assgn[, c('platelets')], list(data_with_clus_assgn$fit.cluster), mean)

# Not much difference across clusters

# serum_creatinine
aggregate(data_with_clus_assgn[, c('serum_creatinine')], list(data_with_clus_assgn$fit.cluster), mean)

# We notice high serum_creatinine levels in cluster 3

# serum_sodium
aggregate(data_with_clus_assgn[, c('serum_sodium')], list(data_with_clus_assgn$fit.cluster), mean)

# We notice no significant difference across clusters

# Smoking 
ggplot(data_with_clus_assgn, aes(x = factor(smoking), fill = fit.cluster))+
  geom_bar(position = "fill")+
  scale_x_discrete(labels  = c("No","Yes"))+
  scale_fill_manual(values = c("#FF0099", "#CCFF00","#33FF00","#FF9900"), name = "Smoking")

# We notice smoking dominates cluster 2 and is least for cluster 4 
# and to some extent low for cluster 1 as well

# follow-up period
aggregate(data_with_clus_assgn[, c('time')], list(data_with_clus_assgn$fit.cluster), mean)

# We notice high follow up period for cluster 1 and least for cluster 3

##### Final Profiling Results ######

# Cluster 1 characteristics -
# Cluster 1 has high avg. creatinine phosphokinase and a high 
# follow up period with the least death rate and ejection fraction.
# They are also males with low anaemiac
# condition and least in age compared to other clusters and have not
# a low % of high bp cases

# Single line summary ->
# Males with low anaemic issues and low high bp cases
# with high creatinine phosphokinase and low ejection fraction
# and have shorter follow up periods

# Cluster 2 characteristics -
# Cluster 2 has again low anaemic only male population.
# They also consist only of smokers 

# Single line summary ->
# Only Males who are smokers with low anaemic issues 

# Cluster 3 characteristics -
# Cluster 3 is higher age group, more anaemic, high bp individuals
# with high serum_creatinine and a low follow up period
# with the highest death rate

# Single line summary ->
# Higher age group male dominated with high bp issues,
# and high serum_creatinine and a low follow up period

# Cluster 4 characteristics -
# Cluster 4 is more diabetic female dominated population
# with a high ejection_fraction but least smokers

# Single line summary ->
# Female dominated diabetic individuals with high bp
# issues and high serum_creatinine and a low follow up period

# When we relate these profiles to death events, we see why
# cluster 3 has disproportionate death event rate as compared to other
# clusters

###### This concludes our analysis of Clustering in our dataset ######
###############################################################

