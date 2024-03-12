
#The aim of the project is to perform clustering on the countries dataset to see any tendency of those countries.
#then, PCA dimension reduction method will be conducted to see how many variables are enough to explain the
#majority of information. After that, clustering will be performed again to see how the clusters changed


library(dplyr)
library(tidyr)
library(tidyverse)
library(caret)
library(ggplot2)
getwd()
setwd("C:/Users/Piotr/Downloads")
Data<-read.csv("countries/countries.csv")

#Let's install some packages for further magic  thingies
library(factoextra)
library(flexclust)
library(fpc)
library(clustertend)
library(cluster)
library(ClusterR)
library(cleandata)
library(knitr)




#1. Data preparation

#The dataset was taken from kaggle (https://www.kaggle.com/datasets/fernandol/countries-of-the-world) with 
#variables like infant mortality, deathrate, birthrate, GDP per capita - all valuable information that will be 
#used for clustering 

#Let's separate character variables(country, region) with numeric variables (the rest)
Data2<-Data
Data2<-Data2%>%relocate(Climate, .after=Service)

#The dataset contains commas instead of dots as a separator, let's swap commas with dots and then 
#replace character with numeric in order to cluster
replace_commas <- function(column) {
  gsub(",", ".", column)
}

Data3 <- Data2 %>% 
  mutate(across(where(is.character), replace_commas))

convert_chr_to_numeric <- function(column) {
  if(is.character(column)) {
    as.numeric(column)
  } else {
    column
  }
}


Data4 <- as.data.frame(lapply(Data3[3:20], convert_chr_to_numeric))

#We will add back columns "Country" and "Region" for easier comparison later
Data4 <- cbind(Data4,Data3[1:2])
#Remove rows with NAs
Data4<-na.omit(Data4)

#Let's use z scaling on all the numerical variables (except the column "Climate" which takes values from 0 to 4)
Data5<-scale(Data4[1:17])
#Add "Climate" back to our now scaled dataset - we're not adding Country and Region because they are 
#of character class 
Data5<-cbind(Data5,Data4$Climate)
#The data has been prepared.

#2. Clustering

#Before we start clustering, we need to check our clustering tendency(using hopkins statistic)
#and the optimal amount of clusters (using elbow and silhouette methods)

#Calculating hopkins statistics
get_clust_tendency(Data5,2,graph=TRUE)
#Our Hopkins statistic is equal to 0.87 which indicates very high clusterability tendency. This is a good sign

#Silhouette method
opt_silhouette<-Optimal_Clusters_KMeans(Data5, max_clusters=10, plot_clusters=TRUE, criterion="silhouette")
#Elbow method
opt_elbow<-Optimal_Clusters_KMeans(Data5, max_clusters=10, plot_clusters = TRUE)
#Both silhouette and elbow methods imply that it is optimal to use 2 clusters

#We're ready to perform clustering
#We will use the k-means clustering

#K-means
cluster_km<-eclust(Data5,FUNcluster="kmeans",2)
#silhouette width 
fviz_silhouette(cluster_km)
#Average Silhouette is equal to 0.23, which indicates low clustering quality. 

#Let's see what story the clusters can tell us, we will choose 10 random countries and regions, 
#some of their characteristics and their clusters 
Data4$cluster<-cluster_km$cluster
Data_not_scaled<-Data4
Data_not_scaled[sample(nrow(Data_not_scaled),10),c(1, 3, 13, 14,19,20,21)]

#Based on those results, we can deduce that the cluster 2 belongs to more developed countries, where deathrate 
#and infant mortality are low, literacy is high - tendencies found in developed or almost developed countries
#Cluster 1 belongs to (but not always) developing and/or poorer countries

#Now, let's perform PCA now and perform the same k-means clustering on it

#3. Principal Component Analysis 

#Principal Component Analysis is used to decrease the number of dimensions while containing most information,
#a tool used to combat the curse of high dimensionality
pca <- prcomp(Data5, center=FALSE, scale=FALSE)
#Let's see how much of variance do those components explain:
summary(pca)

#Scree plot
fviz_eig(pca, choice='eigenvalue')

#Now with percentages
fviz_eig(pca)


fviz_pca_var(pca)
#The two first principals explain 47.3% of the variance which is not a lot. We will perform kaiser rule to 
#establish how many dimensions are explain the majority of the dataset

#Kaiser rule

#We will now see the eigenvalues of our dataset. According to kaiser rule, a good stopping point 
#is to choose only those who have value of 1 or higher.
Data5.cov<-cov(Data5)
Data5.eigen<-eigen(Data5.cov) 
Data5.eigen$value
#only those above 1 should be chosen. According to the output PC1 to PC5 are retained.

#In our case, the first 5 components explain 75% of the variance
library(gridExtra)
PC1 <- fviz_contrib(pca, choice = "var", axes = 1)
PC2 <- fviz_contrib(pca, choice = "var", axes = 2)
PC3 <- fviz_contrib(pca, choice = "var", axes = 3)
PC4 <- fviz_contrib(pca, choice = "var", axes = 4)
PC5 <- fviz_contrib(pca, choice = "var", axes = 5)
grid.arrange(PC1, PC2)
grid.arrange(PC3, PC4)
grid.arrange(PC5)

#Looking at both the graphs of Contribution of variables to Dim-1 and Dim-2 we can see that Infant mortality
#per 1000 births is the leading variable of the dataset with about 36% contribution for dim-1 and 60% for dim-2.
#For dim-1, some other variables also contriute like birthrate, phones per 1000 people, literacy, gdp per capita
#or agriculture.As for dim-2, the situation is similar but those variables contribute to a much less degree
#than infant mortality.



#4. performing clustering on PCA'd dataset

pca_after<-prcomp(Data5,center=FALSE, scale.=FALSE, rank. = 5)
#pca_after$x - this is our PCA'd dataframe (matrix to be exact) we will perform clustering 

#Let's perform the same steps as before, hopkins statistics as well as silhouette and elbow methods

#Hopkins statistics
get_clust_tendency(pca_after$x,2,graph=TRUE)
#It's equal to 0.59, way lower than before (0.87) and closer to 0.5 which is not a good sign

#Silhouette PCA
opt_silhouette_pca<-Optimal_Clusters_KMeans(pca_after$x, max_clusters=10, plot_clusters=TRUE, criterion="silhouette")
#Elbow PCA
opt_elbow_pca<-Optimal_Clusters_KMeans(pca_after$x, max_clusters=10, plot_clusters = TRUE)

#Based on elbow and silhouette methods, we will choose 2 clusters again

#Perform K-means on PCA'd dataset
cluster_pca_kmeans<-eclust(pca_after$x,FUNcluster="kmeans",2)


fviz_silhouette(cluster_pca_kmeans)
#Although the average silhouette width is higher (0.32) than it's non-PCA counterpart (0.23) 
#the low hopkins statistics and high number of PCs (even 5 PCs explain only 0.75 of the variance) might
#indicate that PCA is not the best dimension reduction method for this dataset

#5.Summary

#Clustering performed on this dataset helped with seeing some tendencies, like the division to poorer/developing
#countries (cluster 1) and developed/almost developed/richer countries (cluster 2)
#although there is a good tendency shown as well as hopkins statistics being high (0.87) the average
#width of the silhouette was low (0.23) which, in my opinion, might give some guidance but not 
#clear conclusions
#Dimension reduction and clustering on that dataset was used to see if the results from this average width 
#could be improved. While it did improve, hopkins statistic decreased significantly compared to its non-reduction
#Counterpart. This might indicate that while the dataset might be good to reduce dimensions, PCA might not be
#the appropriate method 

#6. Sources
#https://www.datanovia.com/en/lessons/assessing-clustering-tendency/ 
#https://sanchitamangale12.medium.com/scree-plot-733ed72c8608 
#https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411
#ChatGPT  (converting chr to numeric function and replace commas function)


