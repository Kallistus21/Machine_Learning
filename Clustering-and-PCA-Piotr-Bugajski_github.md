Clustering and PCA Dimension Reduction on Countries Data
================

## Introduction:

The aim of the project is to perform clustering on the countries dataset
to see any tendency of those countries. then, PCA dimension reduction
method will be conducted to see how many variables are enough to explain
the majority of information. After that, clustering will be performed
again to see how the clusters changed

``` r
library(dplyr)
library(tidyr)
library(tidyverse)
library(caret)
library(ggplot2)
library(factoextra)
library(flexclust)
library(fpc)
library(clustertend)
library(cluster)
library(ClusterR)
library(cleandata)
library(knitr)
```

## 1. Data preparation

The dataset was taken from kaggle
(<https://www.kaggle.com/datasets/fernandol/countries-of-the-world>)
with variables like infant mortality, deathrate, birthrate, gpd per
capita - all valuable information that will be used for clustering

``` r
Data<-read.csv("countries/countries.csv")
```

Let’s separate character variables(country, region) with numeric
variables (the rest)

``` r
Data2<-Data
Data2<-Data2%>%relocate(Climate, .after=Service)
```

The dataset contains commas instead of dots as a separator, let’s swap
commas with dots and then replace character with numeric in order to
cluster

``` r
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
```

We will add back countries and regions for easier comparison later

``` r
Data4 <- cbind(Data4,Data3[1:2])
```

Remove rows with NAs

``` r
Data4<-na.omit(Data4)
```

Let’s use z scaling on all the numerical variables (except column
“climate” which takes values from 0 to 4)

``` r
Data5<-scale(Data4[1:17])
```

Add “Climate” back to our now scaled dataset - we’re not adding Country
and Region because they are characters

``` r
Data5<-cbind(Data5,Data4$Climate)
```

The data has been prepared.

## 2. Clustering

Before we start clustering, we need to check our clustering
tendency(using hopkins statistic) and the optimal amount of clusters
(using elbow and silhouette methods)

### Calculating hopkin’s statistics

``` r
get_clust_tendency(Data5,2,graph=TRUE)
```

    ## $hopkins_stat
    ## [1] 0.8693032
    ## 
    ## $plot

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Our hopkin’s statistic is equal to 0.87 which indicates very high
clusterability tendency. This is a good sign

### Silhouette method

``` r
opt_silhouette<-Optimal_Clusters_KMeans(Data5, max_clusters=10, plot_clusters=TRUE, criterion="silhouette")
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

### Elbow method

``` r
opt_elbow<-Optimal_Clusters_KMeans(Data5, max_clusters=10, plot_clusters = TRUE)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

Both silhouette and elbow methods imply that it is optimal to use 2
clusters

We’re ready to perform clustering We will use the k-means clustering

### K-means

``` r
cluster_km<-eclust(Data5,FUNcluster="kmeans",2)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

### Silhouette width

``` r
fviz_silhouette(cluster_km)
```

    ##   cluster size ave.sil.width
    ## 1       1   64          0.31
    ## 2       2  115          0.18

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

Average Silhouette is equal to 0.23, which indicates low clustering
quality.

Let’s see what story the clusters can tell us, we will choose 10 random
countries and regions, some of their characteristics and their clusters

``` r
Data4$cluster<-cluster_km$cluster
Data_not_scaled<-Data4
Data_not_scaled[sample(nrow(Data_not_scaled),10),c(1, 3, 13, 14,19,20,21)]
```

    ##     Population Pop..Density..per.sq..mi.. Birthrate Deathrate
    ## 214   60609153                      247.6     10.71     10.13
    ## 58     9183984                      188.5     23.22      5.73
    ## 150    4076140                       15.2     13.76      7.53
    ## 49     4075261                       79.8     18.32      4.36
    ## 17   147365352                     1023.4     29.80      8.27
    ## 225   21456188                       40.6     42.89      8.30
    ## 194     439117                        2.7     18.02      7.27
    ## 108   34707817                       59.6     39.72     14.02
    ## 110   23113019                      191.8     15.54      7.13
    ## 215  298444215                       31.0     14.14      8.26
    ##                 Country                              Region cluster
    ## 214     United Kingdom  WESTERN EUROPE                            2
    ## 58  Dominican Republic              LATIN AMER. & CARIB           2
    ## 150        New Zealand  OCEANIA                                   2
    ## 49          Costa Rica              LATIN AMER. & CARIB           2
    ## 17          Bangladesh        ASIA (EX. NEAR EAST)                1
    ## 225              Yemen  NEAR EAST                                 1
    ## 194           Suriname              LATIN AMER. & CARIB           2
    ## 108              Kenya  SUB-SAHARAN AFRICA                        1
    ## 110       Korea. North        ASIA (EX. NEAR EAST)                2
    ## 215      United States  NORTHERN AMERICA                          2

Based on those results, we can deduce that the cluster 2 belongs to more
developed countries, where deathrate and infant mortality are low,
literacy is high - tendencies found in developed or almost developed
countries Cluster 1 belongs to (but not always) developing and/or poorer
countries.

Now, let’s perform PCA now and perform the same K-means clustering on it

# 3.Principal Component Analysis

Principal Component Analysis is used to decrease the number of
dimensions while containing most information a tool used to combat the
curse of high dimensionality

``` r
pca <- prcomp(Data5, center=FALSE, scale=FALSE)
```

Let’s see how much of variance do those components explain:

``` r
summary(pca)
```

    ## Importance of components:
    ##                           PC1    PC2    PC3     PC4     PC5     PC6     PC7
    ## Standard deviation     2.4177 2.1311 1.5743 1.36346 1.23752 1.16819 0.95443
    ## Proportion of Variance 0.2662 0.2068 0.1129 0.08466 0.06974 0.06214 0.04148
    ## Cumulative Proportion  0.2662 0.4730 0.5859 0.67051 0.74025 0.80240 0.84388
    ##                            PC8     PC9    PC10    PC11    PC12   PC13    PC14
    ## Standard deviation     0.89013 0.83215 0.68612 0.68024 0.61478 0.5526 0.40269
    ## Proportion of Variance 0.03608 0.03153 0.02144 0.02107 0.01721 0.0139 0.00738
    ## Cumulative Proportion  0.87996 0.91150 0.93294 0.95401 0.97122 0.9851 0.99251
    ##                           PC15    PC16    PC17     PC18
    ## Standard deviation     0.31329 0.25625 0.02618 0.000386
    ## Proportion of Variance 0.00447 0.00299 0.00003 0.000000
    ## Cumulative Proportion  0.99698 0.99997 1.00000 1.000000

### Scree plot

``` r
fviz_eig(pca, choice='eigenvalue')
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

### Now, the scree plot with percentages

``` r
fviz_eig(pca)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
fviz_pca_var(pca)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-19-2.png)<!-- -->

The two first principals explain 47.3% of the variance which is not a
lot. We will perform Kaiser rule to establish how many dimensions are
explain the majority of the dataset

### Kaiser rule

We will now see the eigenvalues of our dataset. According to kaiser
rule, a good stopping point is to choose only those who have value of 1
or higher.

``` r
Data5.cov<-cov(Data5)
Data5.eigen<-eigen(Data5.cov) 
Data5.eigen$value
```

    ##  [1] 5.454601e+00 2.520027e+00 1.875547e+00 1.565978e+00 1.411058e+00
    ##  [6] 9.208962e-01 8.123057e-01 7.162087e-01 5.019030e-01 4.675885e-01
    ## [11] 3.782903e-01 3.178624e-01 2.232605e-01 1.579442e-01 9.727503e-02
    ## [16] 6.522894e-02 6.855649e-04 1.480658e-07

Only those above 1 should be chosen. According to the output PC1 to PC5
are retained.

In our case, the first 5 components explain 75% of the variance

``` r
library(gridExtra)
```

    ## Warning: package 'gridExtra' was built under R version 4.3.2

    ## 
    ## Attaching package: 'gridExtra'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
PC1 <- fviz_contrib(pca, choice = "var", axes = 1)
PC2 <- fviz_contrib(pca, choice = "var", axes = 2)
PC3 <- fviz_contrib(pca, choice = "var", axes = 3)
PC4 <- fviz_contrib(pca, choice = "var", axes = 4)
PC5 <- fviz_contrib(pca, choice = "var", axes = 5)
grid.arrange(PC1, PC2)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
grid.arrange(PC3, PC4)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-21-2.png)<!-- -->

``` r
grid.arrange(PC5)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-21-3.png)<!-- -->

Looking at both the graphs of Contribution of variables to Dim-1 and
Dim-2 we can see that Infant mortality per 1000 births is the leading
variable of the dataset with about 36% contribution for dim-1 and 60%
for dim-2. For dim-1, some other variables also contriute like
birthrate, phones per 1000 people, literacy, gdp per capita or
agriculture.As for dim-2, the situation is similar but those variables
contribute to a much less degree than infant mortality.

## 4. performing clustering on PCA’d dataset

``` r
pca_after<-prcomp(Data5,center=FALSE, scale.=FALSE, rank. = 5)
```

Let’s perform the same steps as before, hopkins statistics as well as
silhouette and elbow methods

### Hopkins statistics

``` r
get_clust_tendency(pca_after$x,2,graph=TRUE)
```

    ## $hopkins_stat
    ## [1] 0.5878839
    ## 
    ## $plot

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

It’s equal to 0.59, way lower than before (0.87) and closer to 0.5 which
is not a good sign

### Silhouette PCA

``` r
opt_silhouette_pca<-Optimal_Clusters_KMeans(pca_after$x, max_clusters=10, plot_clusters=TRUE, criterion="silhouette")
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

### Elbow PCA

``` r
opt_elbow_pca<-Optimal_Clusters_KMeans(pca_after$x, max_clusters=10, plot_clusters = TRUE)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

Based on the elbow and silhouette methods, we will choose 2 clusters
again

### Perform K-means on PCA’d dataset

``` r
cluster_pca_kmeans<-eclust(pca_after$x,FUNcluster="kmeans",2)
```

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
fviz_silhouette(cluster_pca_kmeans)
```

    ##   cluster size ave.sil.width
    ## 1       1   67          0.38
    ## 2       2  112          0.28

![](Clustering-and-PCA-Piotr-Bugajski_github_files/figure-gfm/unnamed-chunk-26-2.png)<!-- -->

Although the average silhouette width is higher (0.32) than it’s non-PCA
counterpart (0.23) the low hopkins statistics and high number of PCs
(even 5 PCs explain only 0.75 of the variance) might indicate that PCA
is not the best dimension reduction method for this dataset

## 5.Summary

Clustering performed on this dataset helped with seeing some tendencies,
like the division to poorer/developing countries (cluster 1) and
developed/almost developed/richer countries (cluster 2) although there
is a good tendency shown as well as hopkins statistics being high (0.87)
the average width of the silhouette was low (0.23) which, in my opinion,
might give some guidance but not clear conclusions Dimension reduction
and clustering on that dataset was used to see if the results from this
average width could be improved. While it did improve, hopkins statistic
decreased significantly compared to its non-reduction Counterpart. This
might indicate that while the dataset might be good to reduce
dimensions, PCA might not be the appropriate method.

## 6.Sources

<https://www.datanovia.com/en/lessons/assessing-clustering-tendency/>

<https://sanchitamangale12.medium.com/scree-plot-733ed72c8608>

<https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411>

ChatGPT (converting chr to numeric function and replace commas function)
