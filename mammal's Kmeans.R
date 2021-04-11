# Describing the data
# The dataset used is part of the package cluster.datasets and contains 25 observations on the following 6 variables:
# name - a character vector for the name of the animals
# water - a numeric vector for the water content in the milk sample
# protein - a numeric vector for the amount of protein in the milk sample
# fat - a numeric vector for the fat content in the milk sample
# lactose - a numeric vector for the amount of lactose in the milk sample
# ash - a numeric vector for the amount of mineral in the milk sample

#install.packages("cluster.datasets")
library('cluster.datasets')
data(all.mammals.milk.1956)
head(all.mammals.milk.1956,10)
dim(all.mammals.milk.1956) # 25  6
df <- all.mammals.milk.1956
str(df)

# The charts below show us the distribution for each variable. 
# Each point represents a mammal species (25 in total).

#install.packages("ggplot2")
library(tidyverse)
library(gridExtra) # to plot multiple graph on same page 

# The color, the size and the shape of points 
# can be changed using the function geom_point() as follow :

# geom_point(size,color,shape)

# Label points in the scatter plot
# The function geom_text() can be used :
# geom_text(label = rownames(dataframe))

data("all.mammals.milk.1956")

plot1 <- all.mammals.milk.1956 %>% 
   ggplot(aes(x = "all mammals", y = water)) + 
   geom_jitter(width = .025, height = 0, size = 2, alpha = .5, color = "blue") +
   labs(x = "", y="percentage of water")
plot2 <-  all.mammals.milk.1956 %>%
  ggplot(aes(x = "all mammals", y = protein)) + 
  geom_jitter(width = .02, height = 0, size = 2, alpha = .6,  color = "orange") +
  labs(x = "", y="percentage of protein")

plot3 <-  all.mammals.milk.1956 %>%
  ggplot(aes(x = "all mammals", y = fat)) + 
  geom_jitter(width = .02, height = 0, size = 2, alpha = .6,  color = "green") +
  labs(x = "", y="percentage of fat")

plot4 <-  all.mammals.milk.1956 %>%
  ggplot(aes(x = "all mammals", y = lactose)) + 
  geom_jitter(width = .02, height = 0, size = 2, alpha = .6,  color = "red") +
  labs(x = "", y="percentage of lactose")

plot5 <-  all.mammals.milk.1956 %>%
  ggplot(aes(x = "all mammals", y = ash)) + 
  geom_jitter(width = .02, height = 0, size = 2, alpha = .6,  color = "violet") +
  labs(x = "", y="percentage of ash")

grid.arrange(plot1, plot2, plot3, plot4, plot5)  

# Choosing a good K
# The bigger is the K you choose, the lower will be the variance within the groups in the clustering. 
# If K is equal to the number of observations, then each point will be a group and the variance will be 0. 
# It's interesting to find a balance between the number of groups and their variance. 
# A variance of a group means how different the members of the group are. 
# The bigger is the variance, the bigger is the dissimilarity in a group.

# we are going to run K-means for an arbitrary K. Let's pick 3.
# As the initial centroids are defined randomly,
# we define a seed for purposes of reprodutability

set.seed(123)
# Let's remove the column with the mammals' names, so it won't be used in the clustering
df1 <- df[,2:6]
str(df1)

# The nstart parameter indicates that we want the algorithm to be executed 20 times.
# R with start with 25 random starting points & find with lowest within cluster variation

km_res <- kmeans(df1,centers = 3, nstart = 20)
# a percentage (89.9%) that represents the compactness of the clustering, that is, 
# show similar are the members within the same group. If all the observations within a group were 
# in the same exact point in the n-dimensional space, then we would achieve 100% of compactness.

# Since we know that, we will use that percentage to help us decide our K value, 
# that is, a number of groups that will have satisfactory variance and compactness.

# Fortunately, this process to compute the "Elbow method" has 
# been wrapped up in a single function (fviz_nbclust):
set.seed(123)
install.packages(c('cluster','factoextra'))
library('cluster')
library('factoextra')


fviz_nbclust(df1, kmeans, method = 'wss')

# By Analysing the chart from right to left, we can see that when the number of 
# groups (K) reduces from 4 to 3 there is a big increase in the sum of squares, bigger than any
# other previous increase. That means that when it passes from 4 to 3 groups there is a reduction 
# in the clustering compactness (by compactness, I mean the similarity within a group). Our goal,
# however, is not to achieve compactness of 100% - for that, we would just take each observation 
# as a group. The main purpose is to find a fair number of groups that could explain satisfactorily
# a considerable part of the data.

# So, let's choose K = 4 and run the K-means again.

km_res1 <- kmeans(df1,centers = 4,nstart = 20)
km_res

# Using 3 groups (K = 3) we had 89.9% of well-grouped data. Using 4 groups (K = 4) that value 
# raised to 95.1%, which is a good value for us.

fviz_cluster(
  km_res,
  data = df1,
  choose.vars = NULL,
  stand = TRUE,
  axes = c(1, 2),
  geom = c("point", "text"),
  repel = TRUE,
  star.plot = TRUE,
  show.clust.cent = TRUE,
  ellipse = TRUE,
  ellipse.type = "convex",
  ellipse.level = 0.95,
  ellipse.alpha = 0.2,
  shape = NULL,
  pointsize = 1.5,
  labelsize = 12,
  main = "Cluster plot",
  xlab = NULL,
  ylab = NULL,
  outlier.color = "black",
  outlier.shape = 19,
  ggtheme = theme_grey()  
  )

fviz_cluster(km_res1,data = df1,
             ellipse.type = 'euclid',
             star.plot = TRUE,
             repel = TRUE,
             ggtheme = theme())

