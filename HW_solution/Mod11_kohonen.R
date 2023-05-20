#  First Name      : Khasha
#  Last Name       : Dehnad
#  Id              : 12345
#  purpose         : performing cluster analysis on IRIS dataset

rm(list=ls())
set.seed(123)
data(iris)
View(iris)
#install.packages("kohonen")
library("kohonen")
?som()
#example(som)

training<-iris[,-5]

iris_som<-som(as.matrix(training), grid = somgrid(3,1))


summary(iris_som)
str(iris_som)
iris_som$unit.classif

table(cluster=iris_som$unit.classif,iris[,5])

#plot(iris_som)
 
summary(iris_som)
#map(iris_som,as.matrix(iris[,-5]))


