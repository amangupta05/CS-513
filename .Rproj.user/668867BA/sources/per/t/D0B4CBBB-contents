#Name: Aman Gupta

# To clear all the global environment
rm(list=ls())

# Reading the required file
file<-file.choose()
q6<-read.csv(file, na.strings = "?", colClasses=c("Sample"="character",
                                                  "F1"="factor","F2"="factor","F3"="factor",
                                                  "F4"="factor","F5"="factor","F6"="factor",
                                                  "F7"="factor","F8"="factor","F9"="factor",
                                                  "Class"="factor"))

#install.packages("C50", repos="http://R-Forge.R-project.org")
#install.packages("C50")
library('C50')


dsn <- na.omit(q6)

index<-sort(sample(nrow(dsn),round(.30*nrow(dsn))))
training<-dsn[-index, -1]
test<-dsn[index, -1]

View(dsn)
C50_class <- C5.0( Class~.,data=training )

summary(C50_class )
plot(C50_class)
C50_predict<-predict( C50_class , test , type="class" )
table(actual=test[,10],C50=C50_predict)
wrong<- (test[,10]!=C50_predict)
c50_rate<-sum(wrong)/length(test[,10])
c50_rate

