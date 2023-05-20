
#  First Name      : Khasha
#  Last Name       : Dehnad
#  Id              : 12345
#  purpose         : Accessing extrenal data and replacing missing value  
#                  : accessing data and perofrming EDA

remove(list=ls())

## Step 1 load the data
## changing ? to NA

#1-Load the "breast-cancer-wisconsin.data.csv" from canvas into R and perform the EDA analysis by:

##bc<-read.csv("C://AIMS/Stevens_/2019_S_CS513/Raw_data/breast-cancer-wisconsin.data.csv",
#         na.strings = "?")
file<-file.choose()
bc<-  read.csv(file, na.strings = "?" ) 
#  I.	Summarizing the each column (e.g. min, max, mean )
#II.	Identifying missing values
summary(bc)
is.na(bc)
missing<-bc[is.na(bc$F6),]

rm('mfv')

#III.	Replacing the missing values with the mode of the column.
#install.packages("modeest") this library does not work any longer
library(modeest)
?mfv
bc[is.na(bc$F6),"F6"]<-mfv1(bc$F6,na_rm=TRUE) 
bc[is.na(bc$F6),"F6"]<-5 
# create statistical most frequent value mfv function (mode) function for mode

mfv <- function(x) {
  unique.x <- unique(x)
  tab<-tabulate(match(x, unique.x))
  unique.x[tab == max(tab)]
}

summary(bc)
mfv_value<-mfv1(bc$F6)

bc[is.na(bc$F6),"F6"]<-mfv_value[1] 

#IV.	Displaying the frequency table of Class vs. F6
table(Class=bc$Class,Sixth_Feture=bc$F6)

pairs(bc[c(2:5,11)], main = "Breast Cancer Graph",
      pch = 21, bg = c("red", "green")[factor(bc$Class)])


boxplot(bc[2:5])
boxplot(bc[6:9])
?hist
?paste
clnms<-colnames(bc)
paste("Breast Cancer column=" )
for(i in 2:3){
  hist(bc[[i]],main=paste("Breast Cancer column= ", clnms[i]))
}





#2- Delete all the objects from your environment. Reload the "breast-cancer-wisconsin.data.csv" from canvas into R. Remove any row with missing value in any of the columns.

#bc<-read.csv("C://AIMS/Stevens_/2019_S_CS513/Raw_data/breast-cancer-wisconsin.data.csv",
#             na.strings = "?")
dev.off()
#read the data again
bc2<-na.omit(bc)





