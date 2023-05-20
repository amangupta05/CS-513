library(C50)
library(caret)
# Load the dataset
rm(list=ls())
file<-file.choose()
data <- read.csv(file)

data_clean <- na.omit(data)
data_clean




num_cols <- c("Month_of_absence", "Day_of_the_week", "Social_drinker", "Social_smoker", "Pet")
cat_cols <- c("Trans_expense_cat", "Dist_to_work", "Age_cat", "Abs_cat")

data_clean[, num_cols] <- lapply(data_clean[, num_cols], as.numeric)
data_clean[, cat_cols] <- lapply(data_clean[, cat_cols], as.factor)


set.seed(123)
splitIndex <- createDataPartition(data_clean$Abs_cat, p=0.7, list=FALSE)
train_data <- data_clean[splitIndex,]
test_data <- data_clean[-splitIndex,]

model <- C5.0(Abs_cat ~ ., data = train_data)


predictions <- predict(model, test_data)
confusionMatrix <- confusionMatrix(predictions, test_data$Abs_cat)
accuracy <- confusionMatrix$overall["Accuracy"]


cat("Accuracy: ", accuracy, "\n")

stats_by_class <- as.data.frame(confusionMatrix$byClass)
stats_by_class

precision_abs_high <- stats_by_class["Class: Abs_High", "Pos Pred Value"]

cat("Precision for Abs_cat=Abs_High: ", precision_abs_high, "\n")
