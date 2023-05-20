# Load required libraries
library(class)

# Set seed for reproducibility
set.seed(123)

# Load data
data <- read.csv("breast-cancer-wisconsin .csv")

# Delete rows with missing values
data <- na.omit(data)

# Split data into training and test sets (70% training, 30% test)
train_index <- sample(1:nrow(data), 0.7*nrow(data))
train_data <- data[train_index,]
test_data <- data[-train_index,]

# Define predictor variables and outcome variable
predictors <- c("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")
outcome <- c("Diagnosis")

# Convert outcome variable to factor data type
train_data[,outcome] <- as.factor(train_data[,outcome])
test_data[,outcome] <- as.factor(test_data[,outcome])

# Apply knn methodology with k = 3
knn_pred_3 <- knn(train = train_data[,predictors], test = test_data[,predictors], cl = train_data[,outcome], k = 3)

# Calculate accuracy for k = 3
mean(knn_pred_3 == test_data[,outcome])