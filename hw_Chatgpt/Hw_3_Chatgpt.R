 
library(class)

# Set seed for reproducibility
set.seed(123)
library(tidyverse)

df <- read_csv("breast-cancer-wisconsin .csv", na = "?")

colnames(df) <- c('id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
                  'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
                  'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis')

df$diagnosis <- recode(df$diagnosis, `2` = "benign", `4` = "malignant")

df <- df %>% drop_na()
# Delete rows with missing values

cols <- c('clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
          'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
          'bland_chromatin','normal_nucleoli','mitoses')
df[cols] <- lapply(df[cols], as.factor)

# install.packages("caret")
library(caret)
set.seed(42)
trainIndex <- createDataPartition(df$diagnosis, p = 0.7, list = FALSE)
train_df <- df[trainIndex,]
test_df  <- df[-trainIndex,]

X_train <- train_df %>% select(-diagnosis) %>% data.matrix()
y_train <- train_df$diagnosis

X_test  <- test_df %>% select(-diagnosis) %>% data.matrix()
y_test  <- test_df$diagnosis

preProcValues<- preProcess(X_train)
X_train<- predict(preProcValues,X_train)
X_test<- predict(preProcValues,X_test)
install.packages("e1071")
library(e1071)


# Apply knn methodology with k = 3
knn_pred_3 <- knn(train = X_train, test = X_test, cl = y_train, k = 3)

# Calculate accuracy for k = 3
accuracy <- mean(knn_pred_3 == y_test)
print(accuracy)
knn_pred_3 <- factor(knn_pred_3, levels = c("benign", "malignant"))
y_test <- factor(y_test, levels = c("benign", "malignant"))

# Calculate confusion matrix
cm3 <- confusionMatrix(knn_pred_3, y_test)

# Apply knn methodology with k = 5
knn_pred_5 <- knn(train = X_train, test = X_test, cl = y_train, k = 5)

# Calculate accuracy for k = 5
accuracy <- mean(knn_pred_5 == y_test)
print(accuracy)
knn_pred_5 <- factor(knn_pred_5, levels = c("benign", "malignant"))
y_test <- factor(y_test, levels = c("benign", "malignant"))

# Calculate confusion matrix
cm5 <- confusionMatrix(knn_pred_5, y_test)
print(cm5)

# Apply knn methodology with k = 10
knn_pred_10 <- knn(train = X_train, test = X_test, cl = y_train, k = 10)

# Calculate accuracy for k = 10
accuracy <- mean(knn_pred_10 == y_test)
print(accuracy)
knn_pred_10 <- factor(knn_pred_10, levels = c("benign", "malignant"))
y_test <- factor(y_test, levels = c("benign", "malignant"))

# Calculate confusion matrix
cm10 <- confusionMatrix(knn_pred_10, y_test)
print(cm10)
print(cm5)
print(cm3)



