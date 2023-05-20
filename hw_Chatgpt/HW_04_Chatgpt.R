install.packages("tidyverse")
library(tidyverse)

df <- read_csv("breast-cancer-wisconsin .csv", na = "?")

colnames(df) <- c('id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
                  'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 
                  'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis')

df$diagnosis <- recode(df$diagnosis, `2` = "benign", `4` = "malignant")

df <- df %>% drop_na()

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

gnb_model<-naiveBayes(X_train,y_train)

y_pred<-predict(gnb_model,X_test)
y_pred <- factor(y_pred, levels = c("benign", "malignant"))
y_test <- factor(y_test, levels = c("benign", "malignant"))

confusionMatrix(y_pred,y_test)

