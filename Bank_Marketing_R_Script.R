# Bank Marketing R Script

# Read the data
# install.packages("readxl")
library(readxl)
bank_data <- readxl::read_excel("C:/BSAN2205/bank_marketing_dataset.xlsx")
View(bank_data)
nrow(bank_data)

# Feature Engineering

# Transform output variable to be binary variable instead of binary category variable
bank_data$response <- as.numeric(bank_data$response=="yes")

# Remove the data with irrelevant values
bank_data <- subset(bank_data, default != "unknown")
bank_data <- subset(bank_data, housing != "unknown")
bank_data <- subset(bank_data, loan != "unknown")
bank_data <- subset(bank_data, job != "unknown")
bank_data <- subset(bank_data, education != "unknown")
bank_data <- subset(bank_data, marital != "unknown")
bank_data <- subset(bank_data, default != "unknown")
bank_data <- subset(bank_data, poutcome != "nonexistent")

nrow(bank_data)

# Dataset is unbalanced
prop.table(table(bank_data$response))

library(ROSE)
# Balance the data with over and under sampling (on the positive and negative cases, respectively)
data_balanced_both <- ovun.sample(response ~ ., data = bank_data, method = "both", N = 41188)$data
table(data_balanced_both$response)
nrow(data_balanced_both)

# Split cleaned dataset
library(caret)
set.seed(987954)
bank_data_sampling_vector <- createDataPartition(data_balanced_both$response, p = 0.80, list = FALSE)
bank_data_train <- data_balanced_both[bank_data_sampling_vector,]
bank_data_test <- data_balanced_both[-bank_data_sampling_vector,]

# Assessing the distribution of the output variable in the training and test sets

bank_data_train_labels <- data_balanced_both$response[bank_data_sampling_vector]
bank_data_test_labels <- data_balanced_both$response[-bank_data_sampling_vector]

# Decision trees

# Specifying, estimating, and plotting the model
# install.packages("tree")
library(tree)

bank_tree <- tree(response ~ ., data = bank_data_train)
summary(bank_tree)
plot(bank_tree)
text(bank_tree, all = T)

library(rpart)
regtree <- rpart(response ~ ., data = bank_data_train)
plot(regtree, uniform = TRUE)
text(regtree, use.n = FALSE, all = TRUE, cex = .8)

# Evaluate the fit of the model to the test data
compute_SSE <- function(correct, predictions) {
  return(sum((correct - predictions) ^ 2))
}

# Assess model fit to the training set

bank_tree_predictions <- predict(regtree, bank_data_train)
(bank_tree_SSE <- compute_SSE(bank_tree_predictions, bank_data_train$response))

# Assess model fit to the test set

bank_tree_test_predictions <- predict(regtree, bank_data_test)
(bank_tree_test_SSE <- compute_SSE(bank_tree_test_predictions, bank_data_test$response))

# Check classification accuracy

# Classification accuracy in training set

bank_data_train$tree_predict <- bank_tree_predictions

bank_data_train$tree_predict_binary[bank_data_train$tree_predict > 0.5] <- 1
bank_data_train$tree_predict_binary[bank_data_train$tree_predict < 0.5] <- 0

table(factor(bank_data_train$response, levels=min(bank_data_train$response):max(bank_data_train$response)), 
      factor(bank_data_train$tree_predict_binary, levels=min(bank_data_train$tree_predict_binary):max(bank_data_train$tree_predict_binary)))

(12845 + 14667) / 32951

# Classification accuracy in the test set

bank_data_test$tree_predict <- bank_tree_test_predictions

bank_data_test$tree_predict_binary[bank_data_test$tree_predict > 0.5] <- 1
bank_data_test$tree_predict_binary[bank_data_test$tree_predict < 0.5] <- 0

table(factor(bank_data_test$response, levels=min(bank_data_test$response):max(bank_data_test$response)), 
      factor(bank_data_test$tree_predict_binary, levels=min(bank_data_test$tree_predict_binary):max(bank_data_test$tree_predict_binary)))

(3261 + 3655) / 8237

# Calculate ROC curve for training and test sets

#install.packages("ROCit")
library(ROCit)

# ROC curve for training set

ROCit_obj_train <- rocit(score = bank_data_train$tree_predict, class = bank_data_train$response)
plot(ROCit_obj_train)

summary(ROCit_obj_train)

# ROC curve for test set

ROCit_obj_test <- rocit(score = bank_data_test$tree_predict, class = bank_data_test$response)
plot(ROCit_obj_test)

summary(ROCit_obj_test)

# Remove columns before training a tree (predicted values)

bank_data_train$tree_predict <- bank_data_train$tree_predict_binary <- NULL

bank_data_test$tree_predict <- bank_data_test$tree_predict_binary <- NULL

# Tune model parameters of the decision tree

# Exploring model parameters
regtree.random <- rpart(response ~ ., data = bank_data_train, 
                        control = rpart.control(minsplit = 20, cp = 0.001, maxdepth = 10))

regtree.random_predictions <- predict(regtree.random, bank_data_test)

(regtree.random_SSE <- compute_SSE(regtree.random_predictions, bank_data_test$response))

# Tuning model parameters
library(e1071)
rpart.ranges <- list(minsplit = seq(5, 50, by = 5), 
                     cp = c(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,0.5),
                     maxdepth = 1:10)
(regtree.tune <- tune(rpart,response ~ ., 
                      data = bank_data_train, ranges = rpart.ranges))

# Training a tree (Expected Run Time: 1 hour and 30 mins)
regtree.tuned <- rpart(response ~ ., data = bank_data_train,  
                       control = rpart.control(minsplit = 5, cp = 0.000, maxdepth = 10))
regtree.tuned_predictions <- predict(regtree.tuned, bank_data_test)
(regtree.tuned_SSE <- compute_SSE(regtree.tuned_predictions, bank_data_test$response))

# Variable importance in tree models
par(las=2)
par(mar=c(5,8,4,2))
barplot(regtree.tuned$variable.importance, main="Variable Importance in Tuned Regression Tree", horiz=TRUE, cex.names=0.8, las=1)

# Add factor to categorical variables
# Reading in and pre-processing the bank dataset

bank_data$job <- as.factor(bank_data$job)
bank_data$job <- relevel(bank_data$job, "unemployed")
bank_data$marital <- as.factor(bank_data$marital)
bank_data$marital <- relevel(bank_data$marital, "single")
bank_data$education <- as.factor(bank_data$education)
bank_data$poutcome <- as.factor(bank_data$poutcome)
bank_data$default <- as.factor(bank_data$default)
bank_data$loan <- as.factor(bank_data$loan)
bank_data$default <- as.factor(bank_data$default)
bank_data$housing <- as.factor(bank_data$housing)

library(ROSE)
# Balance the data with over and under sampling (on the positive and negative cases, respectively)
data_balanced_both <- ovun.sample(response ~ ., data = bank_data, method = "both", N = 41188)$data

# Split cleaned dataset
library(caret)
set.seed(987954)
bank_data_sampling_vector <- createDataPartition(data_balanced_both$response, p = 0.80, list = FALSE)
bank_data_train <- data_balanced_both[bank_data_sampling_vector,]
bank_data_test <- data_balanced_both[-bank_data_sampling_vector,]

# Assessing the distribution of the output variable in the training and test sets

bank_data_train_labels <- data_balanced_both$response[bank_data_sampling_vector]
bank_data_test_labels <- data_balanced_both$response[-bank_data_sampling_vector]

# Check number of rows for train and test data
describe(bank_data_train_labels)
describe(bank_data_test_labels)

# Logistic Regression

# Estimate a binary logistic model on the training set
bank_data_model <- glm(response ~ ., data = bank_data_train, family = binomial)
summary(bank_data_model)

# Transforming the coefficients to odds ratios
exp(coef(bank_data_model))
exp(cbind(OR = coef(bank_data_model), confint(bank_data_model)))

# Assess Model Deviance 

# The log-likelihoods function
log_likelihoods <- function(y_labels, y_probs) {
  y_a <- as.numeric(y_labels)
  y_p <- as.numeric(y_probs)
  y_a * log(y_p) + (1 - y_a) * log(1 - y_p)
}

# The dataset log-likelihood function
dataset_log_likelihood <- function(y_labels, y_probs) {
  sum(log_likelihoods(y_labels, y_probs))
}

# Calculating deviances for each observation in a dataset
deviances <- function(y_labels, y_probs) {
  -2 * log_likelihoods(y_labels, y_probs)
}

# Calculating the deviance for a dataset
dataset_deviance <- function(y_labels, y_probs) {
  sum(deviances(y_labels, y_probs))
}

# Calculating the deviance of a model
model_deviance <- function(model, data, output_column) {
  y_labels = data[[output_column]]
  y_probs = predict(model, newdata = data, type = "response")
  dataset_deviance(y_labels, y_probs)
}

# Calculating the model (residual) deviance
model_deviance(bank_data_model, data = bank_data_train, output_column = 
                 "response")

# Calculating the null deviance
null_deviance <- function(data, output_column) {
  y_labels <- data[[output_column]]
  y_probs <- mean(data[[output_column]])
  dataset_deviance(y_labels, y_probs)
}

null_deviance(data = bank_data_train, output_column = "response")

# Calculating the pseudo-R^2
model_pseudo_r_squared <- function(model, data, output_column) {
  1 - ( model_deviance(model, data, output_column) / 
          null_deviance(data, output_column) )
}

model_pseudo_r_squared(bank_data_model, data = bank_data_train, output_column = "response")

# Classification accuracy and metrics
train_predictions <- predict(bank_data_model, newdata = bank_data_train, type = "response")
train_class_predictions <- as.numeric(train_predictions > 0.5)
mean(train_class_predictions == bank_data_train$response)

test_predictions = predict(bank_data_model, newdata = bank_data_test, type = "response")
test_class_predictions = as.numeric(test_predictions > 0.5)
mean(test_class_predictions == bank_data_test$response)


# Manual calculation of classification accuracy (confusion matrix)
# Training data

bank_data_train$predict <- train_predictions

bank_data_train$predict_binary[bank_data_train$predict > 0.5] <- 1
bank_data_train$predict_binary[bank_data_train$predict < 0.5] <- 0

table(factor(bank_data_train$response, levels=min(bank_data_train$response):max(bank_data_train$response)), 
      factor(bank_data_train$predict_binary, levels=min(bank_data_train$predict_binary):max(bank_data_train$predict_binary)))

(13565 + 13933) / 32951

# Test data

bank_data_test$predict <- test_predictions

bank_data_test$predict_binary[bank_data_test$predict > 0.5] <- 1
bank_data_test$predict_binary[bank_data_test$predict < 0.5] <- 0

table(factor(bank_data_test$response, levels=min(bank_data_test$response):max(bank_data_test$response)), 
      factor(bank_data_test$predict_binary, levels=min(bank_data_test$predict_binary):max(bank_data_test$predict_binary)))

(3380 + 3460) / 8237

# Calculation of ROC curve (Receiver-Operator Characteristic Curve)

#("ROCit")
library(ROCit)

# ROC in training set

ROCit_obj_train <- rocit(score = bank_data_train$predict, class = bank_data_train$response)
plot(ROCit_obj_train)

summary(ROCit_obj_train)

# ROC in test set

ROCit_obj_test <- rocit(score = bank_data_test$predict, class = bank_data_test$response)
plot(ROCit_obj_test)

summary(ROCit_obj_test)


# Binary confusion matrix, precision and recall
(confusion_matrix <- table(predicted = train_class_predictions, actual = bank_data_train$response))
(precision <- confusion_matrix[2, 2] / sum(confusion_matrix[2,]))
(recall <- confusion_matrix[2, 2] / sum(confusion_matrix[,2]))

# Precision-recall curve
library(ROCR)
train_predictions <- predict(bank_data_model, newdata = bank_data_train, type = "response")
pred <- prediction(train_predictions, bank_data_train$response)
perf <- performance(pred, measure = "prec", x.measure = "rec")
plot(perf)



# Remove columns before training a tree (predicted values)

bank_data_train$tree_predict <- bank_data_train$predict_binary <- NULL

bank_data_test$tree_predict <- bank_data_test$predict_binary <- NULL

# Building a bagged predictor for trees

#install.packages("ipred")
library(ipred)

baggedtree <- bagging(response ~ ., data = bank_data_train, nbagg = 300, coob = TRUE, control = rpart.control(minsplit = 5, cp = 0))
baggedtree
baggedtree_predictions <- predict(baggedtree, bank_data_test)

compute_SSE <- function(correct, predictions) {
  return(sum((correct - predictions) ^ 2))
}

(baggedtree_SSE <- compute_SSE(baggedtree_predictions, bank_data_test$response))


# Classification accuracy of bagged tree in the test set

bank_data_test$tree_predict <- baggedtree_predictions

bank_data_test$tree_predict_binary[bank_data_test$tree_predict > 0.5] <- 1
bank_data_test$tree_predict_binary[bank_data_test$tree_predict < 0.5] <- 0

table(factor(bank_data_test$response, levels=min(bank_data_test$response):max(bank_data_test$response)), 
      factor(bank_data_test$tree_predict_binary, levels=min(bank_data_test$tree_predict_binary):max(bank_data_test$tree_predict_binary)))

(4105 + 4122) / 8237

# Accuracy of bagged model

# Draw samples with replacement

M <- 11
seeds <- 70000 : (70000 + M - 1)
n <- nrow(bank_data_train)
sample_vectors <- sapply(seeds, function(x) { set.seed(x); 
  return(sample(n, n, replace = T)) })

# A user-written function for a single logistic regression model

train_1glm <- function(sample_indices) { 
  data <- bank_data_train[sample_indices,]; 
  model <- glm(response ~ ., data = data, family = binomial("logit")); 
  return(model)
}

models <- apply(sample_vectors, 2, train_1glm)

# Evalute the models on the training data

get_1bag <- function(sample_indices) {
  unique_sample <- unique(sample_indices); 
  df <- bank_data_train[unique_sample, ]; 
  df$ID <- unique_sample; 
  return(df)
}

bags <- apply(sample_vectors, 2, get_1bag)

# Create new data frames and predictions

glm_predictions <- function(model, data, model_index) {
  colname <- paste("PREDICTIONS", model_index);
  data[colname] <- as.numeric( 
    predict(model, data, type = "response") > 0.5); 
  return(data[,c("ID", colname), drop = FALSE])
}

training_predictions <- mapply(glm_predictions, models, bags, 1 : M, SIMPLIFY = F)

# Merge the data frames into a single data frame

train_pred_df <- Reduce(function(x, y) merge(x, y, by = "ID",  
                                             all = T), training_predictions)

head(train_pred_df[, 1:5])

# Compute the  training accuracy of the bagged model

train_pred_vote <- apply(train_pred_df[,-1], 1, 
                         function(x) as.numeric(mean(x, na.rm = TRUE) > 0.5))

(training_accuracy <- mean(train_pred_vote == 
                             bank_data_train$response[as.numeric(train_pred_df$ID)]))

# A user-written function for removing observations from the OOB samples

get_1oo_bag <- function(sample_indices) {
  unique_sample <- setdiff(1 : n, unique(sample_indices)); 
  df <- bank_data_train[unique_sample,]; 
  df$ID <- unique_sample; 
  return(df)
}

oo_bags <- apply(sample_vectors, 2, get_1oo_bag)


# Compute the OOB accuracy of the bagged model

oob_predictions <- mapply(glm_predictions, models, oo_bags, 1 : M, SIMPLIFY = F)

oob_pred_df <- Reduce(function(x, y) merge(x, y, by = "ID", all = T), oob_predictions)

oob_pred_vote <- apply(oob_pred_df[,-1], 1, function(x) as.numeric(mean(x, na.rm = TRUE) > 0.5))

(oob_accuracy <- mean(oob_pred_vote == bank_data_train$response[as.numeric(oob_pred_df$ID)], 
                      na.rm = TRUE))

# Compute test set accuracy of the bagged model

get_1test_bag <- function(sample_indices) {
  df <- bank_data_test; 
  df$ID <- row.names(df); 
  return(df)
}

test_bags <- apply(sample_vectors, 2, get_1test_bag)

test_predictions <- mapply(glm_predictions, models, test_bags, 1 : M, SIMPLIFY = F)

test_pred_df <- Reduce(function(x, y) merge(x, y, by = "ID", all = T), test_predictions)

test_pred_vote <- apply(test_pred_df[,-1], 1, function(x) as.numeric(mean(x, na.rm = TRUE) > 0.5))

(test_accuracy <- mean(test_pred_vote == bank_data_test[test_pred_df$ID,"response"], na.rm = TRUE))
