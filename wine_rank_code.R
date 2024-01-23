# Set working directory and load packages
setwd("/Users/jessicarizzo/Desktop/STSCI 5740/Final Project")
library(MASS)
library(stats)
library(ggplot2)
library(class)
library(leaps)
library(glmnet)
library(tree)
library(randomForest)
library(gbm)
library(BART)

################################# READ IN DATA ####################################

wine <- read.csv("wine-quality-white-and-red.csv")

##################################### EDA  ########################################

# N
length(wine$quality)
table(wine$quality)

# Plot Quality, Figure 1
barplot(table(wine$quality)/length(wine$quality), freq=FALSE, main = "Wine Rankings across the Dataset", xlab = "Wine Quality", ylab = "Probablility")

# Wine Type: get n and % 
table(wine$type)
table(wine$type)/length(wine$type)

# Wine Type: get percentages in each group

## White
white <- subset(wine, wine$type == "white")
table(white$quality)
table(white$quality)/length(white$quality)

## Red
red <- subset(wine, wine$type == "red")
table(red$quality)
table(red$quality)/length(red$quality)

# Identify and remove outliers

plot(wine$residual.sugar)
plot(wine$fixed.acidity)
plot(wine$volatile.acidity)
plot(wine$free.sulfur.dioxide)
plot(wine$total.sulfur.dioxide)
plot(wine$sulphates)
plot(wine$pH)
plot(wine$citric.acid)
plot(wine$chlorides)
plot(wine$density)

## One observation has residual sugar = 65.8, remove:
i <- which(wine$residual.sugar > 65)
wine <- wine[-i,]

j <- which(wine$free.sulfur.dioxide > 280)
wine <- wine[-j,]

# Numerical Predictors: Descriptive Statistics

for(i in 2:12){
  cat(colnames(wine)[i])
  cat("\n")
  data <- wine[,i]
  sum <- c(min(data), mean(data), max(data), sd(data)) 
  cat(sum)
  cat("\n\n\n")
}

# Made wine type an indicator variables where 1 = white and 0 = red

wine$type <- ifelse(wine$type=="white", 1, 0)

write.csv(wine, "wine.csv")

# Set seed to ensure reproducible results
set.seed(1)

# Create training and validation sets in 70-30 split
N <- length(wine$quality)
train_i <- sample(1:N, floor(N*.7), replace=FALSE)
train <- wine[train_i,]
test <- wine[-train_i,]

############################### KNN CLASSIFICATION ################################

# Create vector to hold test errors 
test_error_rates <- rep(NA, 3)
test_error_authors <- rep(NA, 2)

# All variables

# We will use 5-fold cross-validation to determine the optimal k
N_t <- length(train$quality)
error_rate <- rep(NA, 5)
val_i <- sample(1:N_t, N_t, replace=FALSE)
group_start_i <- c(seq(1, N_t, floor(N_t/5)))
group_size <- floor(N_t/5)
f <- 1
for (j in seq(1, 9, 2)){
  #create vector to hold cv errors
  fold_errors <- rep(NA, 5)
  # 5 fold c-v
  for (i in 1:5){
    # Set validation and training sets
    fold_i <- val_i[group_start_i[i]:(group_start_i[i+1]-1)]
    val.X <- train[fold_i,]
    t.X <- train[-fold_i, ]
    val.quality <- train[fold_i, 13]
    t.quality <- train[-fold_i, 13]
    # knn 
    knn.pred <- knn(t.X, val.X, t.quality, k=j)
    # calculate test error
    fold_errors[i] <- mean(knn.pred != val.quality)
  }
  error_rate[f] <- mean(fold_errors)
  f <- f + 1
}
cbind(k=seq(1, 9, 2), error_rate)

par(mfrow=c(1,2))
plot(seq(1, 9, 2), error_rate, type="b", xlab="Neighborhood Size", ylab="Error Rate", main="All")

# We find that the k=1 has the lowest error rate.

# Compute the Test MSE
knn.pred <- knn(train[,-13], test[,-13], train[,13], k=1)
test_error_rates[1] <- mean(knn.pred != test[,13])
test_error_rates[1]

# Confusion Matrix
table(knn.pred, test[,13])

# This model does not perform particularly well. It may be that there are not enough 
# data points in each neighborhood (since p is large), so we will choose three variables 
# and repeat the knn classification

# We will calculate a correlation matrix to determine which variable to use
cor(train)[13,]

# All of the correlations are low, but we will choose the three largest in absolute value:
## Alcohol, Density, Volatile Acidity

error_rate <- rep(NA, 5)
f <- 1
for (j in seq(1, 9, 2)){
  #create vector to hold cv errors
  fold_errors <- rep(NA, 5)
  # 5 fold c-v
  for (i in 1:5){
    # Set validation and training sets
    fold_i <- val_i[group_start_i[i]:(group_start_i[i+1]-1)]
    val.X <- train[fold_i,c(3, 9, 12)]
    t.X <- train[-fold_i, c(3, 9, 12)]
    val.quality <- train[fold_i, 13]
    t.quality <- train[-fold_i, 13]
    # knn 
    knn.pred <- knn(t.X, val.X, t.quality, k=j)
    # calculate test error
    fold_errors[i] <- mean(knn.pred != val.quality)
  }
  error_rate[f] <- mean(fold_errors)
  f <- f + 1
}
cbind(k=seq(1, 9, 2), error_rate)
plot(seq(1, 9, 2), error_rate, type="b", xlab="Neighborhood Size", ylab="Error Rate", main="3 Variables")

# Compute test MSE
knn.pred <- knn(train[,c(3,9,12)], test[,c(3,9,12)], train[,13], k=1)
test_error_rates[2] <- mean(knn.pred != test[,13])
test_error_rates[2]

# Confusion Matrix
table(knn.pred, test[,13])

############################ LINEAR DISCIMINANT ANALYSIS ##############################

# We will see if LDA performs better than KNN
lda.fit <- lda(quality ~ ., data=train)
test_error_rates[3] <- mean(test$quality != predict(lda.fit, test)$class)
test_error_rates[3]
# We find that LDA is on par with KNN classification

# Confusion Matrix
table(predict(lda.fit, test)$class, test$quality)

# Error Rates

data.frame("Method"=c("KNN: All variables", "KNN: 3 Variables", "LDA"), "Test Error"=round(test_error_rates, 2))

########################## MULTIPLE LINEAR REGRESSION ###################################

test_mse <- rep(NA, 7)
## We will use forward selection to choose the best predictors at each level
model.fwd=regsubsets(quality~.,data=train,nvmax=12,method="forward")
summary(model.fwd)

## Now, we will use AIC, BIC and Adjusted R^2 to determine the best model
add_order <- c(12, 3, 8, 11, 5, 1, 7, 9, 6, 2, 11, 4)
num_models <- 12
aic <- rep(NA, num_models)
bic <- rep(NA, num_models)
adj_r <- rep(NA, num_models)
in_model <- c(13)
for(i in 1:num_models){
  in_model <- c(add_order[i], in_model)
  dat <- train[,in_model]
  mod <- lm(quality ~., data=dat)
  summary(mod)
  aic[i] <- AIC(mod)
  bic[i] <- BIC(mod)
  adj_r[i] <- summary(mod)$adj.r.squared
  adj_r[i]
}

cbind(Num_Variables = 1:12, aic, bic, adj_r)

cbind(Num_Variables=c(9,12), aic=aic[c(9,12)], bic=bic[c(9,12)], adj_r=adj_r[c(9,12)])

# We find that the model with 9 variables is the best
dat <- train[,c(add_order[1:9], 13)]
mod9 <- lm(quality ~., data=dat)
summary(mod9)
# Calculate MSE
mod9_predict <- predict(mod9, test[,-13])
test_mse[1] <- mean((test$quality - mod9_predict)^2)
test_mse[1]

############################## RIDGE/LASSO #################################
x <- as.matrix(train[,-13])
y <- train$quality
x_test <- as.matrix(test[,-13])
y_test <- test$quality
# Fit ridge
cv.out=cv.glmnet(x,y,alpha=0) 
bestlam=cv.out$lambda.min
bestlam
ridge.mod=glmnet(x,y,alpha=0,lambda=bestlam)
coef(ridge.mod)[,1]
# Calculate MSE
ridge_pred <- predict(ridge.mod, s = bestlam, newx = x_test)
test_mse[2] <- mean((y_test - ridge_pred)^2)
test_mse[2]
# It performs on par with the best forward selection model

# Fit lasso
cv.out=cv.glmnet(x,y,alpha=1) 
bestlam=cv.out$lambda.min
bestlam
lasso.mod=glmnet(x,y,alpha=1,lambda=bestlam)
coef(lasso.mod)[,1]
# Calculate MSE
lasso_pred <- predict(lasso.mod, s = bestlam, newx = x_test)
test_mse[3] <- mean((y_test - lasso_pred)^2)
test_mse[3]
# There is not much difference between ridge and lasso

############################### REGRESSION TREES ###############################
tree_model <- tree(quality ~ ., train)
summary(tree_model) 

plot(tree_model)
text(tree_model, pretty=0)

preds <- predict(tree_model, newdata=test)
test_mse[3] <-  mean( (preds - test$quality)^2 )
cat("Tree Test MSE: ", mean( (preds - test$quality)^2 ) )

# Does pruning improve the tree?
pruned <- cv.tree(tree_model)
plot(pruned$size, pruned$dev, type = "b", xlab="Size", ylab="Deviance")
prune_wine <- prune.tree(tree_model, best=5)
plot(prune_wine)
text(prune_wine, pretty=0)

# No

# Does random forest improve the tree?
forst_wine <- randomForest(quality ~ ., train, importance=T)
yhat_forst <- predict(forst_wine, newdata = test)

# What are the important variables?
importance(forst_wine)
test_mse[5] <- mean( (yhat_forst - test$quality)^2 ) 
cat('RF tree test mse: ', mean( (yhat_forst - test$quality)^2 ) )

# This is the best model so far

# Does boosting improve the tree?
boost_wine <- gbm(quality ~ ., data = train, distribution = 'gaussian', n.trees = 5000)
yhat_boost <- predict(boost_wine, newdata=test)
test_mse[7] <- mean( (yhat_boost - test$quality)^2 )
cat('boosted tree test mse: ', mean( (yhat_boost - test$quality)^2 ) )
#random forest tree is better

# Does BART improve the tree?
bartfit <- gbart(train[,-13], train[,13], test[,-13])
yhat.bart <- bartfit$yhat.test.mean
test_mse[7] <- mean( (yhat.bart - y_test)^2 )
cat('Bart tree test mse: ', mean( (yhat.bart - y_test)^2 ) )
#random forest tree is better

###################### COMPARISON WITH AUTHORS #############################
test_error_rates <- rep(NA, 2)
red_train <- subset(train, type==0)
red_test <- subset(test, type==0)

white_train <- subset(train, type==1)
white_test <- subset(test, type==1)

# Red
N_red <- length(red_train$quality)
error_rate <- rep(NA, 5)
val_i <- sample(1:N_red, N_red, replace=FALSE)
group_start_i <- c(seq(1, N_red, floor(N_red/5)),N_red)
group_size <- floor(N_red/5)
f <- 1
for (j in seq(1, 9, 2)){
  #create vector to hold cv errors
  fold_errors <- rep(NA, 5)
  # 5 fold c-v
  for (i in 1:5){
    # Set validation and training sets
    fold_i <- val_i[group_start_i[i]:(group_start_i[i+1]-1)]
    val.X <- red_train[fold_i,c(11, 10, 8)]
    t.X <- red_train[-fold_i, c(11, 10, 8)]
    val.quality <- red_train[fold_i, 13]
    t.quality <- red_train[-fold_i, 13]
    # knn 
    knn.pred <- knn(t.X, val.X, t.quality, k=j)
    # calculate test error
    fold_errors[i] <- mean(knn.pred != val.quality)
  }
  error_rate[f] <- mean(fold_errors)
  f <- f + 1
}
cbind(k=seq(1, 9, 2), error_rate)
plot(seq(1, 9, 2), error_rate, type="b", xlab="Neighborhood Size", ylab="Error Rate")

# Compute test MSE
knn.pred <- knn(red_train[,c(11,10,8)], red_test[,c(11,10,8)], red_train[,13], k=1)
test_error_rates[1] <- mean(knn.pred != red_test[,13])
test_error_rates[1]
# Confusion Matrix
table(knn.pred, red_test[,13])

# White
N_white <- length(white_train$quality)
error_rate <- rep(NA, 5)
val_i <- sample(1:N_white, N_white, replace=FALSE)
group_start_i <- c(seq(1, N_white, floor(N_white/5)),N_red)
group_size <- floor(N_white/5)
f <- 1
for (j in seq(1, 9, 2)){
  #create vector to hold cv errors
  fold_errors <- rep(NA, 5)
  # 5 fold c-v
  for (i in 1:5){
    # Set validation and training sets
    fold_i <- val_i[group_start_i[i]:(group_start_i[i+1]-1)]
    val.X <- white_train[fold_i,c(11, 12, 5)]
    t.X <- white_train[-fold_i, c(11, 12, 5)]
    val.quality <- white_train[fold_i, 13]
    t.quality <- white_train[-fold_i, 13]
    # knn 
    knn.pred <- knn(t.X, val.X, t.quality, k=j)
    # calculate test error
    fold_errors[i] <- mean(knn.pred != val.quality)
  }
  error_rate[f] <- mean(fold_errors)
  f <- f + 1
}
cbind(k=seq(1, 9, 2), error_rate)
plot(seq(1, 9, 2), error_rate, type="b", xlab="Neighborhood Size", ylab="Error Rate")

# Compute test MSE
knn.pred <- knn(white_train[,c(11,12,5)], white_test[,c(11,12,5)], white_train[,13], k=1)
test_error_rates[2] <- mean(knn.pred != white_test[,13])
test_error_rates[2]
# Confusion Matrix
table(knn.pred, white_test[,13])

cbind(type=c("Red", "White"), round(test_error_rates, 2))

