## ----setup, echo = FALSE-------------------------------------------------
knitr::opts_chunk$set(error = TRUE)


## ----chunk1--------------------------------------------------------------
library(tree)


## ----chunk2--------------------------------------------------------------
library(ISLR2)
attach(Carseats)
High <- factor(ifelse(Sales <= 8, "No", "Yes"))


## ----chunk3--------------------------------------------------------------
Carseats <- data.frame(Carseats, High)


## ----chunk4--------------------------------------------------------------
tree.carseats <- tree(High ~ . - Sales, Carseats)


## ----chunk5--------------------------------------------------------------
summary(tree.carseats)


## ----chunk6--------------------------------------------------------------
plot(tree.carseats)
text(tree.carseats, pretty = 0)


## ----chunk7--------------------------------------------------------------
tree.carseats


## ----chunk8--------------------------------------------------------------
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
Carseats.test <- Carseats[-train, ]
High.test <- High[-train]
tree.carseats <- tree(High ~ . - Sales, Carseats,
    subset = train)
tree.pred <- predict(tree.carseats, Carseats.test,
    type = "class")
table(tree.pred, High.test)
(104 + 50) / 200


## ----chunk9--------------------------------------------------------------
set.seed(7)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats


## ----chunk10-------------------------------------------------------------
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")


## ----chunk11-------------------------------------------------------------
prune.carseats <- prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)


## ----chunk12-------------------------------------------------------------
tree.pred <- predict(prune.carseats, Carseats.test,
    type = "class")
table(tree.pred, High.test)
(97 + 58) / 200


## ----chunk13-------------------------------------------------------------
prune.carseats <- prune.misclass(tree.carseats, best = 14)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
tree.pred <- predict(prune.carseats, Carseats.test,
    type = "class")
table(tree.pred, High.test)
(102 + 52) / 200


## ----chunk14-------------------------------------------------------------
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
tree.boston <- tree(medv ~ ., Boston, subset = train)
summary(tree.boston)


## ----chunk15-------------------------------------------------------------
plot(tree.boston)
text(tree.boston, pretty = 0)


## ----chunk16-------------------------------------------------------------
cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = "b")


## ----chunk17-------------------------------------------------------------
prune.boston <- prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)


## ----chunk18-------------------------------------------------------------
yhat <- predict(tree.boston, newdata = Boston[-train, ])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)


## ----chunk19-------------------------------------------------------------
library(randomForest)
set.seed(1)
bag.boston <- randomForest(medv ~ ., data = Boston,
    subset = train, mtry = 12, importance = TRUE)
bag.boston


## ----chunk20-------------------------------------------------------------
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test)^2)


## ----chunk21-------------------------------------------------------------
bag.boston <- randomForest(medv ~ ., data = Boston,
    subset = train, mtry = 12, ntree = 25)
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test)^2)


## ----chunk22-------------------------------------------------------------
set.seed(1)
rf.boston <- randomForest(medv ~ ., data = Boston,
    subset = train, mtry = 6, importance = TRUE)
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)


## ----chunk23-------------------------------------------------------------
importance(rf.boston)


## ----chunk24-------------------------------------------------------------
varImpPlot(rf.boston)


## ----chunk25-------------------------------------------------------------
library(gbm)
set.seed(1)
boost.boston <- gbm(medv ~ ., data = Boston[train, ],
    distribution = "gaussian", n.trees = 5000,
    interaction.depth = 4)


## ----chunk26-------------------------------------------------------------
summary(boost.boston)


## ----chunk27-------------------------------------------------------------
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")


## ----chunk28-------------------------------------------------------------
yhat.boost <- predict(boost.boston,
    newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)


## ----chunk29-------------------------------------------------------------
boost.boston <- gbm(medv ~ ., data = Boston[train, ],
    distribution = "gaussian", n.trees = 5000,
    interaction.depth = 4, shrinkage = 0.2, verbose = F)
yhat.boost <- predict(boost.boston,
    newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)


## ----chunk30-------------------------------------------------------------
library(BART)
x <- Boston[, 1:12]
y <- Boston[, "medv"]
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[-train, ]
ytest <- y[-train]
set.seed(1)
bartfit <- gbart(xtrain, ytrain, x.test = xtest)


## ----chunk31-------------------------------------------------------------
yhat.bart <- bartfit$yhat.test.mean
mean((ytest - yhat.bart)^2)


## ----chunk32-------------------------------------------------------------
ord <- order(bartfit$varcount.mean, decreasing = T)
bartfit$varcount.mean[ord]

