## ----packages, include=FALSE-----------------------------------------------------------
# If the package is not installed then it will be installed
if(!require("tree")) install.packages("tree")
if(!require("ISLR2")) install.packages("ISLR2")
if(!require("rsample")) install.packages("rsample")
if(!require("rpart.plot")) install.packages("rpart.plot")
if(!require("skimr")) install.packages("skimr")
if(!require("kableExtra")) install.packages("kableExtra")


## --------------------------------------------------------------------------------------
require(ISLR2)
data("Carseats")
help("Carseats")


## ----assignments-----------------------------------------------------------------------
myDescription <- "The data are a simulated data set containing sales of child car seats at different stores [@james2013introduction]"
mydataset <- Carseats


## ----dataDescription-------------------------------------------------------------------
n <- nrow(mydataset)
p <- ncol(mydataset)


## --------------------------------------------------------------------------------------
# as.factor() changes the type of variable to factor
mydataset$High=as.factor(ifelse(mydataset$Sales<=8,"No","Yes"))


## --------------------------------------------------------------------------------------
kable(table(mydataset$High), caption= "Number of observations for each class", col.names = c('High','Freq'))


## --------------------------------------------------------------------------------------
summary(mydataset)


## --------------------------------------------------------------------------------------
skimr::skim(mydataset)


## ----dataPartition---------------------------------------------------------------------
set.seed(2)
pt <- 1/2
train <- sample(1:nrow(mydataset),pt*nrow(mydataset))
mydataset.test <- mydataset[-train,]
High.test <-  mydataset[-train,"High"]


## ----showPartition---------------------------------------------------------------------
kableExtra::kable(table(mydataset[train,"High"]), caption= "Train data: number of observations for each class", col.names = c('High','Freq'))


## ----modelTreeTrain--------------------------------------------------------------------
library(tree)
tree.mydataset=tree(High~.-Sales, mydataset,
                    subset=train, 
                    split="deviance")


## ----summarizeTreeTrain----------------------------------------------------------------
summary(tree.mydataset)
# summary(tree.mydataset2)


## ----plotTree1, fig.cap="Classification tree", fig.height=10, fig.width=12-------------
plot(tree.mydataset)
text(tree.mydataset,pretty=0, cex=0.6)


## --------------------------------------------------------------------------------------
tree.mydataset


## ----TreePerformance-------------------------------------------------------------------
tree.pred=predict(tree.mydataset,mydataset.test,type="class")
res <- table(tree.pred,High.test)
res
accrcy <- sum(diag(res)/sum(res))


## ----prepareForPrunning----------------------------------------------------------------
set.seed(123987)
cv.mydataset=cv.tree(tree.mydataset,FUN=prune.misclass)
names(cv.mydataset)
cv.mydataset


## ----errorRatePlot---------------------------------------------------------------------
par(mfrow=c(1,2))
plot(cv.mydataset$size,cv.mydataset$dev,type="b")
plot(cv.mydataset$k,cv.mydataset$dev,type="b")
par(mfrow=c(1,1))


## ----bestSize--------------------------------------------------------------------------
myBest <- cv.mydataset$size[which.min(cv.mydataset$dev)]


## ----pruneTheTree----------------------------------------------------------------------
prune.mydataset=prune.misclass(tree.mydataset,best=myBest)


## ----plotPrunedTree, fig.cap="The best classification pruned tree", fig.height=10, fig.width=12----
plot(prune.mydataset)
text(prune.mydataset,pretty=0)


## ----testPrunedTree--------------------------------------------------------------------
prunedTree.pred=predict(prune.mydataset,mydataset.test,type="class")
prunedRes <- table(prunedTree.pred,High.test)
prunedRes
prunedAccrcy <- sum(diag(prunedRes)/sum(prunedRes))


## ----prunedTree2, fig.cap="Other classification pruned tree", fig.height=10, fig.width=12----
prune.mydataset=prune.misclass(tree.mydataset, 
                               best = cv.mydataset$size[1])
plot(prune.mydataset)
text(prune.mydataset, pretty=0)


## ----predictPrunedTree2----------------------------------------------------------------
ptree.pred=predict(prune.mydataset, mydataset.test, type="class")
pres <- table(ptree.pred, High.test)
pres
paccrcy <- sum(diag(pres)/sum(pres))


## --------------------------------------------------------------------------------------
require(ISLR2)
data("Carseats")
mydataset <- Carseats


## ----splitTestTrain--------------------------------------------------------------------
# Split the data into training and test sets
set.seed(2)
pt <- 1/2
train <- sample(1:nrow(mydataset), pt * nrow(mydataset))
mydataset.test <- mydataset[-train,]
sales.test <- mydataset$Sales[-train]


## ----fitRegTree1-----------------------------------------------------------------------
# Fit the regression tree using the Sales variable

tree.mydataset <- tree(Sales ~ . , mydataset,
                       subset = train)

# Summary of the fitted regression tree
summary(tree.mydataset)


## ----plotRegTree1----------------------------------------------------------------------
# Plot the regression tree
plot(tree.mydataset)
text(tree.mydataset, pretty = 0, cex = 0.6)


## ----predictTree-----------------------------------------------------------------------
# Predict using the test data
tree.pred <- predict(tree.mydataset, mydataset.test)


## ----checkRegTree----------------------------------------------------------------------
mse1 <- mean((tree.pred - sales.test)^2)
mse1


## ----costComplexityCompute-------------------------------------------------------------
# Prune the regression tree
set.seed(123987)
cv.mydataset <- cv.tree(tree.mydataset, FUN = prune.tree)
names(cv.mydataset)
cv.mydataset


## ----costComplexityPlot----------------------------------------------------------------
# Plot the cross-validation error
par(mfrow = c(1, 2))
plot(cv.mydataset$size, cv.mydataset$dev, type = "b")
plot(cv.mydataset$k, cv.mydataset$dev, type = "b")
par(mfrow = c(1, 1))


## ----costComplexityChoose--------------------------------------------------------------
# Choose the best tree size
myBest <- cv.mydataset$size[which.min(cv.mydataset$dev)]

# Prune the tree with the best size
prune.mydataset <- prune.tree(tree.mydataset, 
                              best = myBest)

# Plot the pruned regression tree
plot(prune.mydataset)
text(prune.mydataset, pretty = 0)

# Predict using the pruned tree
prunedTree.pred <- predict(prune.mydataset, mydataset.test)

# Calculate mean squared error for pruned tree
prunedMSE <- mean((prunedTree.pred - sales.test)^2)
prunedMSE


## ----pruneto5--------------------------------------------------------------------------
# Prune the tree with the best size
pruneto5.mydataset <- prune.tree(tree.mydataset, 
                              best = 6)

# Plot the pruned regression tree
plot(pruneto5.mydataset)
text(pruneto5.mydataset, pretty = 0)

# Predict using the pruned tree
prunedTree5.pred <- predict(pruneto5.mydataset, mydataset.test)

# Calculate mean squared error for pruned tree
prunedMSE5 <- mean((prunedTree5.pred - sales.test)^2)
prunedMSE5


## ----pruneto3--------------------------------------------------------------------------
# Prune the tree with the best size
pruneto3.mydataset <- prune.tree(tree.mydataset, 
                              best = 3)

# Plot the pruned regression tree
plot(pruneto3.mydataset)
text(pruneto3.mydataset, pretty = 0)

# Predict using the pruned tree
prunedTree3.pred <- predict(pruneto3.mydataset, mydataset.test)

# Calculate mean squared error for pruned tree
prunedMSE3 <- mean((prunedTree3.pred - sales.test)^2)
prunedMSE3


## ----loadBostonDat---------------------------------------------------------------------
library(ISLR2)
data("Boston")
datos <- Boston
head(datos, 3)


## ----BostonPlotVars, fig.align='center'------------------------------------------------
color <- adjustcolor("forestgreen", alpha.f = 0.5)
ps <- function(x, y, ...) {  # custom panel function
  panel.smooth(x, y, col = color, col.smooth = "black", 
               cex = 0.7, lwd = 2)
}
nc<- ncol(datos)
pairs(datos[,c(1:6,nc)], cex = 0.7, upper.panel = ps, col = color)
# pairs(datos[,c(7:14)], cex = 0.7, upper.panel = ps, col = color)



## ----BostonSplitTestTrain--------------------------------------------------------------
set.seed(123)
train <- sample(1:nrow(datos), size = nrow(datos)/2)
datos_train <- datos[train,]
datos_test  <- datos[-train,]


## ----BostonFitTree1--------------------------------------------------------------------
set.seed(123)
regTree<- tree::tree(
                    formula = medv ~ .,
                    data    = datos_train,
                    split   = "deviance",
                    mincut  = 20,
                    minsize = 50
                  )
summary(regTree)


## ----BostonPlotTree--------------------------------------------------------------------
par(mar = c(1,1,1,1))
plot(x = regTree, type = "proportional")
text(x = regTree, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")


## ----BostonFitBigTree------------------------------------------------------------------
regTree2<- tree::tree(
                    formula = medv ~ .,
                    data    = datos_train,
                    split   = "deviance",
                    mincut  = 1,
                    minsize = 2,
                    mindev  = 0
                  )


## ----BostonComputeComplexity-----------------------------------------------------------

set.seed(123)
cv_regTree2 <- tree::cv.tree(regTree2, K = 5)


## ----BostonOptimalAlpha----------------------------------------------------------------
optSize <- rev(cv_regTree2$size)[which.min(rev(cv_regTree2$dev))]
paste("Optimal size obtained is:", optSize)


## ----BostonPlotAlphas------------------------------------------------------------------
library(ggplot2)
library(ggpubr)


resultados_cv <- data.frame(
                   n_nodes  = cv_regTree2$size,
                   deviance = cv_regTree2$dev,
                   alpha    = cv_regTree2$k
                 )

p1 <- ggplot(data = resultados_cv, aes(x = n_nodes, y = deviance)) +
      geom_line() + 
      geom_point() +
      geom_vline(xintercept = optSize, color = "red") +
      labs(title = "Error vs tree size") +
      theme_bw() 
  
p2 <- ggplot(data = resultados_cv, aes(x = alpha, y = deviance)) +
      geom_line() + 
      geom_point() +
      labs(title = "Error vs penalization (alpha)") +
      theme_bw() 

ggarrange(p1, p2)


## ----BostonPruneTree-------------------------------------------------------------------
finalTree <- tree::prune.tree(
                  tree = regTree2,
                  best = optSize
               )

par(mar = c(1,1,1,1))
plot(x = finalTree, type = "proportional")
text(x = finalTree, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")


## ----BostonPredictAndCheck1------------------------------------------------------------
predicciones <- predict(regTree, newdata = datos_test)
test_rmse    <- sqrt(mean((predicciones - datos_test$medv)^2))
paste("Error de test (rmse) del árbol inicial:", round(test_rmse,2))


## ----BostonPredictAndCheckFinal--------------------------------------------------------
predicciones_finales <- predict(finalTree, newdata = datos_test)
test_rmse    <- sqrt(mean((predicciones_finales - datos_test$medv)^2))
paste("Error de test (rmse) del árbol final:", round(test_rmse,2))

