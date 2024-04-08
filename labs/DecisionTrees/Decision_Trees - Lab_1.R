
# If the package is not installed then it will be installed

if(!require("ISLR")) install.packages("ISLR")
if(!require("rsample")) install.packages("rsample")
if(!require("rpart.plot")) install.packages("rpart.plot")
if(!require("skimr")) install.packages("skimr")
if(!require("kableExtra")) install.packages("kableExtra")




require(ISLR2)
data("Carseats")
help("Carseats")



myDescription <- "The data are a simulated data set containing sales of child car seats at different stores [@james2013introduction]"
mydataset <- Carseats



n <- nrow(mydataset)
p <- ncol(mydataset)



# as.factor() changes the type of variable to factor
mydataset$High=as.factor(ifelse(mydataset$Sales<=8,"No","Yes"))



kable(table(mydataset$High), caption= "Number of observations for each class", col.names = c('High','Freq'))



summary(mydataset)



skimr::skim(mydataset)



set.seed(2)
pt <- 1/2
train <- sample(1:nrow(mydataset),pt*nrow(mydataset))
mydataset.test <- mydataset[-train,]
High.test <-  mydataset[-train,"High"]



kableExtra::kable(table(mydataset[train,"High"]), caption= "Train data: number of observations for each class", col.names = c('High','Freq'))



library(tree)
tree.mydataset=tree(High~.-Sales, mydataset,
                    subset=train, 
                    split="deviance")



summary(tree.mydataset)
# summary(tree.mydataset2)



plot(tree.mydataset)
text(tree.mydataset,pretty=0, cex=0.6)



tree.mydataset



tree.pred=predict(tree.mydataset,mydataset.test,type="class")
res <- table(tree.pred,High.test)
res
accrcy <- sum(diag(res)/sum(res))



set.seed(123987)
cv.mydataset=cv.tree(tree.mydataset,FUN=prune.misclass)
names(cv.mydataset)
cv.mydataset



par(mfrow=c(1,2))
plot(cv.mydataset$size,cv.mydataset$dev,type="b")
plot(cv.mydataset$k,cv.mydataset$dev,type="b")
par(mfrow=c(1,1))



myBest <- cv.mydataset$size[which.min(cv.mydataset$dev)]



prune.mydataset=prune.misclass(tree.mydataset,best=myBest)



plot(prune.mydataset)
text(prune.mydataset,pretty=0)



prunedTree.pred=predict(prune.mydataset,mydataset.test,type="class")
prunedRes <- table(prunedTree.pred,High.test)
prunedRes
prunedAccrcy <- sum(diag(prunedRes)/sum(prunedRes))



prune.mydataset=prune.misclass(tree.mydataset, 
                               best = cv.mydataset$size[1])
plot(prune.mydataset)
text(prune.mydataset, pretty=0)



ptree.pred=predict(prune.mydataset, mydataset.test, type="class")
pres <- table(ptree.pred, High.test)
pres
paccrcy <- sum(diag(pres)/sum(pres))



tree.mydataset2 <- tree(Sales~.-High, mydataset,
                    subset=train, 
                    split="deviance")
summary(tree.mydataset2)



library(MASS)
data("Boston")
datos <- Boston
head(datos, 3)



color <- adjustcolor("forestgreen", alpha.f = 0.5)
ps <- function(x, y, ...) {  # custom panel function
  panel.smooth(x, y, col = color, col.smooth = "black", 
               cex = 0.7, lwd = 2)
}
pairs(datos[,c(1:6,14)], cex = 0.7, upper.panel = ps, col = color)
pairs(datos[,c(7:14)], cex = 0.7, upper.panel = ps, col = color)




set.seed(123)
train <- sample(1:nrow(datos), size = nrow(datos)/2)
datos_train <- datos[train,]
datos_test  <- datos[-train,]



set.seed(123)
arbol_regresion <- tree::tree(
                    formula = medv ~ .,
                    data    = datos_train,
                    split   = "deviance",
                    mincut  = 20,
                    minsize = 50
                  )
summary(arbol_regresion)



par(mar = c(1,1,1,1))
plot(x = arbol_regresion, type = "proportional")
text(x = arbol_regresion, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")



arbol_regresion <- tree::tree(
                    formula = medv ~ .,
                    data    = datos_train,
                    split   = "deviance",
                    mincut  = 1,
                    minsize = 2,
                    mindev  = 0
                  )

# Optimization
set.seed(123)
cv_arbol <- tree::cv.tree(arbol_regresion, K = 5)



size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
paste("Optimal size obtained is:", size_optimo)



library(ggplot2)
library(ggpubr)


resultados_cv <- data.frame(
                   n_nodes  = cv_arbol$size,
                   deviance = cv_arbol$dev,
                   alpha    = cv_arbol$k
                 )

p1 <- ggplot(data = resultados_cv, aes(x = n_nodes, y = deviance)) +
      geom_line() + 
      geom_point() +
      geom_vline(xintercept = size_optimo, color = "red") +
      labs(title = "Error vs tree size") +
      theme_bw() 
  
p2 <- ggplot(data = resultados_cv, aes(x = alpha, y = deviance)) +
      geom_line() + 
      geom_point() +
      labs(title = "Error vs penalization (alpha)") +
      theme_bw() 

ggarrange(p1, p2)



arbol_final <- tree::prune.tree(
                  tree = arbol_regresion,
                  best = size_optimo
               )

par(mar = c(1,1,1,1))
plot(x = arbol_final, type = "proportional")
text(x = arbol_final, splits = TRUE, pretty = 0, cex = 0.8, col = "firebrick")



predicciones <- predict(arbol_regresion, newdata = datos_test)
test_rmse    <- sqrt(mean((predicciones - datos_test$medv)^2))
paste("Error de test (rmse) del árbol inicial:", round(test_rmse,2))



predicciones_finales <- predict(arbol_final, newdata = datos_test)
test_rmse    <- sqrt(mean((predicciones - datos_test$medv)^2))
paste("Error de test (rmse) del árbol final:", round(test_rmse,2))



if(!require(AmesHousing))
  install.packages("AmesHousing", dep=TRUE)
ames <- AmesHousing::make_ames()



if(!require(rsample))
  install.packages("rsample", dep=TRUE)
# Stratified sampling with the rsample package
set.seed(123)
split <- rsample::initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)



require(rpart)
ames_dt1 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train
  # method  = "anova"
)



require(rpart.plot)
rpart.plot(ames_dt1, cex=0.5)



printcp(ames_dt1)



plotcp(ames_dt1)



if(!require(vip))
  install.packages("vip", dep=TRUE)
require(vip)
## ? vip
vip(ames_dt1, num_features = 40, bar = FALSE)

