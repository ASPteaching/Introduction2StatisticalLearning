## ----packages, include=FALSE---------------------------------------
# If the package is not installed then it will be installed

if(!require("ISLR")) install.packages("ISLR")
if(!require("rsample")) install.packages("rsample")
if(!require("rpart.plot")) install.packages("rpart.plot")
if(!require("skimr")) install.packages("skimr")
if(!require("kableExtra")) install.packages("kableExtra")



## ----PIMAdescription-----------------------------------------------
library(skimr)
data("PimaIndiansDiabetes2", package = "mlbench")
skim(PimaIndiansDiabetes2)


## ----PIMAbuildTree-------------------------------------------------
library(rpart)
model1 <- rpart(diabetes ~., data = PimaIndiansDiabetes2)
# par(xpd = NA) # otherwise on some devices the text is clipped


## ----PIMATree1-----------------------------------------------------
print(model1)


## ----PIMAPlot1-----------------------------------------------------
plot(model1)
text(model1, digits = 3, cex=0.8)


## ----PIMAPlotNice1-------------------------------------------------
rpart.plot(model1, cex=.7)
detach(package:rpart.plot)


## ----PIMAaccuracy1-------------------------------------------------
predicted.classes<- predict(model1, PimaIndiansDiabetes2, "class")
mean(predicted.classes == PimaIndiansDiabetes2$diabetes)


## ----PIMATestTrain1------------------------------------------------
set.seed(123)
ssize <- nrow(PimaIndiansDiabetes2)
propTrain <- 0.8
training.indices <-sample(1:ssize, floor(ssize*propTrain))
train.data  <- PimaIndiansDiabetes2[training.indices, ]
test.data <- PimaIndiansDiabetes2[-training.indices, ]


## ----PIMAtestTrain2------------------------------------------------
model2 <- rpart(diabetes ~., data = train.data)
predicted.classes.test<- predict(model2, test.data, "class")
mean(predicted.classes.test == test.data$diabetes)

