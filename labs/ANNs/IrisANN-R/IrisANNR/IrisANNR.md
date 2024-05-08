<font color='blue'>Cell 1 - Load Iris data set



```python
## Retrieve iris data set

### Upload 'data1.csv' file from the either the drive, or your folder:
# 1) Click the folder icon on the left side of the screen.
# 2) Click the 'upload to session' icon
# 3) locate the 'data1.csv' file and upload it.

iris <- read.csv('data1.csv')
summary(iris)
head(iris)
```


      Sepal_Length    Sepal_Width     Petal_Length    Petal_Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.054   Mean   :3.759   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
       Species         
     Length:150        
     Class :character  
     Mode  :character  
                       
                       
                       



<table class="dataframe">
<caption>A data.frame: 6 × 5</caption>
<thead>
	<tr><th></th><th scope=col>Sepal_Length</th><th scope=col>Sepal_Width</th><th scope=col>Petal_Length</th><th scope=col>Petal_Width</th><th scope=col>Species</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>2</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>3</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>4</th><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>5</th><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>6</th><td>5.4</td><td>3.9</td><td>1.7</td><td>0.4</td><td>Iris-setosa</td></tr>
</tbody>
</table>



<font color='blue'>Cell 2 - Normalize the data


```python
## Normalize the data
# Save a copy of the original data set
iris_org <- iris

# Function that normalizes the iris data set 
normalizeData <- function(data){
        # Uses scale function to normalize data, with mean = 0, and sd = 1
        norm <- data.frame(scale(data[,1:4]))
        # Adds back on the Species column
        new_data <- cbind(norm, data$Species)
        # Renames the 5th column
        names(new_data)[5] <- 'Species'
        # Returns the data
        return(new_data)
} 
# Call upon the normalizeData function
iris <- normalizeData(iris)
# Shows the first 6 rows of the new iris data set.
head(iris)
```


<table class="dataframe">
<caption>A data.frame: 6 × 5</caption>
<thead>
	<tr><th></th><th scope=col>Sepal_Length</th><th scope=col>Sepal_Width</th><th scope=col>Petal_Length</th><th scope=col>Petal_Width</th><th scope=col>Species</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>-0.8976739</td><td> 1.0286113</td><td>-1.336794</td><td>-1.308593</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>2</th><td>-1.1392005</td><td>-0.1245404</td><td>-1.336794</td><td>-1.308593</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>3</th><td>-1.3807271</td><td> 0.3367203</td><td>-1.393470</td><td>-1.308593</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>4</th><td>-1.5014904</td><td> 0.1060900</td><td>-1.280118</td><td>-1.308593</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>5</th><td>-1.0184372</td><td> 1.2592416</td><td>-1.336794</td><td>-1.308593</td><td>Iris-setosa</td></tr>
	<tr><th scope=row>6</th><td>-0.5353840</td><td> 1.9511326</td><td>-1.166767</td><td>-1.046525</td><td>Iris-setosa</td></tr>
</tbody>
</table>



<font color='blue'>Cell 3 - Randomize and split data


```python
# Cell 10 : Split data set

train_percnt <-  2/3 #  2/3  rows , (aka 100 data rows) will be used to train the ANN
train_idx <- sample(nrow(iris), train_percnt * nrow(iris)) # Randomly selects rows

iris_train <- iris[train_idx, ] # selects 100 rows for the training data set
iris_test <- iris[-train_idx, ] # selects the other 50 rows for the test data set

x_train <- iris_train[,1:4] # selects the first 4 columns for the x (input) data
x_test <-  iris_test[,1:4] 

head(x_train)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>Sepal_Length</th><th scope=col>Sepal_Width</th><th scope=col>Petal_Length</th><th scope=col>Petal_Width</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>24</th><td>-0.8976739</td><td> 0.5673506</td><td>-1.1667665</td><td>-0.9154908</td></tr>
	<tr><th scope=row>69</th><td> 0.4307224</td><td>-1.9695830</td><td> 0.4201569</td><td> 0.3948491</td></tr>
	<tr><th scope=row>145</th><td> 1.0345390</td><td> 0.5673506</td><td> 1.1002669</td><td> 1.7051890</td></tr>
	<tr><th scope=row>56</th><td>-0.1730941</td><td>-0.5858010</td><td> 0.4201569</td><td> 0.1327811</td></tr>
	<tr><th scope=row>86</th><td> 0.1891958</td><td> 0.7979809</td><td> 0.4201569</td><td> 0.5258831</td></tr>
	<tr><th scope=row>61</th><td>-1.0184372</td><td>-2.4308437</td><td>-0.1466015</td><td>-0.2603209</td></tr>
</tbody>
</table>



<font color='blue'>Cell 4 : One hot encoding function


```python
# Cell 4 : One hot encoding function

to_one_hot_encode <- function(z){
        y <- z[,5]
        binarized <- matrix(data = NA, nrow=nrow(z), ncol=3)
        count <- 1
        for (x in y){
                if(x == 'Iris-setosa'){
                        binarized[count,] <- c(1,0,0)
                }else if( x == "Iris-versicolor" ){
                        binarized[count,] <- c(0,1,0)
                }else if( x == "Iris-virginica" ){
                        binarized[count,] <- c(0,0,1)
                }else {binarized[count,] <- c(X,X,X)}
                count = count + 1
        }
        return (binarized);
}




y_train <- to_one_hot_encode(iris_train)
y_test <- to_one_hot_encode(iris_test)

dim(y_train)
dim(y_test)

head(y_train)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>100</li><li>3</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>50</li><li>3</li></ol>




<table class="dataframe">
<caption>A matrix: 6 × 3 of type dbl</caption>
<tbody>
	<tr><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>0</td><td>1</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
</tbody>
</table>



<font color='blue'>Cell 5 : Activation sigmoid functions


```python
# Cell 6 : Activation sigmoid functions

sigmoid <- function(x){
        return (1/(1+exp(-x)))
}

sigmoid_deriv <- function(x){
        return (sigmoid(x)*(1-sigmoid(x)))
}

user_softmax <- function(A){
        return (exp(A)/rowSums(exp(A)))
}


```

<font color='blue'>Cell 6: Defining The Training Function for the ANN


```python
training <- function(lr, batch_size, epochs, x_train){
        w0 <-  2*matrix(runif(ncol(x_train)*hidden_size), ncol = hidden_size) -1
        w1 <- 2*matrix(runif(hidden_size*3), ncol = 3) - 1
        
        bh <- runif(hidden_size)
        bo <- runif(3)
        
        num_batch <- floor(nrow(x_train)/batch_size);
        
        errors <- list();
        
        for (epoch in 0:(epochs-1)){
                #####  Feed Forward #######
                for (current_batch in 0:num_batch){
                        # Feed Forward Phase 1
                        batch_start <- current_batch * batch_size;
                        batch_end <- batch_start + batch_size;
                        if (batch_end>nrow(x_train)){
                                batch_end <- nrow(x_train)
                        }
                        input_batch <- as.matrix(x_train[batch_start:batch_end,]);
                        
                        zh <- (input_batch %*% w0) + bh;
                        layer1 <-  sigmoid(zh);
                        
                        # Feed Forward Phase 2
                        zo <- (layer1 %*% w1) + bo;
                        layer2 <- user_softmax(zo);
                        
                        ##### Back Propagation ######
                        
                        # Back propagation phase 1
                        labels_batch <- y_train[batch_start:batch_end,];
                        layer2_error = layer2 - labels_batch;
                        layer2w_delta = (t(layer1) %*% layer2_error);
                        
                        layer2b_delta <- layer2_error
                        
                        # Back porpagation phase 2
                        
                        dcost_dah <- (layer2_error %*% t(w1))
                        dah_dzh <- sigmoid_deriv(zh);
                        
                        layer1_error <- dah_dzh * dcost_dah;
                        layer1w_delta <- (t(input_batch) %*% layer1_error);
                        
                        layer1b_delta <- layer1_error
                        
                        # Update Weights
                        w0 = w0 - (lr*layer1w_delta);
                        bh =  bh - (lr * colSums(layer1b_delta));
                        
                        w1 =  w1 - (lr * layer2w_delta);
                        bo = bo - (lr * colSums(layer2b_delta));
                        
                }
                
                # Update error(s)
                error <-  mean(abs(layer2_error));
                errors = append(errors,error)
        }
        return_list <- list(w0, bh, w1, bo, error, errors);
        return (return_list)
}
        
 


```

<font color='blue'>Cell 7: Evaluation function  


```python
evaluation <- function(params, tst_set){
        w0 <-  params[[1]]
        bh <-  params[[2]]
        w1 <-  params[[3]]
        bo <-  params[[4]]
        tst_set <- as.matrix(tst_set)
        # Phase 1
        zh <- (tst_set %*% w0) + bh
        layer1 <-  sigmoid(zh);
        
        # Phase 2 layer2 is final output
        zo <- (layer1 %*% w1) +bo;
        layer2 <- user_softmax(zo);
        
        return (layer2);
}
```

<font color='blue'>Cell 8: Select Parameters and train the data set
  


```python
# Set parameters
learning_rate <- 0.01
hidden_size <-  5
batch_size = 10
epochs = 500

# Call upon the training function
trained_data <- training(learning_rate, batch_size, epochs, x_train)  

# break returned list into different parameters
trained_params <- trained_data[1:4] # training parameters
error <- trained_data[[5]] 
errors <- trained_data[[6]]
```

<font color='blue'>Cell 9: plotting the accuracy chart
  


```python
# Cell 14 : plotting the accuracy chart

one_to_100 <- c(1:length(errors))
accuracy <- signif(((1-error)*100),4)
accuracy <- paste(accuracy,"%")
plot(one_to_100, errors ,main="Training Error", ylab = "Error percentage",
     xlab = "Number of Runs", type = "l", col="blue")

cat("Training Accuracy: ",accuracy)
```

    Training Accuracy:  99.73 %


    
![png](output_17_1.png)
    


<font color='blue'>Cell 10: Evalutate and Create a Confusion Matrix
  


```python
# Cell 15 : Run the evaluation function 
# Using the trained parameters, the 'evaluate' function outputs a list of the most likely flowers
# We then compare the results with the actual results(y_test) in the form of a Confustion Matrix
prediction <- evaluation(trained_params,x_test) 

# Confusion matrix 

# Applying the which.max function across each row of the predicted values.
predicted_values <- apply(prediction, 1, which.max)
# Doing the same with the y_test dataset
observed_values <- apply(y_test, 1, which.max)
# Create the table
table <- table(predicted_values, observed_values)
rownames(table) <- c('B','C','H')
colnames(table) <- c('B','C','H') 

table
# Function to normalize the confusion matrix
normalizeCM <- function(cm){
        cm_norm <- cm
        for ( i in 1:nrow(cm)){
                for ( j in 1:ncol(cm)){
                        cm_norm[i,j] <- cm[i,j]/sum(cm[i,])
                }
        }
        return(round(cm_norm, 2))
}
cat("\n")
cat("Normalized Confusion Matrix: ")
cat("\n")
normalizeCM(table)
```


                    observed_values
    predicted_values  B  C  H
                   B 14  0  0
                   C  3  8  9
                   H  0  6 10


    
    Normalized Confusion Matrix: 
    


                    observed_values
    predicted_values    B    C    H
                   B 1.00 0.00 0.00
                   C 0.15 0.40 0.45
                   H 0.00 0.38 0.62




<font color='blue'>Cell 8


```python
# Function to Normalize new input data
normalizeNewValues <- function(features){
        norm_features <- data.frame('Sepal_Length'= 0, 'Sepal_Width'= 0,
                                       'Petal_Length'= 0, 'Petal_Width'= 0)
        for (i in 1:4){
                x <- ((features[i]-mean(iris_org[,i]))/sd(iris_org[,i]))
                norm_features[1,i] <- x
                
        } 
        return(norm_features)
}
```

<font color='blue'>Cell 9- Test the Neural Network - Define the InputValue function


```python
## Testing the Neural Network with user inputted values
# The below function inputValue asks the user to enter a numeric value for a certain feature
inputValue <- function(feature_name){
  # creates message to prompt user to input the respective value
  message <- paste("Enter ", feature_name, ": ")
  while(TRUE){
    # If below zero or above 10, prints out error message, and prompts the user once more 
    # to input a value for the respective feature
    value = as.double(readline(prompt = message))
    if(value<0 | value>10){
      warning("Invalid entry, please try again")
    } else {
      
      return(value)
    }
  } 
}
```

<font color='blue'>Cell 10- Define the predictSpecies function


```python
# Create a species level vector to transpose numeric values of 1,2,3 into 
# 'iris-setosa', 'iris-versicolor', and 'iris-virginica', respectively
species_levels <- levels(factor(iris$Species))

## Function to predict species 
predictSpecies <- function(measurements){
        # Call upon the  normalizeNewValues function
        norm_measurements <- normalizeNewValues(measurements)
        # Using the evaluation function and our created neural network, this creates prediction 
        # probabilities for the 3 possible outcomes 
        prediction_prob <- evaluation(trained_params, norm_measurements) # run values through the evaluation program
        # Select the most probably species. 
        prediction <- which.max(prediction_prob)
        # Change prediction from numeric to character using species_levels
        prediction <- species_levels[prediction]
        # Inform user of calculated prediction
        cat("That flower is most likely: ", prediction)
}


```

<font color='blue'>Cell 11


```python
# Call upon the inputValue function four times to obtain the four input values
print('Measurements need to be numeric values between 0 and 10')
sepal_length <-  inputValue('Sepal Length')
sepal_width <-  inputValue('Sepal Width')
petal_length <- inputValue('Petal Length')
petal_width <- inputValue('Petal Width')

# create a list with the four given measurements
features = c(sepal_length, sepal_width, petal_length, petal_width)

# Call upon the predictSpecies function 
predictSpecies(features)
```

    [1] "Measurements need to be numeric values between 0 and 10"
    Enter  Sepal Length : 5.1
    Enter  Sepal Width : 3.5
    Enter  Petal Length : 1.4
    Enter  Petal Width : .2
    That flower is most likely:  Iris-setosa
