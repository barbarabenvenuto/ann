Artificial Neural Network - Classification of Stars and Simple Stellar
Populations
================
Bárbara Benvenuto, Paula Coelho

Ah a gente precisa documentar os dados. 

This is an ANN classification algorithm written in R. First of all we implement the
functions we use to compose the network.

Activation functions convert an input signal of a node to an output
signal that can be used as input in the next layer. Here we use the
Sigmoid and Softmax functions.

``` r
# sigmoid activation function: f(x) = 1/(1+exp(-x))
sigmoid = function(x, d = FALSE) {
  if (d == FALSE) {return(1.0/(1+exp(-x)))}
  else {return(x*(1.0-x))} }

# softmax activation function: f(xi) = exp(xi)/sum(exp(xi)), xi the ith component of the vector x
softmax = function(A) {return(exp(A) / rowSums(exp(A)))}
```

The next function loads the dataset and splits it into the three samples
we need: training (60%), validation (20%) and test (20%).

``` r
# in 4363: 2617,873,873   1:2617 2618:3490 3491:4363
load_dataset = function() {
  
  dpath = "~/Downloads/anndataset_n.txt"
  dataset = read.delim(dpath)
  colnames(dataset) = c('F378-R','F395-R','F410-R','F430-R','F515-R','F660-R',
                        'F861-R','G-R','I-R','R-R','U-R','Z-R')
  
  dataset = dataset[sample(nrow(dataset)),]  # sorting randomly
  
  X = matrix(unlist(dataset), ncol = 12)
  Y = row.names(dataset)
  for (i in 1:length(Y)) {Y[i] = ifelse(substr(Y[i],1,1) == "t", 1, 2)}  # 1 star, 2 ssp
  Y = as.integer(Y)
  
  # splitting
  X_train = X[1:2617,]
  Y_train = Y[1:2617]
  X_validation = X[2618:3490,]
  Y_validation = Y[2618:3490]
  X_test = X[3491:4363,]
  Y_test = Y[3491:4363]
  
  samples = list("X_train" = X_train, 
                  "Y_train" = Y_train, 
                  "X_validation" = X_validation, 
                  "Y_validation" = Y_validation, 
                  "X_test" = X_test, 
                  "Y_test" = Y_test)
  return(samples) }
```

The dataset here is composed by 12 features of SSP’s (from line 1 to
636) and stars (the remain).

Given the trained model, this routine predicts the classification of
elements, based on the array of probabilities, which has the
probabilities of each object being a SSP or a star. Then it predicts the
classification taking the higher probability.

``` r
predict = function(model, X) {
  fp = foward_propagation(model, X)
  probs = fp$probs
  colnames(probs) = c(1,2)
  r = as.integer(colnames(probs)[apply(probs,1,which.max)])
  return(r) }
```

This routine starts the Ws (weight matrices) and bias arrays and train
the nn using the training sample and checking its performance on
validation dataset.

``` r
build_train_model = function(ann_model) {
  
  # initialize the weigths (random values) and bias (=0)
  W1 = matrix(rnorm(ann_model$n_input_dim * ann_model$n_hlayer), ann_model$n_input_dim, ann_model$n_hlayer) / sqrt(ann_model$n_input_dim) 
  b1 = matrix(0L,1,ann_model$n_hlayer)
  W2 = matrix(rnorm(ann_model$n_hlayer * ann_model$n_output_dim), ann_model$n_hlayer, ann_model$n_output_dim) / sqrt(ann_model$n_hlayer) 
  b2 = matrix(0L,1,ann_model$n_output_dim)
  
  # define model which will contains Ws and biases
  model = list("W1" = W1, "b1" = b1, "W2" = W2, "b2" = b2)
  
  # loop over the n_passes
  for(i in 1:ann_model$n_passes) {
    
    # foward propagation
    fp = foward_propagation(model, ann_model$X_train)
    probs = fp$probs
    a2 = fp$a2
    
    # backpropagation
    model = back_propagation(ann_model, probs, a2, model)
    
    if(i%%50 == 0) {print(sprintf("Score after iteration %i: %f", i, score(predict(model, ann_model$X_validation), ann_model$Y_validation)))}
  }
  
  return(model) }
```

Let a1 be the array of features. The Foward Propagation function gives
us the output layer signals, which are the classification probabilities.

``` r
foward_propagation = function(model, X){
  # foward propagation
  W1 = model$W1
  b1 = model$b1
  W2 = model$W2
  b2 = model$b2
  
  a1 = X
  z1 = a1 %*% W1
  # adding b1
  for (i in 1:ncol(b1)){
    for (j in 1:nrow(z1)){
      z1[j,i] = z1[j,i] + b1[1,i] } }
  a2 = sigmoid(z1) # hidden layer activation function: sigmoid
  z2 = a2 %*% W2
  # adding b2
  for (i in 1:ncol(b2)){
    for (j in 1:nrow(z2)){
      z2[j,i] = z2[j,i] + b2[1,i] } }
  probs = softmax(z2) # hidden layer activation function: softmax
  
  return(list("probs" = probs, "a2" = a2)) }
```

Then we have the BackPropagation function, with the Gradient Descent
algorithm. It changes the parameters values on a way that minimizes the
loss function, based on its derivatives with respect to the weights and
bias (Chain Rule).

``` r
back_propagation = function(ann_model, probs, a2, model) {
  
  # loading model
  W1 = model$W1
  b1 = model$b1
  W2 = model$W2
  b2 = model$b2
  
  # backpropagating
  
  error = probs
  for (i in 1:ann_model$n_train){
    error[i,Y_train[i]] = error[i,Y_train[i]] - 1 }  # loss function derivative
  delta1 = error %*% t(W2) * sigmoid(a2, d = TRUE)
  
  # weights
  dW2 = t(a2) %*% error
  dW1 = t(ann_model$X_train) %*% delta1
  
  # bias
  db2 = colSums(error)
  db1 = colSums(delta1)
  
  # add regularization terms (b1 and b2 don't have rt)
  
  dW2 = dW2 + ann_model$reg_lambda * W2 
  dW1 = dW1 + ann_model$reg_lambda * W1
  
  # update parameter (gradient descent)
  W1 = W1 + -ann_model$epsilon * dW1 
  b1 = b1 + -ann_model$epsilon * db1 
  W2 = W2 + -ann_model$epsilon * dW2 
  b2 = b2 + -ann_model$epsilon * db2 
  
  # update parameters to the model
  model = list("W1" = W1, "b1" = b1, "W2" = W2, "b2" = b2)
  
  return(model) }
```

The score function calculates the rate of correctly classified objects.

``` r
score = function(class_out, Y) {  # class_out := output (classification)
  count = 0
  for (i in 1:length(Y)) { if (Y[i] == class_out[i]) { count = count + 1} }
  score = count/length(Y)
  return(score) }
```

Now we run the network. In the end, we have the final score, that shows
the network performance on a totally unknown dataset, the test sample.

``` r
samples = load_dataset()  # loading dataset

# training sample
X_train = samples$X_train
Y_train = samples$Y_train
n_train = nrow(X_train)

# validation sample
X_validation = samples$X_validation
Y_validation = samples$Y_validation
n_validation = nrow(X_validation)

# test sample
X_test = samples$X_test 
Y_test = samples$Y_test
n_test = nrow(X_test)

# ann parameter
epsilon = 0.001 # learning rate
reg_lambda = 0.00 # regularization term
n_hlayer = 10 # hidden layer
n_input_dim = ncol(X_train)
n_passes = 1000
n_output_dim = 2 # output

ann_model = list("X_train" = X_train, 
                 "Y_train" = Y_train, 
                 "X_validation" = X_validation, 
                 "Y_validation" = Y_validation, 
                 "X_test" = X_test, 
                 "Y_test" = Y_test,
                 "n_train" = n_train,
                 "n_validation" = n_validation,
                 "n_test" = n_test,
                 "epsilon" = epsilon,
                 "reg_lambda" = reg_lambda,
                 "n_hlayer" = n_hlayer,
                 "n_input_dim" = n_input_dim,
                 "n_passes" = n_passes,
                 "n_output_dim" = n_output_dim)

model = build_train_model(ann_model)  # building and training ANN model
```

    ## [1] "Score after iteration 50: 0.877434"
    ## [1] "Score after iteration 100: 0.879725"
    ## [1] "Score after iteration 150: 0.890034"
    ## [1] "Score after iteration 200: 0.903780"
    ## [1] "Score after iteration 250: 0.906071"
    ## [1] "Score after iteration 300: 0.916380"
    ## [1] "Score after iteration 350: 0.919817"
    ## [1] "Score after iteration 400: 0.935853"
    ## [1] "Score after iteration 450: 0.996564"
    ## [1] "Score after iteration 500: 0.877434"
    ## [1] "Score after iteration 550: 0.997709"
    ## [1] "Score after iteration 600: 0.998855"
    ## [1] "Score after iteration 650: 0.998855"
    ## [1] "Score after iteration 700: 0.998855"
    ## [1] "Score after iteration 750: 0.998855"
    ## [1] "Score after iteration 800: 0.998855"
    ## [1] "Score after iteration 850: 0.998855"
    ## [1] "Score after iteration 900: 0.998855"
    ## [1] "Score after iteration 950: 1.000000"
    ## [1] "Score after iteration 1000: 1.000000"

``` r
score_final = score(predict(model, X_test), Y_test)   
print(sprintf("Final Score: %f", score_final))
```

    ## [1] "Final Score: 1.000000"
