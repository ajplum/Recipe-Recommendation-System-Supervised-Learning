##clear your working directory
rm(list=ls())


##load libraries
library(tidyverse)
library(ggplot2)
library(readr)
library(jsonlite)
library(caret)
library(gains)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(dplyr)
library(klaR)
library(forecast)


food <- read.csv("/Users/AustinTheBoss/Documents/COSC 6520/food.csv")
recipes <- read_csv("/Users/AustinTheBoss/Documents/COSC 6520/recipe_subset.csv")
vegan <- read.csv("/Users/AustinTheBoss/Documents/COSC 6520/vegan.csv")




#Ensemble Tree first
#Dairy_Nut is target
recipe_titles <- data.frame(recipes$title)
recipes1 <- dplyr::select(recipes, c("Has_beans","Has_beef", "Has_eggs",
                             "Has_pepper","Has_onion", "Has_sugar", "Dairy_Nut"))

#Factorize the Dairy_Nut as well as each ingredient variable
recipes1$Has_beans <- as.factor(recipes1$Has_beans)
recipes1$Has_beef <- as.factor(recipes1$Has_beef)
recipes1$Has_eggs <- as.factor(recipes1$Has_eggs)
recipes1$Has_pepper <- as.factor(recipes1$Has_pepper)
recipes1$Has_onion <- as.factor(recipes1$Has_onion)
recipes1$Has_sugar <- as.factor(recipes1$Has_sugar)
recipes1$Dairy_Nut <- as.factor(recipes1$Dairy_Nut)


set.seed(1)
test_indices <- sample(1:nrow(recipes1), 500)
test_recipes <- recipes1[test_indices,]
recipes1 <- recipes1[-test_indices,]

##partition the data
set.seed(1)
myIndex <- createDataPartition(recipes1$Dairy_Nut, p=0.6, list=FALSE)
trainSet <- recipes1[myIndex,]
validationSet <- recipes1[-myIndex,]


set.seed(1)
randomforest_tree <- randomForest(Dairy_Nut ~., 
                                  data = trainSet, 
                                  ntree = 100, 
                                  mtry = 2, 
                                  importance = TRUE)
varImpPlot(randomforest_tree, type=1)
predicted_class <- predict(randomforest_tree, validationSet)
confusionMatrix(predicted_class, as.factor(validationSet$Dairy_Nut), positive = "1")
predicted_prob <- predict(randomforest_tree, validationSet, type= 'prob')

##convert Dairy_Nut to a numeric
validationSet$Dairy_Nut <- as.numeric(as.character(validationSet$Dairy_Nut))

##create gains table
gains_table <- gains(validationSet$Dairy_Nut, predicted_prob[,2])
gains_table

##create the cumulative lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSet$Dairy_Nut)) ~ c(0, gains_table$cume.obs), 
     xlab = '# of cases', 
     ylab = "Cumulative", 
     type = "l")
lines(c(0, sum(validationSet$Dairy_Nut))~c(0, dim(validationSet)[1]), 
      col="red", 
      lty=2)

##create the decile-wise lift chart
barplot(gains_table$mean.resp/mean(validationSet$Dairy_Nut), 
        names.arg=gains_table$depth, 
        xlab="Percentile", 
        ylab="Lift", 
        ylim=c(0, 3.0), 
        main="Decile-Wise Lift Chart")

##create the receiver operator curve
roc_object <- roc(validationSet$Dairy_Nut, predicted_prob[,2])
plot.roc(roc_object)
auc(roc_object)


##use the algorithm to assign new scores to the scoring data set
test_recipes <- dplyr::select(test_recipes,-c("Dairy_Nut"))
predicted_class_score <- predict(randomforest_tree, test_recipes, type = "class")
predicted_class_score
predicted_class_prob <- predict(randomforest_tree, test_recipes, type = "prob")
predicted_class_prob


### Decile wise lift chart shows that I would need roughly 90% of my data to train the model,
### it's a bit better than random but not by much




# K nearest neighbor to classify similar foods based on nutritional values for recommendations

#Take out 100 foods to use as test data, remove the "Eaten" variable
# Randomly select 5000 recipe indices
set.seed(1)
eaten_indices <- sample(1:nrow(food), 100)

eaten_test <- food[eaten_indices,]
test_names <- data.frame(eaten_test$Food.Name)
eaten_test <- dplyr::select(eaten_test, -c("X","Food.Name","Food.Category","Eaten"))
eaten <- food[-eaten_indices,]
eaten <- dplyr::select(eaten, -c("X","Food.Name","Food.Category"))


Data1<- scale(eaten[1:10])
Data1<- data.frame(Data1, eaten$Eaten)
colnames(Data1)[11] <- 'Eaten'
Data1$Eaten<- as.factor(Data1$Eaten)


set.seed(1)
myFoodIndex<- createDataPartition(Data1$Eaten, p=0.6, list=FALSE)
trainSetEaten <- Data1[myFoodIndex,]
validationSetEaten <- Data1[-myFoodIndex,]


myCtrl <- trainControl(method="cv", number=10)
myGrid <- expand.grid(.k=c(1:10))

set.seed(1)
KNN_fit <- train(Eaten ~ ., data=trainSetEaten, method = "knn", trControl=myCtrl, tuneGrid = myGrid)
KNN_fit

KNN_Class <- predict(KNN_fit, newdata = validationSetEaten)
confusionMatrix(KNN_Class, validationSetEaten$Eaten, positive = '1')




KNN_Class_prob <- predict(KNN_fit, newdata = validationSetEaten, type='prob')
KNN_Class_prob
##this output shows the associated probabilities


##cumulative gain table
##convert Eaten back to numerical (requirement of the gains package)
validationSetEaten$Eaten <- as.numeric(as.character(validationSetEaten$Eaten))
gains_table <- gains(validationSetEaten$Eaten, KNN_Class_prob[,2])
gains_table


##cumulative lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSetEaten$Eaten))~c(0, gains_table$cume.obs), 
     xlab = "# of cases", 
     ylab = "Cumulative", 
     main="Cumulative Lift Chart", 
     type="l")
lines(c(0, sum(validationSetEaten$Eaten))~c(0, dim(validationSetEaten)[1]), 
      col="red", 
      lty=2)

##decile-wise lift
barplot(gains_table$mean.resp/mean(validationSetEaten$Eaten), 
        names.arg=gains_table$depth, 
        xlab="Percentile", 
        ylab="Lift", 
        ylim=c(0,3), 
        main="Decile-Wise Lift Chart")

##ROC
roc_object <- roc(validationSetEaten$Eaten, KNN_Class_prob[,2])
plot.roc(roc_object)
auc(roc_object)

#Test Data

##scale
ScoreData<- scale(eaten_test)

##run the prediction
KNN_Score <- predict(KNN_fit, newdata=ScoreData)

##append the classification results back to the original data set
ScoreData <- data.frame(test_names,eaten_test, KNN_Score)




#Naive Bayes to classify recipes based on complexity

recipes2 <- dplyr::select(recipes, c("title", "Has_beans","Has_beef", "Has_eggs",
                                     "Has_pepper","Has_onion", "Has_sugar", "Dairy_Nut", "Complex"))

# Randomly select 5000 recipe indices
set.seed(1)
selected_indices <- sample(1:nrow(recipes2), 500)


# Extract the selected recipes
unknown_recipes <- recipes2[selected_indices, ]
known_recipes <- recipes2[-selected_indices,]

#To be used later for testing 
known_recipes <- dplyr::select(known_recipes, -c("title"))
unknown_names <- data.frame (unknown_recipes$title)
unknown_recipes <- dplyr::select(unknown_recipes, -c("title", "Complex"))


known_recipes$Complex <- as.factor(known_recipes$Complex)

##partition
set.seed(1)
myIndexRecipe<- createDataPartition(known_recipes$Complex, p=0.6, list=FALSE)
trainSetRecipe <- known_recipes[myIndexRecipe,]
validationSetRecipe <- known_recipes[-myIndexRecipe,]

#k-fold cross validation
myCtrl <- trainControl(method="cv", number=10)

##train the model
set.seed(1)
nb_fit <- train(Complex ~., data = trainSetRecipe, method = "nb", trControl = myCtrl)
nb_fit

##validate the model
nb_class <- predict(nb_fit, newdata = validationSetRecipe)
confusionMatrix(nb_class, validationSetRecipe$Complex, positive = '1')


##create the gains table
nb_class_prob <- predict(nb_fit, newdata = validationSetRecipe, type = 'prob')
validationSetRecipe$Complex <- as.numeric(validationSetRecipe$Complex)
gains_table <- gains(validationSetRecipe$Complex, nb_class_prob[,2])
gains_table

##cumulative lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSetRecipe$Complex)) ~ c(0, gains_table$cume.obs), 
     xlab = '# of cases', 
     ylab = "Cumulative", 
     type = "l")
lines(c(0, sum(validationSetRecipe$Complex))~c(0, dim(validationSetRecipe)[1]), col="red", lty=2)

##decile-wise lift chart
barplot(gains_table$mean.resp/mean(validationSetRecipe$Complex), 
        names.arg=gains_table$depth, 
        xlab="Percentile", 
        ylab="Lift", 
        ylim=c(0,1.5), 
        main="Decile-Wise Lift Chart")

##receiver operator curve
roc_object <- roc(validationSetRecipe$Complex, 
                  nb_class_prob[,2])
plot.roc(roc_object)
auc(roc_object)


#Test on unknown_recipes

##data management
known_recipes$Complex <- as.factor(known_recipes$Complex)
##run the model
nb_class_score <- predict(nb_fit, 
                          newdata=unknown_recipes)
##add the predicted values to the data frame
myScoreData <- data.frame(unknown_names,unknown_recipes, 
                          nb_class_score)



head(myScoreData)





#### Single Classification Tree



##data management

vegan <- dplyr::select(vegan,-c("X.1","Food.Name","X"))

set.seed(1)
test_veganInd <- sample(1:nrow(vegan), 100)
test_vegan <- vegan[test_veganInd ,]
vegan <- vegan[-test_veganInd ,]

vegan$Category <- ifelse(vegan$Category == "Vegan", 1, 0)
vegan$Category <- as.factor(vegan$Category)


##training data set and validation data set
set.seed(1)
myIndexVegan <- createDataPartition(vegan$Category, p=0.7, list=FALSE)
trainSetVegan <- vegan[myIndexVegan,]
validationSetVegan <- vegan[-myIndexVegan,]




##FULL TREE
set.seed(1)
full_tree <- rpart(Category ~ ., 
                   data = trainSetVegan, 
                   method = "class", 
                   cp = 0, 
                   minsplit = 2, 
                   minbucket = 1)
prp(full_tree, 
    type = 1, 
    extra = 1, 
    under = TRUE)


printcp(full_tree)
##smallest xerror  is at tree 7, suggesting tree # 7 is optimal
##x error of 0.048346 and
##xstd of 0.0049007 sum to 0.0532467, no smaller tree with smaller xerror 
##so tree # 7 is both the best-pruned and minimum error tree

##PRUNED TREE
##utilizing information from the FULL TREE output
pruned_tree <- prune(full_tree, cp =  0.00118746)
prp(pruned_tree, 
    type = 1, 
    extra = 1, 
    under = TRUE)

##MODEL PERFORMANCE
##using the validation data set

predicted_class <- predict(pruned_tree, validationSetVegan, type = "class")

##Confusion matrix
confusionMatrix(predicted_class, validationSetVegan$Category, positive = "1")
##accuracy (0.9713), sensitivity (0.9786), specificity (0.9642)

##Examine the probabilities of each validation case belonging to the target
##class instead of its class membership
predicted_prob <- predict(pruned_tree, validationSetVegan, type= 'prob')
head(predicted_prob)



##MODEL PERFORMANCE INDEPENDENT OF CUTOFF
validationSetVegan$Category <- as.numeric(as.character(validationSetVegan$Category))
gains_table <- gains(validationSetVegan$Category, predicted_prob[,2])
gains_table


##LIFT CHART
##cumulative lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSetVegan$Category)) ~ c(0, gains_table$cume.obs), 
     xlab = '# of cases', 
     ylab = "Cumulative", 
     type = "l")
lines(c(0, sum(validationSetVegan$Category))~c(0, dim(validationSetVegan)[1]), col="red", lty=2)


##DECILE-WISE LIFT CHART
barplot(gains_table$mean.resp/mean(validationSetVegan$Category), 
        names.arg=gains_table$depth, 
        xlab="Percentile", 
        ylab="Lift", 
        ylim=c(0, 3.0), 
        main="Decile-Wise Lift Chart")


##RECEIVER OPERATOR CURVE
roc_object <- roc(validationSetVegan$Category, predicted_prob[,2])
plot.roc(roc_object)
auc(roc_object)

##SCORING NEW CASES

head(test_vegan)
predicted_class_score <- predict(pruned_tree, test_vegan, type = "class")
predicted_class_score
##this first output shows the predicted case value 
predicted_class_prob <- predict(pruned_tree, test_vegan, type = "prob")
predicted_class_prob
##the second output shows the associated probabilities
