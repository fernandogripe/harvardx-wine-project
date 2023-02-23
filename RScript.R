##Fernando Gripe
##HarvardX PH125.9x Data Science: Capstone
##Project: Application and comparison of different Machine Learning methods to discover the quality of red wine
##dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv


##########################################################
# Install and load libraries
# Note: this process could take a couple of minutes
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(reshape)) install.packages("reshape", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(visNetwork)) install.packages("visNetwork", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(ggplot2)
library(reshape)
library(rpart)
library(visNetwork)
library(GGally)
library(class)
library(cvms)


library(caret)        #confusionMatrix
library(MASS)         #Linear Discriminant Analysis
library(randomForest) #random forest
library(nnet)         #multinom - logistic regression
library(xgboost)      #xgboost
library(e1071)        #Naive Bayes Classifier


options(timeout = 120)
set.seed(42)

##########################################################
# Dataset
# Load original dataset and create Train and Test dataset  
##########################################################

#data_loading and processing
df_red = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red',sep=';')

df_red <- df_red %>% mutate(quality.factor = as.factor(df_red$quality))

#spliting into train (80%) and test (20%) set
indx = sample(1:nrow(df_red), size = 0.8 * nrow(df_red))

TrainSet = df_red[indx,]
TestSet = df_red[-indx,]

#checkin the split, if test is 20% of total rows
nrow(TestSet) / ( nrow(TrainSet) + nrow(TestSet) )


##########################################################
# Data Resume
##########################################################

summary(TrainSet, digits = max(3))

as.data.frame(head(TrainSet))

str(TrainSet)

##########################################################
# Data Visualization
##########################################################

#Plot Wines Quality vs quantity
TrainSet%>%
  ggplot(aes(x = quality.factor))+
  geom_bar() +
  geom_text(
    aes(label=after_stat(count)),
    stat='count', vjust=-0.6, size=4)+
  labs(title='Wines Quality vs quantity')

#Plot Correlation between quality vs variables in dataset
TrainSet[,-c(13)] %>%
  cor() %>%
  subset(,  c("quality")) %>%
  melt()%>% 
  filter(X1!="quality") %>%
  ggplot(aes(X2,X1, fill=value))+
  scale_fill_gradient2(low = "orange", mid = "white", high = "blue")+
  geom_tile(color='black')+
  geom_text(aes(label=paste(round(value,2))),size=4, color='black')+
  labs(title='Correlation between quality vs variables in dataset ',
       subtitle = 'Quality is positively correlated with alcohol, sulphates, citric acid and fixed acidity. However, is negatively correlated with volatile acidity' ,x='' ,y='' )


#GG Pairs Plot
ggpairs(TrainSet[,c(1,2,3,10,11,13)] , aes(color = quality.factor, alpha = 0.5) 
        ,upper = list(
          continuous = wrap("cor", size = 3))
        ) + theme(strip.text.x = element_text(size = 13),
                  strip.text.y = element_text(size = 9),
                  axis.text = element_text(size = 10)
        )  +
  labs(title='Dispersion and behavior between main correlated variables vs quality', x='',y='')




#Wine Quality Classification Tree

classification_tree <- rpart(quality.factor~alcohol+volatile.acidity+citric.acid+density+pH+sulphates, TrainSet)
visNetwork::visTree(classification_tree, main = "Wine Quality Classification Tree")


##########################################################
# Model: Naive Bayes
##########################################################

mod.naivebayes <- naiveBayes(quality.factor ~ ., TrainSet[,-c(12)])
confusionMatrix(predict(mod.naivebayes,  TestSet), TestSet$quality.factor)


##########################################################
# Model: KNN
##########################################################


#Finding the number of observations
sqrt(NROW(TrainSet))

#trying both

mod.knn.35 <- knn(train=TrainSet[,-c(12)], test=TestSet[,-c(12)], cl=TrainSet[,c(13)], k=35)
mod.knn.36 <- knn(train=TrainSet[,-c(12)], test=TestSet[,-c(12)], cl=TrainSet[,c(13)], k=36)

confusionMatrix(table(mod.knn.35 ,TestSet$quality.factor))
confusionMatrix(table(mod.knn.36 ,TestSet$quality.factor))


##########################################################
# Model: Multinomial Logistic Regression
##########################################################


mod.logistic.regression = multinom(quality.factor~., TrainSet[,-c(12)])
confusionMatrix(predict(mod.logistic.regression, TestSet),  TestSet$quality.factor)


##########################################################
# Model: LDA
##########################################################


mod.linear.discriminant.analysis  <-lda(quality.factor ~ ., TrainSet[,-c(12)])
confusionMatrix(predict(mod.linear.discriminant.analysis, TestSet)$class, TestSet$quality.factor)

##########################################################
# Model: Random Forest
##########################################################

set.seed(42)
mod.random.forest.default <- randomForest(quality.factor~., TrainSet[,-c(12)])
confusionMatrix(predict(mod.random.forest.default, TestSet), TestSet$quality.factor)

#checking if the default mtry is good enough
#it can take several minutes

control <- trainControl(method='repeatedcv', number=5, repeats=2)
plot(train(quality.factor~., data=TrainSet[,-c(12)], method="rf", tuneLength=15, trControl=control))


#############################################################
# Model: XGBoost
##########################################################
set.seed(42)
xgb_train <- xgb.DMatrix(data = as.matrix(TrainSet[1:11]), label = as.integer(TrainSet$quality) - 3)
xgb_test <- xgb.DMatrix(data = as.matrix(TestSet[1:11]), label = as.integer(TestSet$quality) - 3)
xgb_params <- list(
  max_depth = 5,
  subsample = 0.9,
  eta                = 0.08,              
  gamma              = 0.7,                 
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = length(levels(as.factor(df_red$quality)))
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 300,
  verbose = 0
)

xgb_model


importance_matrix <- xgb.importance(
  feature_names = colnames(xgb_train), 
  model = xgb_model
)

ggplot(data=importance_matrix, aes(x=reorder(Feature, Gain), y= Gain )) +
  geom_bar(stat="identity") + coord_flip() +
  labs(title = "Feature importance", x = "Feauture", y = "Gain")


xgb_preds <- predict(xgb_model, as.matrix(TestSet[1:11]), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- levels(as.factor(df_red$quality))
xgb_preds


xgb_preds$PredictedClass <- apply(xgb_preds, 1, function(y) colnames(xgb_preds)[which.max(y)])
xgb_preds$ActualClass <- levels(as.factor(df_red$quality))[as.integer(TestSet$quality) - 3 + 1]
xgb_preds

head(xgb_preds)

cm <- confusionMatrix(factor(xgb_preds$ActualClass), factor(xgb_preds$PredictedClass, levels = 3:8))

cm


