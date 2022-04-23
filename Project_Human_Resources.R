library(dplyr)

setwd("C:/Users/MI/OneDrive/Documents/Data Science/My Working Directory_R/Data Files")
hr_train <- read.csv("hr_train.csv")
hr_test <- read.csv("hr_test.csv")
View(hr_train) ; View(hr_test)
glimpse(hr_train)

## Data preparation 

hr_test$left = NA # Create Response variable in test Dataset

hr_train$data='train'
hr_test$data='test'

hr <- rbind(hr_train, hr_test) ; View(hr) # Combine test and train datasets
glimpse(hr)
attach(hr)

table(hr_train$left)

# Checking for NA values
for (i in 1:ncol(hr)){
  print(paste(names(hr)[i], " - ", sum(is.na(hr[, i]))))
}

# Dummy variables
sort(table(sales))
hr <- hr %>%
  mutate(Sales_hr = as.numeric(sales == "hr"),
         Sales_accounting = as.numeric(sales == "accounting"),
         Sales_randD = as.numeric(sales == "RandD"),
         Sales_marketing = as.numeric(sales == "marketing"),
         Sales_product_mng = as.numeric(sales == "product_mng"),
         Sales_it = as.numeric(sales == "IT"),
         Sales_support = as.numeric(sales == "support"),
         Sales_technical = as.numeric(sales == "technical"),
         Sales_sales = as.numeric(sales == "sales")) %>%
  select(-sales)

table(salary)
hr <- hr %>%
  mutate(Salary_low = as.numeric(salary == "low"),
         Salary_medium = as.numeric(salary == "medium")) %>%
  select(-salary)
glimpse(hr)

hr_train=hr %>% filter(data=='train') %>% select(-data)
hr_test=hr %>% filter(data=='test') %>% select (-data,-left)

## Model Building

set.seed(10)
s=sample(1:nrow(hr_train),0.8*nrow(hr_train))
hr_train1=hr_train[s,] # training data
hr_train2=hr_train[-s,] # holdout/validation data

###### USING LOGISTIC REGRESSION ######

library(car)

# removing variables based on vif > 5
hr_vif <- lm(left ~ . , hr_train1)
sort(vif(hr_vif), decreasing = T)[1:5]

hr_vif <- lm(left ~ . -  Sales_sales, hr_train1)
sort(vif(hr_vif), decreasing = T)[1:5]

formula(hr_vif)

model1 <- glm(left ~ (satisfaction_level + last_evaluation + number_project + 
                        average_montly_hours + time_spend_company + Work_accident + 
                        promotion_last_5years + Sales_hr + Sales_accounting + Sales_randD + 
                        Sales_marketing + Sales_product_mng + Sales_it + Sales_support + 
                        Sales_technical + Sales_sales + Salary_low + Salary_medium) - 
                Sales_sales, hr_train1, family = "binomial")
summary(model1)

model_step <- step(model1)
summary(model_step)

formula(model_step)

# removing variable based on p-value > 0.05
model_step <- glm(left ~ satisfaction_level + last_evaluation + number_project + 
                    average_montly_hours + time_spend_company + Work_accident + 
                    Sales_randD + Salary_low + Salary_medium, 
                    hr_train1, family = "binomial")
round(sort((summary(model_step)$coefficients)[,4]), 4)
summary(model_step)

# Checking probability score

# for Training data hr_train1
train.score <- predict(model_step, newdata = hr_train1, type = 'response')
train.score[1:5]

# for validation data hr_train2
val.score <- predict(model_step, newdata = hr_train2, type = 'response')
val.score[1:5]

# AUC Curve
library(pROC)
# for training data - 0.724
auc(roc(hr_train1$left, train.score))
# for validation data- 0.7327
auc(roc(hr_train2$left, val.score))

plot(roc(hr_train1$left, train.score))
plot(roc(hr_train2$left, val.score))

plot(roc(hr_train2$left ,val.score),
     col="yellow", lwd=3, main="ROC Curve", 
     asp = NA, legacy.axes = TRUE)

# Predicted class at cut-off 0.4
# for training data
train_pred <- ifelse(train.score > 0.4, 1, 0)

# for validation data
val_pred <- ifelse(val.score > 0.4, 1, 0)
val_pred[15:25]
hr_train2$left[15:25] 

# rough CM
table(hr_train1$left, train_pred)
table(hr_train2$left, val_pred)

# Creating Confusion Matrix
library(caret)

# for Training data
confusionMatrix(factor(train_pred), factor(hr_train1$left))
# Accuracy : 0.7323          
# Sensitivity : 0.8496         
# Specificity : 0.4499 


# for validation data
confusionMatrix(factor(val_pred), factor(hr_train2$left))
# Accuracy : 0.74          
# Sensitivity : 0.8535         
# Specificity : 0.4641

## Create model for entire training data

# removing variables based on vif > 5
hr_vif <- lm(left ~ . - Sales_sales, hr_train)
sort(vif(hr_vif), decreasing = T)[1:5]

formula(hr_vif)

final_model <- glm(left ~ (satisfaction_level + last_evaluation + number_project + 
                             average_montly_hours + time_spend_company + Work_accident + 
                             promotion_last_5years + Sales_hr + Sales_accounting + Sales_randD + 
                             Sales_marketing + Sales_product_mng + Sales_it + Sales_support + 
                             Sales_technical + Sales_sales + Salary_low + Salary_medium) - 
                     Sales_sales, hr_train, family = "binomial")
summary(final_model)

final_model_step <- step(final_model)
summary(final_model_step)

formula(final_model_step)

# removing variable based on p-value > 0.05
final_model_step <- glm(left ~ satisfaction_level + last_evaluation + number_project + 
                          average_montly_hours + time_spend_company + Work_accident + 
                          promotion_last_5years + Salary_low + Salary_medium, 
                          hr_train, family = "binomial")
round(sort((summary(final_model_step)$coefficients)[,4]), 4)

# predicting probabilities

tran.prob <- predict(final_model_step, newdata = hr_train, type = 'response')

auc(roc(hr_train$left, tran.prob))

plot(roc(hr_train$left ,tran.prob),
     col="yellow", lwd=3, main="ROC Curve", 
     asp = NA, legacy.axes = TRUE)

# Predicted class at cut-off 0.4

pred_class <- ifelse(tran.prob > 0.4, 1, 0)
pred_class[150:170]
hr_train$left[150:170] # 6/10 correct 

# final confusion matrix

confusionMatrix(factor(pred_class), factor(hr_train$left))
# Accuracy : 0.7337         
# Sensitivity : 0.8514         
# Specificity : 0.4494 


###### USING DECISION TREE ######

library(tree)
library(cvTools) 

dtree <- tree(left ~ . , hr_train1)
dtree
plot(dtree)
text(dtree)

trn.tree <- predict(dtree, newdata = hr_train1)
trn.tree[1:5]
hr_train1$left[1:5]

val.tree <- predict(dtree, newdata = hr_train2)
val.tree[1:15]
hr_train2$left[1:15]

# AUC Curve
library(pROC)
# for training data - 0.8247
auc(roc(hr_train1$left, trn.tree))
# for validation data- 0.8444
auc(roc(hr_train2$left, val.tree))

plot(roc(hr_train1$left, trn.tree))
plot(roc(hr_train2$left, val.tree))

plot(roc(hr_train2$left ,val.tree),
     col="yellow", lwd=3, main="ROC Curve", 
     asp = NA, legacy.axes = TRUE)

# Predicted class at cut-off 0.4
# for training data
train_pred <- ifelse(trn.tree > 0.5, 1, 0)

# for validation data
val_pred <- ifelse(val.tree > 0.5, 1, 0)
val_pred[15:25]
hr_train2$left[15:25] 

# Creating Confusion Matrix
library(caret)

# for Training data
confusionMatrix(factor(train_pred), factor(hr_train1$left))
# Accuracy : 0.8715          
# Sensitivity : 0.9569         
# Specificity : 0.6659 


# for validation data
confusionMatrix(factor(val_pred), factor(hr_train2$left))
# Accuracy : 0.8767          
# Sensitivity : 0.9496         
# Specificity : 0.6993

## Create decision tree model for entire training data

dtree_final <- tree(left ~ . , hr_train1)

# predicting probabilities

tran.prob <- predict(dtree_final, newdata = hr_train)

auc(roc(hr_train$left, tran.prob)) # 0.8286

plot(roc(hr_train$left ,tran.prob),
     col="yellow", lwd=3, main="ROC Curve", 
     asp = NA, legacy.axes = TRUE)

# Predicted class at cut-off 0.4

pred_class <- ifelse(tran.prob > 0.5, 1, 0)
pred_class[155:175]
hr_train$left[155:175] # 17/20 correct 

pred_class[150:170]
hr_train$left[150:170] # 15/20 correct 

# final confusion matrix

confusionMatrix(factor(pred_class), factor(hr_train$left))
# Accuracy : 0.8726        
# Sensitivity : 0.9554         
# Specificity : 0.6725   


# For final test data

test.prob <- predict(dtree_final, newdata = hr_test)
test_class <- ifelse(test.prob > 0.4, 1, 0)


# For final test data

test.prob <- predict(dtree_final, newdata = hr_test, type = 'response')
test_class <- ifelse(test.prob > 0.4, 1, 0)

table(test_class)

# Storing results in a file

write.csv(test_class,"Asmita_Kangane_P4_Part2.csv",row.names = F)
