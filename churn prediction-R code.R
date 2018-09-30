###### libraries

install.packages("randomForest")
install.packages("party")
install.packages("ggthemes")
install.packages("ROCR")
install.packages("clue")


library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)

library(corrgram)
library(nnet)
library(class)
library(tree)
library(pgmm)
library(penalizedLDA)
library(ROCR)

library (dbscan)
library("HSAUR2")
library(scatterplot3d)
library(mclust)
library(mlbench)
library(fpc)
library(cluster)
library(factoextra)
library(clue)
library(rattle)

######## transform the dataset for prediction

Churn
str(Churn)
colnames((Churn))

churn_predict <-Churn
colnames(churn_predict) <-  c("AccountLength", "VMailMessage" , "DayMins","EveMins","NightMins",   
                             "IntlMins","CustServCalls","Churn","IntlPlan","VMailPlan",    
                            "DayCalls","DayCharge","EveCalls","EveCharge","NightCalls",   
                            "NightCharge","IntlCalls","IntlCharge","State","AreaCode","Gender")

colnames(churn_predict)

churn_predict$Churn <- as.factor(mapvalues(churn_predict$Churn , from=c("0","1"),to=c("No", "Yes")))
churn_predict$IntlPlan <- as.factor(mapvalues(churn_predict$IntlPlan, from=c("0","1"),to=c("No", "Yes")))
churn_predict$VMailPlan <- as.factor(mapvalues(churn_predict$VMailPlan, from=c("0","1"),to=c("No", "Yes")))

churn_predict$Churn <-as.factor(churn_predict$Churn)
churn_predict$Gender <-as.factor(churn_predict$Gender)

churn_predict$AreaCode <- NULL
churn_predict$State <- NULL

churn_predict <-as.data.frame(churn_predict)
churn_predict <-  churn_predict[ , c(1:7 , 9:19 , 8)]

str(churn_predict)

table(churn_predict$Churn)

######  split the dataset into training and testing (80%/20%)
intrain<- createDataPartition(churn_predict$Churn,p=0.8,list=FALSE)
set.seed(2017)
training<- churn_predict[intrain,]
testing<- churn_predict[-intrain,]

###### check the proportions of the training and testing datasets to be aqual to the initial 
table(training$Churn)
table(testing$Churn)

dim(training); dim(testing)  ##Confirm the splitting is correct

#######tree model

treemodel <- tree(Churn ~., data = training)
summary(treemodel)

plot(treemodel)
text(treemodel,pretty=0)

p1 <- predict(treemodel , testing ,type = "class")

cm <- print(table(p1, testing$Churn, 
                  dnn=c("Predicted", "Actual")))


AccuracyTree <- print((cm[2,2]+cm[1,1])/sum(cm) * 100)
Sensitivity <- print(cm[2,2]/(cm[2,2]+cm[1,2])*100)
Specificity <- print(cm[1,1]/(cm[1,1]+cm[2,1])*100)


############prune tree


set.seed(100)
treevalidate <- cv.tree(object = treemodel,FUN = prune.misclass )# prune.misclass prunes the tree based on error rates
treevalidate
plot(x=treevalidate$size, y=treevalidate$dev, type="b")

treemodel2 <- prune.misclass(treemodel, best = 11)
summary(treemodel2)
plot(treemodel2)
text(treemodel2, pretty=0)

p2 <- predict(treemodel2 , testing ,type = "class")

cm2 <- print(table(p2, testing$Churn, 
                   dnn=c("Predicted", "Actual")))


AccuracyTree2 <- print((cm2[2,2]+cm2[1,1])/sum(cm2) * 100)
Sensitivity2 <- print(cm2[2,2]/(cm2[2,2]+cm2[1,2])*100) 
Specificity2 <- print(cm2[1,1]/(cm2[1,1]+cm2[2,1])*100)

### pruning the tree to 11 nodes resulted in an improvement to sensitivity at a slight cost to specificity and model accuracy



######### random forest


rfModel <- randomForest(Churn ~., data = training)
print(rfModel)
###The error rate is relatively low when predicting “No”, and the error rate is much higher when predicting “Yes”.

testing$Churn <- as.factor(mapvalues(testing$Churn , from=c("0","1"),to=c("No", "Yes")))

pred_rf <- predict(rfModel, testing)
caret::confusionMatrix(pred_rf, testing$Churn)


plot(rfModel)

#####tune the rf model



t <- tuneRF(training[, -19], training[, 19], stepFactor = 0.5, plot = TRUE, ntreeTry = 300, trace = TRUE, improve = 0.05)


rfModel_new <- randomForest(Churn ~., data = training, ntree = 300, mtry = 8, importance = TRUE, proximity = TRUE)
print(rfModel_new)

pred_rf_new <- predict(rfModel_new, testing)
caret::confusionMatrix(pred_rf_new, testing$Churn)

########Random Forest Feature Importance
varImpPlot(rfModel_new, sort=T, n.var = 18, main = 'Top  Feature Importance')


####There’s two measures of feature importance that are reported, mean decrease in accuracy, and mean decrease in gini.

###The first is the decrease in accuracy of out of bag samples when the variable feature is excluded from the model.

####The second is the mean decrease in gini. This metric has to do with the decrease in node impurity that results from splits over that variable. 
####The higher the mean decrease in gini, the lower the node impurity. Basically, this means that the lower the node impurity, the more likely the split will produce a left node that is dedicated to one class, and a right node that is dedicated to another class. If the split is totally pure, the left node will be 100% of one class, and the right will be 100% of another class. This is obviously more optimal for making predictions than having two nodes of mixed classes.

### ROC and AUC value
churn.predict.prob <- predict(rfModel_new, testing, type="prob")
pr <- prediction(churn.predict.prob[,2], testing$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc



######logistic regression

str(training)
LogModel <- glm(Churn ~ .,family=binomial(link="logit"),data=training[ , c(-11 , -13 , -15 , -17)])
print(summary(LogModel))
anova(LogModel, test="Chisq")

testing$Churn <- as.character(testing$Churn)
testing$Churn[testing$Churn=="No"] <- "0"
testing$Churn[testing$Churn=="Yes"] <- "1"
fitted.results <- predict(LogModel,newdata=testing,type='response')
fitted.results <- ifelse(fitted.results > 0.5 ,1 ,0)

misClasificError <- mean(fitted.results != testing$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))


print("Confusion Matrix for Logistic Regression"); table(testing$Churn, fitted.results > 0.5 ,dnn=c("Predicted", "Actual"))
exp(cbind(OR=coef(LogModel), confint(LogModel)))

pr <- prediction(fitted.results, testing$Churn)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc



#########clustering

churn_cluster <- churn_predict
str(churn_cluster)

churn_cluster$DayCharge <-NULL
churn_cluster$EveCharge <- NULL
churn_cluster$IntlCharge <- NULL
churn_cluster$NightCharge <- NULL

churn_cluster$IntlPlan <-NULL
churn_cluster$VMailPlan <- NULL
churn_cluster$Gender <-NULL

churn_cluster2 <- churn_cluster[churn_cluster$Churn == 'No',]
str(churn_cluster2)

str(churn_cluster2[ , c(2:7 , 11)])


########## model based clustering
res2 <- Mclust(churn_cluster2[ , c(2:7 , 11)])
summary(res2)

clPairs(churn_cluster2[ , c(2:7 , 11)], cl=res2$classification,symbols = 16, col = c(1:7),
                                                      lower.panel = NULL )
################### dbscan

library (dbscan)

kNNdistplot(churn_cluster2[ , c(2:7 , 11)], k=5)
abline(h = 25, lty = 2)

res <- dbscan(churn_cluster2[ , c(2:7 , 11)], 30, 5)
print(res)

clPairs(churn_cluster2[ , -12], cl=res2$classification,symbols = 16, col = c(1:7),
        lower.panel = NULL )




################################## hierarchical

data2_eucl <- dist(churn_cluster2[ , c(2:7 , 11)], method = 'manhattan')
data2_eucl_m <- as.matrix(data2_eucl)

data2_eucl_ward <- hclust(data2_eucl, method='ward.D')
data2_eucl_single <- hclust(data2_eucl, method='single')
data2_eucl_complete <- hclust(data2_eucl, method='complete')
data2_eucl_average <- hclust(data2_eucl, method='average')
data2_eucl_centroid <- hclust(data2_eucl, method='centroid')

plot(data2_eucl_ward)
rect.hclust(data2_eucl_ward, k = 5, border = "red")

plot(data2_eucl_single )
rect.hclust(data2_eucl_single , k = 5, border = "red")

plot(data2_eucl_complete)
rect.hclust(data2_eucl_complete, k = 5, border = "red")

plot(data2_eucl_average)
rect.hclust(data2_eucl_average, k = 5, border = "red")

plot(data2_eucl_centroid)
rect.hclust(data2_eucl_centroid, k = 5, border = "red")

data2_eucl_ward_groups <- cutree(data2_eucl_ward, k = 4)
data2_eucl_single_groups <- cutree(data2_eucl_single, k = 5)
data2_eucl_complete_groups <- cutree(data2_eucl_complete, k = 5)
data2_eucl_average_groups <- cutree(data2_eucl_average, k = 5)
data2_eucl_centroid_groups <- cutree(data2_eucl_centroid, k = 5)


str(data2_eucl_ward_groups)
as.factor(data2_eucl_ward_groups)

clPairs(churn_cluster2[ , c(2:7 , 11)], cl=data2_eucl_ward_groups,symbols = 16, col = c(1:4),
        lower.panel = NULL )

############################## k means
fit.km <- kmeans(churn_cluster2[ , c(2:7 , 11)], 4, nstart=25)

set.seed(123)
fviz_nbclust(churn_cluster2[ , c(2:7 , 11)], kmeans, method = "wss")
fviz_nbclust(churn_cluster2[ , c(2:7 , 11)], kmeans, method = "silhouette")
fviz_cluster(fit.km, data = churn_cluster2[ , c(2:7 , 11)])

clPairs(churn_cluster2[ , c(2:7 , 11)], cl=fit.km$cluster,symbols = 16, col = c(1:4),
        lower.panel = NULL )

createClusterPlot(fit.km)


table(Churn$Churn)
