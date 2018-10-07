library(ROCR)
library(pROC)

# Set working directory to where csv file is located
setwd("/home/michaeljgrogan/Documents/a_documents/computing/data science/datasets")

# Read the data
mydata<-read.csv("dividendinfo.csv")
attach(mydata)

train<-head(mydata,160)
test<-tail(mydata,40)
attach(train)
attach(test)

# Logistic regression
logreg1 <- glm(dividend ~ fcfps + earnings_growth + de + mcap + current_ratio, data=train, family="binomial")
summary(logreg1)
test$dividend
fcfprobability=logreg1$coefficients[2]
oddsratio=exp(fcfprobability)

probability=(oddsratio/(1+oddsratio))
probability

pred1 <- predict(logreg1, test, type="response")
y_pred1_num <- ifelse(pred1 > 0.5, 1, 0)
y_pred1 <- factor(y_pred1_num, levels=c(0, 1))
plot(roc(test$dividend,y_pred1_num))
rocPred <- roc(test$dividend,y_pred1_num)
auc1 <- auc(rocPred)
print(auc1)