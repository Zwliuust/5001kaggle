df=read.csv("/Users/liuziwan/Downloads/all/train.csv")
test_data=read.csv("/Users/liuziwan/Downloads/all/test.csv")
install.packages("caret")
library(caret)
dmy<-dummyVars("~.",data=df)#将类别变量转换为哑变量
dftrsf<-data.frame(predict(dmy,newdata=df))
dftrsf[dftrsf$n_jobs==-1,10]=20#处理n_jobs变量
dmy2<-dummyVars("~.",data=test_data)
test_data<-data.frame(predict(dmy2,newdata=test_data))
test_data[test_data$n_jobs==-1,10]=20
test_labels <- test_data$id
dftrsf$id <- NULL#删除ID
test_data$id <- NULL
test_data$time <- NA
all <- rbind(dftrsf, test_data)
time=all$time
useless_information=all$n_features-all$n_informative
n_cluster=all$n_classes*all$n_clusters_per_class
all_feature=data.frame(all[,1:16],useless_information)#加上新变量：冗余特征个数
all_feature=data.frame(all_feature,n_cluster)#加上新变量：总的cluster
all=data.frame(all_feature,time)
all$random_state<- NULL#删掉相关性低的变量

#矫正和标准化数值变量
numericVarNames=c( "l1_ratio","alpha", "max_iter", "n_jobs", "n_samples", "n_features", 
                   "n_classes", "n_clusters_per_class", "n_informative", "flip_y", "scale","useless_information", "n_cluster")
DFnumeric <- all[, names(all) %in% numericVarNames]
for(i in 1:ncol(DFnumeric)){
  if (abs(skew(DFnumeric[,i]))>0.8){
    DFnumeric[,i] <- log(DFnumeric[,i] +1)
  }
}

PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)
DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)
all$time <- log(all$time)#将time变量取对数
time=all$time
allfeatures=data.frame(all[,1:4], DFnorm)
alldata=data.frame(allfeatures,time)

train_x=alldata[!is.na(alldata$time),1:17]
train_y=alldata[!is.na(alldata$time),18]
test=alldata[is.na(alldata$time),1:17]

#LASSO
set.seed(27042018)
install.packages("glmnet")
library(glmnet)
my_control <-trainControl(method="cv", number=5,selectionFunction ="oneSE")
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0003))
lasso_mod <- train(x=train_x, y=train_y, method='glmnet', metric = "RMSE", trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune
min(lasso_mod$results$RMSE)#0.4045

#XGBoost
install.packages("xgboost")
library(xgboost)
xgb_grid = expand.grid(
  nrounds = 2000,
  eta = c(0.009,0.01,0.011),
  max_depth = c( 2,4,5),
  gamma = c(0,0.01,0.05),
  colsample_bytree=c(0.7,0.75,0.8),
  min_child_weight=c(0.4,0.5,0.55),
  subsample=c( 0.55,0.6, 0.65)
)
xgb_caret <- train(x=train_x, y=train_y, method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
xgb_caret$bestTune
label_train <- train_y

dtrain <- xgb.DMatrix(data = as.matrix(train_x), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test))
#最佳参数
default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0.01,
  colsample_bytree=0.8,
  max_depth=2, #default=6
  min_child_weight=0.2, #default=1
  subsample=0.7
)
#交叉验证选择最佳nrounds
xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 3000, 
                 nfold = 5, showsd = T, stratified = T, print_every_n = 40,
                 early_stopping_rounds = 10, maximize = F)
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 838)#0.259063+0.02527

#XGBoost 预测值
XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred)
#LASSO预测值
LassoPred <- predict(lasso_mod, test)
predictions_lasso <- exp(LassoPred) 

#取加权平均
sub_avg4 <- data.frame(id = test_labels, time = (2*predictions_XGB+predictions_lasso)/3)
write.csv(sub_avg4, file = "/Users/liuziwan/Downloads/5001/individual project/all/avg_pre4.csv")



