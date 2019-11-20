
# for training data
setwd('/Users/sayaneshome/20newsgroups/') #set the pathname with all files here
install.packages('tidyverse')
library(tidyverse)
df <- read.csv('train_data.csv', header = T)
df1 <- read.csv('train_label.csv',header = T)
names(df) <- c("d", "w","w_d")
names(df1) <- c("n")
df1$d <- seq.int(nrow(df1))

# setwd('/Users/sayaneshome/20newsgroups/')
# df <- read.table(text =
#                    "d w w_d
#                  1 2 2
#                  2 3 1
#                  3 4 10
#                  4 5 11
#                  6 10 12
#                  8 6 7
#                  9 7 9
#                  11 3 8 ", header = T)
# 
# df1 <- read.table(text =
#                    "n d
#                 1 1
#                 2 2
#                 3 11
#                 4 9
#                 5 8
#                 6 6
#                 7 4
#                 1 3 ", header = T)

v1 <- merge(df,df1,by="d",all = TRUE)
library(tidyverse)
c1 <- v1 %>%
  group_by(d) %>%
  mutate(tw_d = sum(w_d))

#classifier_2 <- merge(classifier_1,train_data_label,by = 'docIdx',all=TRUE)
#classifier_2[ , colSums(is.na(classifier_2)) == 0]

c2 <- c1 %>%
  group_by(w,n) %>%
  mutate(w_n = sum(w_d)) # %>%
#summarise(newsgroup_ID,w_n)
#till here

c3 <- c2 %>%
  group_by(n) %>%
  mutate(tw_n = sum(w_n))

vocab <- read.table('vocabulary.txt',header = FALSE)
vocab$ID <- seq.int(nrow(vocab))
unique(vocab)
vc <- nrow(unique(vocab))

c3$MLE <- c3$w_n/c3$tw_n
c3$Bayesianestimator <- (c3$w_n+1)/(vc+c3$tw_n)
docs <- length(unique(c3$d))
p1 <- c3 %>%
  group_by(n) %>%
  mutate(nd = length(unique(d))) 
p1$p_wj <- p1$nd/docs
p1 <- select(p1,n,p_wj)
p1 <- unique(p1)
p1 <- head(p1,-1)
print('Class Priors:')
view(p1)
write.table(p1,'priors.txt')

#n1 <- c3 %>%
#  group_by(n) %>%
#  filter(n,tw_n)
#n1 <- unique(n1)

#merger <- merge(n1,c2,by = "n",all=TRUE)

w1 <- c3 %>%
  ungroup() %>%
  expand(w, n)

w2 <- c3 %>%
  select(w,n,w_n,tw_n,d)

merger <- merge(w1,w2,by=c('w','n'),all = TRUE)
merger[is.na(merger)] <- 0
merger <- merger %>%
  group_by(n) %>%
  mutate(tw_n = sum(w_n))
merger <- unique(merger)
merger <- merger[!merger$d == "0", ]

merger$MLE <- log(merger$w_n/merger$tw_n)
merger$Bayesianestimator <- log((merger$w_n+1)/(vc+merger$tw_n))
merger <- merge(merger,p1)
merger$p_wj_log <- log(merger$p_wj)

#for MLE
#sumMLE<-c3 %>%  group_by(d) %>% summarise(sumMLE = sum(log(MLE)))
sum_E<-merger %>%  group_by(d) %>% 
  summarise(sumMLE = sum((MLE)),sumBE = sum(Bayesianestimator))
dd <- merge(merger,sum_E,by='d')
dd$Omega_NB<-dd$sumBE+dd$p_wj_log
dd$Omega_MLE<-dd$sumMLE+dd$p_wj_log
maxMLE <- dd %>% group_by(d) %>% filter(Omega_MLE == max(Omega_MLE))
maxBayesianb <- dd %>% group_by(d) %>% filter(Omega_NB == max(Omega_NB))

results_NB <- maxBayesianb %>%
  group_by(d,n) %>%
  select(Omega_NB)
results_NB <- unique(results_NB)

results_MLE <- maxMLE %>%
  group_by(d,n) %>%
  select(Omega_MLE)
results_MLE <- unique(results_MLE)

write.table(results_NB,'results_OmegaNB_trainingset.txt')
write.table(results_MLE,'results_OmegaMLE_trainingset.txt')

#Performance evaluation of training dataset

d_count <- length(unique(df1$d))
n_count <- length(unique(df1$n))

predict_MLE_n <- results_MLE %>%
  group_by(d) %>%
  select(n)

predict_BE_n <- results_NB %>%
  group_by(d) %>%
  select(n)

count_MLE_match <- nrow(merge(df1, predict_MLE_n))
accuracy_MLE <- count_MLE_match/d_count
count_BE_match <- nrow(merge(df1, predict_BE_n))
accuracy_BE <- count_BE_match/d_count
print('Overall Accuracy of BE for training dataset')
print(accuracy_BE)
print ('accuracy_group wise for BE ')
print(count_BE)


count_BE <- count_BE %>%
  group_by(n)%>%
  summarise(accuracy_group = length(d)/length(d))


M<-matrix(0L, nrow=n_count, ncol=n_count)
for (i in 1:d_count){
  true=df1$n[df1$d==i]
  predict=results_NB$n[results_NB$d==i]
  M[true,predict]=M[true,predict]+1
}     
print('Confusion matrix for BE for training dataset')
print(M)
write.table(M,'confusionmatrix_BE_trainingset.txt')


#for test data 

df <- read.csv('train_data.csv', header = T)
df1 <- read.csv('train_label.csv',header = T)
names(df) <- c("d", "w","w_d")
names(df1) <- c("n")
df1$d <- seq.int(nrow(df1))

v1 <- merge(df,df1,by="d",all = TRUE)
library(tidyverse)
c1 <- v1 %>%
  group_by(d) %>%
  mutate(tw_d = sum(w_d))

#classifier_2 <- merge(classifier_1,train_data_label,by = 'docIdx',all=TRUE)
#classifier_2[ , colSums(is.na(classifier_2)) == 0]

c2 <- c1 %>%
  group_by(w,n) %>%
  mutate(w_n = sum(w_d)) # %>%

c3 <- c2 %>%
  group_by(n) %>%
  mutate(tw_n = sum(w_n))

vocab <- read.table('vocabulary.txt',header = FALSE)
vocab$ID <- seq.int(nrow(vocab))
unique(vocab)
vc <- nrow(unique(vocab))

c3$MLE <- c3$w_n/c3$tw_n
c3$Bayesianestimator <- (c3$w_n+1)/(vc+c3$tw_n)
docs <- length(unique(c3$d))


w1 <- c3 %>%
  ungroup() %>%
  expand(w, n)

w2 <- c3 %>%
  select(w,n,w_n,tw_n,d)

merger <- merge(w1,w2,by=c('w','n'),all = TRUE)
merger[is.na(merger)] <- 0
merger <- merger %>%
  group_by(n) %>%
  mutate(tw_n = sum(w_n))
merger <- unique(merger)
merger <- merger[!merger$d == "0", ]

merger$MLE <- log(merger$w_n/merger$tw_n)
merger$Bayesianestimator <- log((merger$w_n+1)/(vc+merger$tw_n))
merger <- merge(merger,p1)
merger$p_wj_log <- log(merger$p_wj)

#for MLE
#sumMLE<-c3 %>%  group_by(d) %>% summarise(sumMLE = sum(log(MLE)))
sumMLE<-merger %>%  group_by(d) %>% 
  summarise(sumMLE = sum((MLE)),sumB=sum((Bayesianestimator)))
dd <- merge(merger,sumMLE,by='d')
dd$Omega_NB<-dd$sumB+dd$p_wj_log
dd$Omega_MLE<-dd$sumMLE+dd$p_wj_log
maxMLE <- dd %>% group_by(d) %>% filter(Omega_MLE == max(Omega_MLE))
maxBayesianb <- dd %>% group_by(d) %>% filter(Omega_NB == max(Omega_NB))

results_NB <- maxBayesianb %>%
  group_by(d,n) %>%
  select(Omega_NB)
results_NB <- unique(results_NB)

results_MLE <- maxMLE %>%
  group_by(d,n) %>%
  select(Omega_MLE)
results_MLE <- unique(results_MLE)
write.table(results_NB,'results_OmegaNB_testset.txt')
write.table(results_MLE,'results_OmegaMLE_testset.txt')

#Performance evaluation of test dataset

d_count1 <- length(unique(df1$d))
n_count1 <- length(unique(df1$n))

predict_MLE_n <- results_MLE %>%
  group_by(d) %>%
  select(n)

predict_BE_n <- results_NB %>%
  group_by(d) %>%
  select(n)

count_MLE <- merge(df1,predict_MLE_n)
count_MLE <- count_MLE %>%
  group_by(n)%>%
  summarise(accuracy_group = length(d)/length(d))

count_MLE <- count_MLE %>%
  group_by(n)%>%
  summarise(accuracy_group = length(d)/length(d))

count_BE <- merge(df1, predict_MLE_n)

count_BE <- count_BE %>%
  group_by(n)%>%
  summarise(accuracy_group = length(d)/length(d))
count_MLE_match <- nrow(merge(df1, predict_MLE_n))
accuracy_MLE <- count_MLE_match/d_count
count_BE_match <- nrow(merge(df1, predict_BE_n))
accuracy_BE <- count_BE_match/d_count
print('Overall Accuracy of BE for test dataset:')
print(accuracy_BE)

print('Overall Accuracy of MLE for test dataset:')
print(accuracy_MLE)

print ('accuracy_group wise for MLE ')
print(count_MLE)
print ('accuracy_group wise for BE ')
print(count_BE)

M<-matrix(0L, nrow=n_count1, ncol=n_count1)
for (i in 1:d_count1){
  true=df1$n[df1$d==i]
  predict=results_MLE$n[results_MLE$d==i]
  M[true,predict]=M[true,predict]+1
}     
print('Confusion Matrix for MLE for test dataset')
print(M)
write.table(M,'confusionmatrix_MLE_testset.txt')

M<-matrix(0L, nrow=n_count, ncol=n_count)
for (i in 1:d_count){
  true=df1$n[df1$d==i]
  predict=results_NB$n[results_NB$d==i]
  M[true,predict]=M[true,predict]+1
}     
print('Confusion Matrix for BE for test dataset')
print(M)
write.table(M,'confusionmatrix_BE_testset.txt')





