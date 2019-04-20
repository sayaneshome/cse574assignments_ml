# refresh package cache
import weka.core.jvm as jvm
import numpy as np
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz 
import arff


jvm.start(max_heap_size="512m",system_cp=True, packages=True)
data_dir = "/Users/sayaneshome/Lab02_cse573/db/"
print('question 1 solutions : \n')

dataset = arff.load(open(data_dir + 'voting_record.arff', 'r'))
data = np.array(dataset['data'])
data_x = data[:,1:9]
y = data[:,10]
X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.2)
np1 = np.array_split(data, 5)
cls2 = tree.DecisionTreeClassifier(criterion = 'entropy')
#print(X_train,X_test,y_test,y_train)
#print(cls2)
cls2 = cls2.fit(X_train, y_train)
y_pred_en = cls2.predict(X_test)
dot_data = tree.export_graphviz(cls2, out_file= data_dir + 'tree_voting.dot')
graph = graphviz.Source(dot_data)
print("Accuracy for voting records dataset is ", accuracy_score(y_test,y_pred_en)*100)


dataset = arff.load(open(data_dir + 'test.arff', 'r'))
data = np.array(dataset['data'])
data_x = data[:,1:9]
y = data [:,10]
X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.2)
np1 = np.array_split(data, 5)
cls2 = tree.DecisionTreeClassifier(criterion = 'entropy')

print(np1_test.shape)
#print(X_train,X_test,y_test,y_train)
#print(cls2)
cls2 = cls2.fit(X_train, y_train)
y_pred_en = cls2.predict(X_test)
dot_data = tree.export_graphviz(cls2, out_file= data_dir + 'tree_breastcancer.dot')
graph = graphviz.Source(dot_data)
print('.dot format for Tree images are stored in the folder\n')
print('For accessing png and pdf files for .dot format,please type the following commands in the terminal\n')
print('dot -Tpng tree_breastcancer.dot -o tree_breastcancer.png\n')
print("Accuracy for breast cancer data(full) is ", accuracy_score(y_test,y_pred_en)*100)


print('question 2 solutions : \n')

##evl = Evaluation(data)
##evl.crossvalidate_model(cls, data, 5, Random(1))
##print(evl.percent_correct)
##print(evl.summary())
##print(evl.class_details())
##
##f = (100 - evl.percent_correct)/100
##s = f*(1-f)/len(data)

conf_interval_p = f + 1.96*(np.sqrt(s))
conf_interval_n = f - 1.96*(np.sqrt(s))
print('0.95 Confidence Intervals for Breast cancer data records\n',conf_interval_n,',',conf_interval_p)
data1 = loader.load_file(data_dir + "voting_record.arff")
data1.class_is_last()


#cls = Classifier(classname="weka.classifiers.trees.J48")
#print(cls.options)
cls.build_classifier(data1)
print (cls)


evl1 = Evaluation(data1)
evl1.crossvalidate_model(cls, data1, 5, Random(1))
print(evl1.percent_correct)
print(evl1.summary())
#print(evl1.class_details())
f1 = (100 - evl1.percent_correct)/100
s1 = f*(1-f)/len(data1)

conf_interval_p1 = f1 + 1.96*(np.sqrt(s1))
conf_interval_n1 = f1 - 1.96*(np.sqrt(s1))

print('0.95 Confidence Intervals for Voting records\n',conf_interval_n1,',',conf_interval_p1)

test = [(0,1,2),(9,0,1),(0,1,3),(0,1,8)]
print(test)
test_np = np.array_split(test,4)

#http://staffwww.itn.liu.se/~aidvi/courses/06/dm/lectures/lec6.pdf

print('question 3 solutions : \n')




##print(np1_train)
##print(np1_test)

print(len(data))


for i in range(1,5):
##    add = np1[:i]
##    np1_test = np1[i]
##    np1_train= add+np1[i+1:]
##    np_train = np.array(np1_train)
##    np_test = np.array(np1_test)

    np1_train = np.concatenate(np1[i:]+np1[:i-1])
    #np_test = np1[:i]
    #np_p = np1[2,:]
    #print(np_p.shape)
    np1_test = np1[i]
    print(np1_train.shape)
    print(np1_test.shape)
    #np1_train = np.append(add)
   
    X_train = np1_train[:,1:9]
    X_test = np1_test[:,1:9]
    y_test = np1_test[:,10]
    y_train = np1_train[:,10]
    cls2 = tree.DecisionTreeClassifier(criterion = 'entropy')
   # print(X_train,X_test,y_test,y_train)
 #   print(cls2)
    cls2 = cls2.fit(X_train, y_train)
    y_pred_en = cls2.predict(X_test)
    dot_data = tree.export_graphviz(cls2, out_file= data_dir + 'tree'+str(i) +'.dot')
    graph = graphviz.Source(dot_data)
    print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
    i = i+1 
##>>> cv_results = cross_validate(lasso, X, y, cv=3,
##...                             return_train_score=False)
##>>> sorted(cv_results.keys())                         
##['fit_time', 'score_time', 'test_score']
##>>> cv_results['test_score']    
##array([0.33150734, 0.08022311, 0.03531764])
##X_test = np1_X[5]
##y_train = np1_Y[1],np1_Y[2],np1_Y[3]
##Y_train.append(np1_Y[4])
##y_test = np1_Y[5]
##print(dot_data)
##print(graph)
##for i in range(1,5):
##    test = np.array(np1[i])
##    #cls.build_classifier(train)
##    print(test)
##    cls = Classifier(classname="weka.classifiers.trees.J48")
###print(cls.options)
##    cls.build_classifier(test)
##    print (cls)
##print(data)


##import numpy as np
##
##test = [(0,1,2),(9,0,1),(0,1,3),(0,1,8)]
##test=np.array(test)
##test = np.array_split(test, 4)
##t_0 = test[:0]
##t_1 = test[0]
##new_test= t_0+test[1:]
##print(new_test)   
# as np.array:


jvm.stop()

#References
#link : https://fracpete.github.io/python-weka-wrapper/examples.html#build-classifier-on-dataset-output-predictions



