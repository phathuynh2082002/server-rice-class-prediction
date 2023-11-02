# Khai báo thư viện
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

# Đọc dữ liệu từ file Rice.xlsx
# df = pd.read_excel('Rice.xlsx', sheet_name='Data')
df = pd.read_excel('data_sets.xlsx', sheet_name='Data')
# Kiểm tra xem tập dữ liệu có bị thiếu hay không
# has_empty_cells = df.isnull().values.any()

# Thay thế các giá trị bị thiếu bằng giá trị trung bình
column_means = df.mean()
df = df.fillna(column_means)

# Xóa các điểm nhiễu ở giá trị cực đại trong 2 cột Major_Axis_Length và Major_Axis_Length
def denoised_data(data):
     while(data['Major_Axis_Length'].max()/data['Major_Axis_Length'].mean() > 5.0):
          data = data[data['Major_Axis_Length'] != data['Major_Axis_Length'].max()]
     while (data['Major_Axis_Length'].max() / data['Minor_Axis_Length'].mean() > 5.0):
          data = data[data['Minor_Axis_Length'] != data['Minor_Axis_Length'].max()]
     return data

new_df = denoised_data(df)

# Lọc dữ liệu
X_data = new_df.drop(columns=['Id','Class'])

X = X_data.copy()
X_predict = X_data.copy()
y= new_df['Class']

# Chuẩn hóa dữ liệu huấn luyện
def standar_data(data):
     scaler = StandardScaler()
     scaler.fit(data)
     data = scaler.transform(data)
     return data

def standar_data_predict(data):
     mean = np.mean(X_predict)
     std = np.std(X_predict)
     data = data - mean
     data /= std
     return data

X = standar_data(X)

# Sử dụng giải thuật K Láng Giềng
def knn(X_train, X_test, y_train):
     knn = KNeighborsClassifier(n_neighbors=15)
     knn.fit(X_train, y_train)
     y_pred = knn.predict(X_test)
     return y_pred

# Sử dụng giải thuật Cây Quyết Định
def desion_tree(X_train, X_test, y_train):
     decision_tree = DecisionTreeClassifier(criterion="gini", min_samples_leaf=30)
     decision_tree.fit(X_train, y_train)
     y_pred = decision_tree.predict(X_test)
     return y_pred

# Sử dụng giải thuật Bayes Thơ Ngây
def naive_bayes(X_train, X_test, y_train):
     naive_bayes = GaussianNB()
     naive_bayes.fit(X_train, y_train)
     y_pred = naive_bayes.predict(X_test)
     return y_pred

# Sử dụng giải thuật Vecto hỗ trợ
def sup_vector(X_train, X_test, y_train):
     model = svm.SVC(kernel='rbf')
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     return  y_pred

# Sử dụng giải thuật Rừng Ngẫu Nhiên
def random_forest(X_train, X_test, y_train):
     random_forest = RandomForestClassifier(n_estimators=100)
     random_forest.fit(X_train, y_train)
     y_pred = random_forest.predict(X_test)
     return y_pred

# Confusion maxtrix
def cfn_matrix(y_test, y_pred, title):
     cm = confusion_matrix(y_test, y_pred)
     cm_df = pd.DataFrame(cm,index=['C', 'O'],columns=['C', 'O'])
     plt.figure(figsize=(6, 4))
     sns.heatmap(cm_df, annot=True, fmt='.1f')
     plt.title(title)
     plt.ylabel('True label')
     plt.xlabel('Predicted label')
     plt.show()

# Tính độ chính xác của giải thuật
def acc(y_test, y_pred):
     return accuracy_score(y_test, y_pred)

# Tính trung bình độ chính xác của từng giải thuật sau n lần lặp
def acc_avg(n):
     acc_knn = 0
     acc_dt = 0
     acc_nb = 0
     acc_svm = 0
     acc_rf = 0
     for x in range(n):

          # Chia tập dữ liệu với nghi thức k-fold
          # kf = KFold(n_splits=5,shuffle=True)
          # for train_index, test_index in kf.split(X, y):
          #      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
          #      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

          # Chia tập dữ liệu với nghi thức Hold-out
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

          knn_predict = knn(X_train, X_test, y_train)
          desion_tree_predict = desion_tree(X_train, X_test, y_train)
          naive_bayes_predict = naive_bayes(X_train, X_test, y_train)
          sup_vector_predict = sup_vector(X_train, X_test, y_train)
          random_forest_predict = random_forest(X_train, X_test, y_train)

          # cfn_matrix(y_test, knn_predict,'knn')
          # cfn_matrix(y_test, desion_tree_predict,'desion_tree')
          # cfn_matrix(y_test, naive_bayes_predict,'naive_bayes')
          # cfn_matrix(y_test, sup_vector_predict,'sup_vector')
          # cfn_matrix(y_test, random_forest_predict,'random_forest')

          acc_knn += acc(y_test,knn_predict)
          acc_dt += acc(y_test,desion_tree_predict)
          acc_nb += acc(y_test,naive_bayes_predict)
          acc_svm += acc(y_test,sup_vector_predict)
          acc_rf += acc(y_test,random_forest_predict)
     print("Độ chính xác của KNN là : %.3f" % (acc_knn/n))
     print("Độ chính xác của Decision Tree là : %.3f" %(acc_dt/n))
     print("Độ chính xác của Naive Bayes là : %.3f" %(acc_nb/n))
     print("Độ chính xác của SVM là : %.3f" %(acc_svm/n))
     print("Độ chính xác của Random Forest là : %.3f" %(acc_rf/n))

# Lưu mô hình máy học ứng dụng
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
def svm_save():
     model = svm.SVC(kernel='rbf')
     model.fit(X_train, y_train)
     sup_vector = 'sup_vector_model.sav'
     pickle.dump(model, open(sup_vector, 'wb'))

# svm_save()
# acc_avg(50)
