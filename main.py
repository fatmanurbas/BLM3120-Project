import seaborn as sns
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# databasei kaydetmek
fraudData = pd.read_csv("onlinefraud.csv")

# type kısmını bölmek
fraudData = pd.get_dummies(fraudData, prefix=['type'], columns=['type'], drop_first=True)

# preprocessing
fraudData.isFraud.value_counts()

# class count
class_count_0, class_count_1 = fraudData['isFraud'].value_counts()

# Separate class
class_0 = fraudData[fraudData['isFraud'] == 0]
class_1 = fraudData[fraudData['isFraud'] == 1]  # print the shape of the class
class_0_under = class_0.sample(4 * class_count_1)

# balancing_data; yeni verisetimiz oluyor
balancing_data = pd.concat([class_0_under, class_1], axis=0)
#####balancing_data['isFraud'].value_counts().plot(kind='bar', title='count (target)')

# x ve y yi ayırmak
X = balancing_data[['oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
                    'type_TRANSFER', 'amount', 'oldbalanceOrg', 'newbalanceOrig']].values
y = balancing_data.loc[:, 'isFraud'].values

# x ve y yi bölmek
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale etmek
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
#print(scaler.mean_)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import preprocessing
from sklearn import utils

lab = preprocessing.LabelEncoder()
y_train = lab.fit_transform(y_train)
y_test = lab.fit_transform(y_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def log_reg():
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = round(accuracy_score(y_test, predictions), 3)
    cm1 = cm(y_test, predictions)
    sns.heatmap(cm1, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Logistic Regression Accuracy Score: {0}'.format(score), size=15)
    #plt.show()
    plt.savefig("reg.png")
    plt.clf()
    roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    return str(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

# decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm

def dec_tree(max_depth_input):
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    d_tree1 = DecisionTreeClassifier(max_depth=max_depth_input)
    d_tree1.fit(X_train, y_train)
    predictions = d_tree1.predict(X_test)
    score = round(accuracy_score(y_test, predictions), 3)
    cm1 = cm(y_test, predictions)
    sns.heatmap(cm1, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Decision Tree Accuracy Score: {0}'.format(score), size=15)
    #plt.show()
    plt.savefig("tree.png")
    plt.clf()
    return str(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

# random forest
from sklearn.ensemble import RandomForestClassifier

def rand_for(max_depth_input):
    rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth_input)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    score = round(accuracy_score(y_test, predictions), 3)
    cm1 = cm(y_test, predictions)
    sns.heatmap(cm1, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Random Forest Accuracy Score: {0}'.format(score), size=15)
    #plt.show()
    plt.savefig("rand.png")
    plt.clf()
    return str(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

import tkinter as tk
from tkinter import *
from tkinter import Label
from PIL import Image, ImageTk

class GUi():

    def fr(self):
        frame = tk.Tk()
        frame.title("Menu")
        frame.geometry("500x500")

        def getrf_input():
            global rf_entry, rf_str
            rf_str = int(rf_entry.get())
            rf_frame()

        def getdt_input():
            global dt_entry, dt_str
            dt_str = int(dt_entry.get())
            dt_frame()

        def dt_frame():
            dt_frame = Toplevel(frame)
            dt_frame.title("Decision Tree")
            dt_frame.geometry("500x500")
            label = tk.Label(dt_frame, text="Decision Tree Result Screen for Max Depth= " '{}'.format(dt_str), font=('Arial', 10, 'bold'))
            label.pack()
            tree_output = dec_tree(dt_str)
            label2 = tk.Label(dt_frame, text=tree_output).pack()
            photo = Image.open("tree.png")
            photo = photo.resize((300, 300), Image.ANTIALIAS)
            photo_image = ImageTk.PhotoImage(photo)
            label3 = tk.Label(dt_frame, image=photo_image)
            label3.pack()

            dt_frame.mainloop()

        def rf_frame():
            rf_frame = Toplevel(frame)
            rf_frame.title("Random Forest")
            rf_frame.geometry("500x500")
            label = tk.Label(rf_frame, text="Random Forest Result Screen for Max Depth= " '{}'.format(rf_str), font=('Arial', 10, 'bold'))
            label.pack()
            rand_output = rand_for(rf_str)
            label2 = tk.Label(rf_frame, text=rand_output).pack()
            photo = Image.open("rand.png")
            photo = photo.resize((300, 300), Image.ANTIALIAS)
            photo_image = ImageTk.PhotoImage(photo)
            label3 = tk.Label(rf_frame, image=photo_image)
            label3.pack()

            rf_frame.mainloop()

        def frame1():
            frame1 = Toplevel(frame)
            frame1.title("Select Depth for Decision Tree")
            frame1.geometry("300x100")
            global dt_entry
            label1 = tk.Label(frame1, text="Enter the max depth of the tree:")
            label1.pack()
            dt_entry = tk.Entry(frame1, width=7)
            dt_entry.pack()
            input_buton = tk.Button(frame1, text="Confirm", command=getdt_input)
            input_buton.pack()

            frame1.mainloop()

        def frame2():
            frame2 = Toplevel(frame)
            frame2.title("Select Depth for Random Forest")
            frame2.geometry("300x100")
            global rf_entry
            label1 = tk.Label(frame2, text="Enter the max depth of the tree:")
            label1.pack()
            rf_entry = tk.Entry(frame2, width=7)
            rf_entry.pack()
            input_buton = tk.Button(frame2, text="Confirm", command=getrf_input)
            input_buton.pack()

            frame2.mainloop()

        def lr_frame():
            lr_frame = Toplevel(frame)
            lr_frame.title("Logistic Regression")
            lr_frame.geometry("500x500")
            label = tk.Label(lr_frame, text="Logistic Regression Result Screen:", font=('Arial', 10, 'bold'))
            label.pack()
            str = log_reg()
            label2 = tk.Label(lr_frame, text=str).pack()
            photo = Image.open("reg.png")
            photo = photo.resize((300, 300), Image.ANTIALIAS)
            photo_image = ImageTk.PhotoImage(photo)
            label3 = tk.Label(lr_frame, image=photo_image)
            label3.pack()
            lr_frame.mainloop()

        label1 = tk.Label(text="SELECT ONE OF THE FOLLOWING ALGORITHMS: ").place(x=130, y=80)
        buton1 = tk.Button(text="Random Forest", command=frame2, bg="red", width=20, height=3).place(x=190, y=130)
        buton2 = tk.Button(text="Decision Tree", command=frame1, bg="blue", width=20, height=3).place(x=190, y=210)
        buton3 = tk.Button(text="Logistic Regression", command=lr_frame, bg="green", width=20, height=3).place(x=190,
                                                                                                             y=290)
        name1 = tk.Label(frame, text="19011084-Fatmanur BAŞ", font=('Arial', 10, 'bold')).place(x=175, y=420)
        name2 = tk.Label(frame, text="19011088-Dilara DELEN", font=('Arial', 10, 'bold')).place(x=175, y=450)


        frame.mainloop()

interface = GUi()
interface.fr()
