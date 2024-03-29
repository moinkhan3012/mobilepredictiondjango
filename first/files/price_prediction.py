import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#%matplotlib inline
from sklearn.model_selection import train_test_split

# from sklearn.neighbors import KNeighborsClassifier
# print("load data")
# dataset=pd.read_csv('E:\\Smart-Phone-Price-Prediction\\datasets\\train.csv')
# print(dataset.head())
# X=dataset.drop('price_range',axis=1)
# y=dataset['price_range']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
#
# test={'battery_power':1044,'blue':1,
# 'clock_speed':2.4,
# 'dual_sim':1,
# 'fc':5,
# 'four_g':1,
# 'int_memory':42,
# 'm_dep':0.6,
# 'mobile_wt':230,
# 'n_cores ':6,
# 'pc':8,
# 'px_height':1233,
# 'px_width':1713,
# 'ram':4533,
# 'sc_h' :15,
# 'sc_w' :8,
# 'talk_time':23,
# 'three_g':1,
# 'touch_screen':1,
# 'wifi ' :1}


class Specs:
    def __init__(self,battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range):
        self.battery_power = battery_power
        self.blue = blue
        self.clock_speed =clock_speed
        self.dual_sim=dual_sim
        self.fc = fc
        self.four_g =four_g
        self.int_memory= int_memory
        self.m_dep = m_dep
        self.mobile_wt= mobile_wt
        self.n_cores = n_cores
        self.pc = pc
        self.px_height = px_height
        self.px_width = px_width
        self.ram=ram
        self.sc_h = sc_h
        self.sc_w= sc_w
        self.talk_time =talk_time
        self.three_g=three_g
        self.touch_screen =touch_screen
        self.wifi = wifi
        self.price_range= price_range

def showhead():
    dataset = pd.read_csv('E:\\Smart-Phone-Price-Prediction\\datasets\\train.csv')
    return dataset.head()

class LR:
    def __init__(self):
        dataset = pd.read_csv('E:\\Smart-Phone-Price-Prediction\\datasets\\train.csv')
        X = dataset.drop('price_range', axis=1)
        y = dataset['price_range']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

        from sklearn.linear_model import LinearRegression
        self.lm= LinearRegression()
        self.lm.fit(X_train,y_train)
        self.score = self.lm.score(X_test,y_test)

    def LRscore(self):
        return self.score

    def LRpredict(self,s):
        if( hasattr(s,'price_range') == True):
            delattr(s,'price_range')
        return self.lm.predict(pd.DataFrame(s.__dict__,index=[0]))*10000


class KNN:
    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier
        dataset = pd.read_csv('E:\\Smart-Phone-Price-Prediction\\datasets\\train.csv')
        self.X = dataset.drop('price_range', axis=1)
        self.y = dataset['price_range']
        self.X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

        self.knn = KNeighborsClassifier(n_neighbors=10)
        self.knn.fit(X_train,y_train)
        self.score = self.knn.score(X_test,y_test)

    def elbow(self):
        from sklearn.neighbors import KNeighborsClassifier
        error_rate = []
        for i in range(1, 20):

            knn1 = KNeighborsClassifier(n_neighbors=i)
            knn1.fit(self.X_train,self. y_train)
            pred_i = knn1.predict(self.X_test)
            error_rate.append(np.mean(pred_i != self.y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 20), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=5)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        return plt

if __name__ == "__main__":

    s = Specs(battery_power=1044,blue=1,clock_speed =7.4,dual_sim =1,fc =5,four_g =1,int_memory  =82,m_dep  =0.6,mobile_wt  =190,n_cores   =6,pc  =8,px_height  =1533,px_width  =1913,ram  =8333,sc_h   =15,sc_w   =8,talk_time  =23,three_g  =1,touch_screen  =1,wifi=1,price_range=None)
    s1 = LR()
    print(s1.LRscore())
    print(s1.LRpredict(s))
    s2 = KNN()
    print(s2.score)
    plt1 = s2.elbow()
    plt1.show()


