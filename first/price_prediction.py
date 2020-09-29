import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class Specs:
    def __init__(self):
        self.clock_speed = None
        self.fc = None
        self.int_memory = None
        self.n_cores = None
        self.pc = None
        self.ram = None
        self.price_range = None

def showhead():
    dataset = pd.read_csv('E:\\newdata\\new_data.csv')
    return dataset.head()


class LR:
    def __init__(self):
        dataset = pd.read_csv('E:\\newdata\\new_data.csv')


        self.X = dataset.drop('price_range', axis=1)

        #normalization of data
        Xn = ((self.X - self.X.min()) / (self.X.max() - self.X.min()))

        #assigning weights to the variables
        Xn['ram'] = ((self.X['ram'] - self.X['ram'].min()) / (self.X['ram'].max() - self.X['ram'].min())) * 2.7
        Xn['int_memory'] = ((self.X['int_memory'] - self.X['int_memory'].min()) / (self.X['int_memory'].max() - self.X['int_memory'].min())) * 2.4
        Xn['clock_speed'] = ((self.X['clock_speed'] - self.X['clock_speed'].min()) / (self.X['clock_speed'].max() - self.X['clock_speed'].min())) * 0.7
        Xn['n_cores'] = ((self.X['n_cores'] - self.X['n_cores'].min()) / (self.X['n_cores'].max() - self.X['n_cores'].min())) * 0.2
        Xn['pc'] = ((self.X['pc'] - self.X['pc'].min()) / (self.X['pc'].max() - self.X['pc'].min())) * 1.1
        Xn['fc'] = ((self.X['fc'] - self.X['fc'].min()) / (self.X['fc'].max() - self.X['fc'].min())) * 1.1


        y = dataset['price_range']
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.33, random_state=20)

        from sklearn.linear_model import LinearRegression
        self.lm = LinearRegression()
        self.lm.fit(X_train, y_train)
        #y_pred = self.lm.predict(X_test)

        self.score = round(self.lm.score(X_test, y_test),3)

    def LRpredict(self, s):
        if (hasattr(s, 'price_range') == True):
            delattr(s, 'price_range')

        data = pd.DataFrame([s.__dict__],index=[0])
        dfc = pd.DataFrame()
        dfc['ram'] = ((data['ram'] - self.X['ram'].min()) / (self.X['ram'].max() - self.X['ram'].min())) * 2.7
        dfc['int_memory'] = ((data['int_memory'] - self.X['int_memory'].min()) / (self.X['int_memory'].max() - self.X['int_memory'].min())) * 2.4
        dfc['clock_speed'] = ((data['clock_speed'] - self.X['clock_speed'].min()) / (self.X['clock_speed'].max() - self.X['clock_speed'].min())) * 0.7
        dfc['n_cores'] = ((data['n_cores'] - self.X['n_cores'].min()) / (self.X['n_cores'].max() - self.X['n_cores'].min())) * 0.2
        dfc['pc'] = ((data['pc'] - self.X['pc'].min()) / (self.X['pc'].max() - self.X['pc'].min())) * 1.1
        dfc['fc'] = ((data['fc'] - self.X['fc'].min()) / (self.X['fc'].max() - self.X['fc'].min())) * 1.1


        price= round(self.lm.predict(dfc)[0],3)

        if price>=0 and price <= 1.5:
            price = price*7500

        elif price >=1.5 and price <= 2.5:
            price = price*7800

        elif price >=2.5 and price <= 3.3:
            price = price*8300
        elif price >=3.4 and price <= 4:
            price = price*8450

        else :
            price = price * 8650

        '''
        if price>=0 and price <= 1.8:
            price = price*8000

        elif price >=1.8 and price <= 2.9:
            price = price*8200

        elif price >=2.9 and price <= 3.9:
            price = price*8500

        else :
            price = price * 9000
             
         
        '''

        return price




class NBC :
    def __init__(self):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score


        dataset = pd.read_csv('E:\\newdata\\new_data.csv')

        self.X = dataset.drop('price_range', axis=1)

        #normalization of data
        Xn = ((self.X - self.X.min()) / (self.X.max() - self.X.min()))

        #assigning weights to the variables

        Xn['ram'] = ((self.X['ram'] - self.X['ram'].min()) / (self.X['ram'].max() - self.X['ram'].min())) * 2.7
        Xn['int_memory'] = ((self.X['int_memory'] - self.X['int_memory'].min()) / (self.X['int_memory'].max() - self.X['int_memory'].min())) * 2.4
        Xn['clock_speed'] = ((self.X['clock_speed'] - self.X['clock_speed'].min()) / (self.X['clock_speed'].max() - self.X['clock_speed'].min())) * 0.7
        Xn['n_cores'] = ((self.X['n_cores'] - self.X['n_cores'].min()) / (self.X['n_cores'].max() - self.X['n_cores'].min())) * 0.2
        Xn['pc'] = ((self.X['pc'] - self.X['pc'].min()) / (self.X['pc'].max() - self.X['pc'].min())) * 1.1
        Xn['fc'] = ((self.X['fc'] - self.X['fc'].min()) / (self.X['fc'].max() - self.X['fc'].min())) * 1.1

        y = dataset['price_range']
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.33, random_state=20)

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

        predict_train = self.model.predict(X_train)
        #accuracy_train = accuracy_score(y_train, predict_train)
        predict_test = self.model.predict(X_test)

        self.score = accuracy_score(y_test,predict_test)

    def NBCpredict(self,s):

        if (hasattr(s, 'price_range') == True):
            delattr(s, 'price_range')


        data = pd.DataFrame([s.__dict__],index=[0])


        dfc = ((data - self.X.min()) / (self.X.max() - self.X.min()))
        dfc['ram'] = ((data['ram'] - self.X['ram'].min()) / (self.X['ram'].max() - self.X['ram'].min())) * 2.7

        dfc['int_memory'] = ((data['int_memory'] - self.X['int_memory'].min()) / (self.X['int_memory'].max() - self.X['int_memory'].min())) * 2.4
        dfc['clock_speed'] = ((data['clock_speed'] - self.X['clock_speed'].min()) / (self.X['clock_speed'].max() - self.X['clock_speed'].min())) * 0.7
        dfc['n_cores'] = ((data['n_cores'] - self.X['n_cores'].min()) / (self.X['n_cores'].max() - self.X['n_cores'].min())) * 0.2
        dfc['pc'] = ((data['pc'] - self.X['pc'].min()) / (self.X['pc'].max() - self.X['pc'].min())) * 1.1
        dfc['fc'] = ((data['fc'] - self.X['fc'].min()) / (self.X['fc'].max() - self.X['fc'].min())) * 1.1

        return round(self.model.predict(dfc)[0],3)

class KNN:
    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier

        dataset = pd.read_csv('D:\\django_project2\\new_data.csv')

        self.X = dataset.drop('price_range', axis=1)
        Xn = ((self.X - self.X.min())/(self.X.max() - self.X.min()))
        self.y = dataset['price_range']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xn, self.y, test_size=0.33,random_state=20)

        self.knn = KNeighborsClassifier(n_neighbors=4)
        self.knn.fit(self.X_train, self.y_train)
        self.score = round(self.knn.score(self.X_test, self.y_test),10)

    def elbow(self):
        plt.scatter(self.X['ram'],self.X['int_memory'])
        plt.xlabel('fileds')
        plt.ylabel('price_range')
        plt.show()
        from sklearn.neighbors import KNeighborsClassifier
        error_rate = []
        for i in range(1, 10):
            knn1 = KNeighborsClassifier(n_neighbors=i)
            knn1.fit(self.X_train, self.y_train)
            pred_i = knn1.predict(self.X_test)
            error_rate.append(np.mean(pred_i != self.y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 10), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=5)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        return plt



# if __name__ == "__main__":

# s = Specs(clock_speed =3.4,fc =5,int_memory  =82,n_cores   =6,pc  =8,ram  =8333,price_range=None)
# s3 = NBC()
# print(s3.score)
# print(s3.NBCpredict(s))
#
# s1 = LR()
# # print(s1.score)
# print(s1.LRpredict(s))
# s2 = KNN()
# print(s2.score)
# plt1 = s2.elbow()
# plt1.show()
