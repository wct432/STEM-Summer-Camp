import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class simple_neural_net(object):
    
    def __init__(self, lr=0.01, epochs=10, activation='relu'):
        self.lr_ = lr
        self.epochs_ = epochs
        self.activations_dict_ = {'sigmoid': self.sigmoid,'tanh': self.tanh,'relu': self.relu,'leaky_relu': self.leaky_relu}
        self.activation_ = self.activations_dict_[activation]
        self.fitted_values_ = []
        self.costs_ = []
        self.best_epoch_ = 0

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        return ((e ** x) - (e ** -x)) / ((e ** x) + (e ** -x))

    def relu(self, x) :
        return np.where(x>0,x,0)

    def leaky_relu(self, x) :
        return np.where(x > 0, x, x * 0.01)
    
    def linear(self, x):
        return x
    
    def mse(self, actual, predicted):
        '''Calculates Mean Square Error for cost'''
        actual = np.array(actual)
        predicted = np.array(predicted)
        differences = np.subtract(actual, predicted)
        squared_differences = np.square(differences)
        return squared_differences.mean()
    
        return differences
    
    def neuron(self, X,W, activation):
        X=np.array(X)                     # Inputs
        W=np.array(W)                     # Weights
        X_biased = np.append(X,self.b_)   # Adding Bias
        X_weighted = X_biased * W;        # Combining the X values and their weights by multiplying them together
        X_sum = np.sum(X_weighted);       # Sum
        output = self.activation_(X_sum); # Activation Function
        return output                     # Output
    
    def init_params(self):
        '''Initializes weights and bias'''
        self.w_ = np.random.randn(4, 1)*0.01
        self.b_ = 1
    
    def propagate(self, X, y):
        '''Propogation'''
        m = X.shape[0]
        A = [self.neuron(x, self.w_, self.activation_) for x in X]
        
        #find the cost
        predictions = [self.neuron(x, self.w_, self.activation_) for x in X]
        self.fitted_values_.append(predictions)
        cost = self.mse(y,predictions)

        #find gradient (back propagation)
        dw = (1/m) * np.dot(X, (A-y).T)
        db = (1/m) * np.sum(A-y)
        self.costs_.append(cost)
        grads = {"dw": dw,
                 "db": db}
        return grads
    
    def gradient_descent(self, X, y):
        '''Performs gradient descent; updates weights and bias based on gradients found during backpropogation.'''
        costs = []
        non_improve_count = 0
        for i in range(self.epochs_):
            grads = self.propagate(X, y)
            
            if i > 0:
                if self.costs_[i] > self.costs_[i - 1]:
                    non_improve_count += 1
            
            if self.costs_[i] < self.costs_[self.best_epoch_]:
                self.best_epoch_ = i
                
            if non_improve_count > 1:
                self.lr_ = (self.lr_ / 20)
                non_improve_count = 0
                
            #update parameters
            self.w_ = self.w_ - (self.lr_ * grads["dw"])
            self.b_ = self.b_ - (self.lr_ * grads["db"])

#             if i % 10000 == 0:
#                 print(f'Cost after iteration {i}: {self.costs_[i]}, LR: {self.lr_:00f}')

        return None

    def fit(self,X,y):
        self.init_params()
        self.gradient_descent(X,y)

    def predict(self,X):
        return [self.neuron(x, self.w_, self.activation_) for x in X]
        


    def plot_fitted_vals(self,X,y,training=0):
        fig, ax = plt.subplots(figsize=(12,3))
        sns.scatterplot(x=X,y=y, ax=ax, color='orange', s=100, label='Actual Values')
        sns.scatterplot(x=X,y=np.array(self.fitted_values_[training]),ax=ax, color='blue', label='Predicted Values')
        ax.set_title(f'Fitted Values at Epoch {training} with Model Error of {self.costs_[training]:.4f}', fontsize=16)