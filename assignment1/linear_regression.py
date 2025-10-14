import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class LinearRegression():
    
    def __init__(self, learning_rate=0.00001, epochs=50000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        pass
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        if hasattr(X, 'values'): 
            X = X.values
        if hasattr(y, 'values'):  
            y = y.values
            
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        m, n = X.shape  

        self.weights = np.zeros((n, 1)) 
        self.bias = 0  
        
        # Gradient Descent
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

    
    def predict(self, X):
        """
        Generates predictions
        """
        if hasattr(X, 'values'):
            X = X.values
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred.flatten()

    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradients for weights and bias
        """
        m = len(y)
        
        # For weights
        grad_w = (1 / m) * np.dot(X.T, (y_pred - y))
        
        # For bias
        grad_b = (1 / m) * np.sum(y_pred - y)
        
        return grad_w, grad_b
    
    def _compute_loss(self, y, y_pred):
        """
        Computes the MSE loss
        """

        m = 1/(2*len(y))
        return m * np.sum((y - y_pred) ** 2)
        
    def update_parameters(self, grad_w, grad_b):
        """
        Update weights and bias
        """
        
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b 
        
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    


class LogisticRegression():
    
    def __init__(self, learning_rate=0.05, epochs=2000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        pass
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        m, n = X.shape  

        self.weights = np.zeros((n, 1)) 
        self.bias = 0  
        
        # Gradient Descent
        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self._compute_cross_entropy_loss(y, y_pred)
            self.losses.append(loss)
            
    
    def predict(self, X):
        """
        Generates predictions
        """
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int).flatten()
    
    def predict_probabilities(self, X):
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred.flatten()
        

    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradients for weights and bias
        """
        m = len(y)
        
        # For weights
        grad_w = (1 / m) * np.dot(X.T, (y_pred - y))
        
        # For bias
        grad_b = (1 / m) * np.sum(y_pred - y)
        
        return grad_w, grad_b
    
    def _compute_loss(self, y, y_pred):
        """
        Computes the MSE loss
        """

        m = 1/(2*len(y))
        return m * np.sum((y - y_pred) ** 2)
    
    def _compute_cross_entropy_loss(self, y, y_pred):
        m = len(y)
        epsilon = 1e-15  
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
    def update_parameters(self, grad_w, grad_b):
        """
        Update weights and bias
        """
        
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b 
        
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
  
        
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
class DecisionTreeBuilder():
    def __init__(self, max_depth=5, random_state=42):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        pass
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_probabilities(self, X):
        return self.model.predict_proba(X)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class DecisionTreeDecrypter():
    def __init__(self, max_depth=5, random_state=42, power_of_ten=3, chosen_feature='data_stream_3'):
        self.max_depth = max_depth
        self.random_state = random_state
        self.power_of_ten = power_of_ten
        self.chosen_feature = chosen_feature
        self.feature_index = None
        pass
    
    def fit(self, X, y):  
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        feature_names = [f'data_stream_{i}' for i in range(X.shape[1])]
        
        self.feature_index = feature_names.index(self.chosen_feature)
            
        X_trans = self.to_Binary(X.copy())
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        self.model.fit(X_trans, y)
        

    def predict(self, X):
        X = np.array(X)
        X_trans = self.to_Binary(X.copy())
        return self.model.predict(X_trans)
    
    def predict_proba(self, X):
        X = np.array(X)
        X_trans = self.to_Binary(X.copy())
        return self.model.predict_proba(X_trans)
        
    def to_Binary(self, X):  
        X_trans = X.copy()
        
        feature_values = X_trans[:, self.feature_index]
        binary_values = (feature_values * (10 ** self.power_of_ten)).astype(int) % 2
        X_trans[:, self.feature_index] = binary_values
        
        return X_trans
 
from sklearn.ensemble import RandomForestClassifier
class RandomForestDecrypter():  
    def __init__(self, max_depth=5, random_state=42, power_of_ten=3, chosen_feature='data_stream_3'):
        self.max_depth = max_depth
        self.random_state = random_state
        self.power_of_ten = power_of_ten
        self.chosen_feature = chosen_feature
        self.feature_index = None
        pass
    
    def fit(self, X, y):  
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        feature_names = [f'data_stream_{i}' for i in range(X.shape[1])]
        
        self.feature_index = feature_names.index(self.chosen_feature)
            
        X_trans = self.to_Binary(X.copy())
        self.model = RandomForestClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        self.model.fit(X_trans, y)
        

    def predict(self, X):
        X = np.array(X)
        X_trans = self.to_Binary(X.copy())
        return self.model.predict(X_trans)
    
    def predict_proba(self, X):
        X = np.array(X)
        X_trans = self.to_Binary(X.copy())
        return self.model.predict_proba(X_trans)
        
    def to_Binary(self, X):  
        X_trans = X.copy()
        
        feature_values = X_trans[:, self.feature_index]
        binary_values = (feature_values * (10 ** self.power_of_ten)).astype(int) % 2
        X_trans[:, self.feature_index] = binary_values
        
        return X_trans
  

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
class NexusRatingModel():
    def __init__(self):
        self.meta_model = SklearnLinearRegression()
        self.label_encoders = {}
        self.stack_model = None
    
    def fix_columns(self, train, test):
        train = np.array(train)
        test = np.array(test)

        moved_target_col = test[:, -1]  
        test = np.delete(test, -1, axis=1)

        nexus_position = 1  
        test = np.insert(test, nexus_position, moved_target_col, axis=1)
        
        return train, test
    
    def handle_data(self, data, is_train=True):
        data = np.array(data)
        data_clean = data.copy()

        for col_idx in range(data_clean.shape[1]):
            if col_idx != 'nexus_rating':  
                col = data_clean[:, col_idx]
                if np.issubdtype(col.dtype, np.number):
                    median = np.nanmedian(col)
                    col[np.isnan(col)] = median
                else:  
                    unique, counts = np.unique(col[~np.isnan(col)], return_counts=True)
                    mode = unique[np.argmax(counts)] if len(unique) > 0 else 'Unknown'
                    col[np.isnan(col)] = mode

                    if is_train:
                        self.label_encoders[col_idx] = LabelEncoder()
                        col = self.label_encoders[col_idx].fit_transform(col)
                    else:
                        if col_idx in self.label_encoders:
                            known_cats = self.label_encoders[col_idx].classes_
                            col = np.array([x if x in known_cats else known_cats[0] for x in col])
                            col = self.label_encoders[col_idx].transform(col)

                data_clean[:, col_idx] = col
        
        return np.array(data_clean)
    
    def train_model(self, train):
        train_clean = self.handle_data(train, is_train=True)
        X = train_clean[:, :-1]
        y = np.log1p(train_clean[:, -1]) 

        stack = StackingRegressor(estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=5, random_state=42))
        ], final_estimator=self.meta_model, cv=5, n_jobs=1)
    
        stack.fit(X, y)
        
        self.stack_model = stack
    
    def predict(self, test):
        test_clean = self.handle_data(test, is_train=False)
        X_test = test_clean[:, :-1]
        y_true = test_clean[:, -1]

        y_pred_log = self.stack_model.predict(X_test)
        predictions = np.expm1(y_pred_log)  
    
        return y_true, predictions

