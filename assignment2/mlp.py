from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class mlp():
    def __init__(self):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(2,)),  
            Dense(128, activation='relu'), 
            Dense(1, activation='sigmoid'),  
        ])
        warmup_steps = 100
        initial_learning_rate = 0.01
        decay_steps = 1000
           
    def compile(self):
        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])
        
    def fit(self, X, y):
        self.model.fit(X, y, epochs=100, 
          batch_size=20, 
          validation_split=0.2)
          
    def evaluate(self, X, y):
        self.model.evaluate(X, y, verbose=0)
