from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

class mlp():
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(256, activation='sigmoid'),  
            Dense(128, activation='sigmoid'), 
            Dense(10, activation='softmax'),  
        ])

