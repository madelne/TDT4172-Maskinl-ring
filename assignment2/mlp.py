import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class mlp():
    def __init__(self, activation='relu'):
        self.model = Sequential([
            Dense(256, activation=activation, input_shape=(2,)),  
            Dense(128, activation=activation), 
            Dense(1, activation='sigmoid'),  
        ])
        self.warmup_steps = 100
        self.initial_learning_rate = 0.01
        self.decay_steps = 1000
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=self.initial_learning_rate,
        decay_steps=self.decay_steps,)
        
    def __call__(self):
        warmup_lr = self.initial_learning_rate * (self.warmup_steps / self.warmup_steps)
        return tf.cond(
            self.warmup_steps < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.lr_schedule(self.warmup_steps - self.warmup_steps)
        )  
        
    def compile(self):
        optimizer = Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  
              metrics=['accuracy'])
        
    def fit(self, X, y, callbacks=None):
        history = self.model.fit(X, y, epochs=100, 
          batch_size=20, 
          validation_split=0.2,
          callbacks=callbacks
          )
        return history
          
    def evaluate(self, X, y):
        self.model.evaluate(X, y, verbose=0)
        