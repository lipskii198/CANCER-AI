#this ai will tell if a tumor is malign(1) or no(0), this AI IS NOT A MEDICAL TOOL
#thanks to KhanradCoder for the data 
import panda as pd 
from sklearn.modelselection import train_test_splits
import tesorflow as tf 
dataset = pd.read_csvl('cancer.csv')

#this contains all the data of the tumor except if it's malign or not 
x=dataset.drop(columns=["diagnosis(1=m, 0=b)"])

#this contains only the resoults
y=dataset["diagnosis(1=m, 0=b)"]

#we split the training data to avoid over-fitting(AI performs well on the training data but not on the new one), 20% will be used for testing
x_train,x_test,y_train,y_test = train_test_splits(x, y, test_size = 0.2)

#creating the model using keras
model = tf.keras.models.Sequentail()

#creating the input layer (256 is obv too big, you can try and rescale it to any power of 2)
#by using the sigmoid function any value will be between 0 and 1, this will help with complexity and accuracy 
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape,activation='sigmoid'))
#just adding middle and output layer 
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#since we use binary data binary_crossentropy is very efficient 
model.compile(optimizer='adams', loss='binary_crossentropy', metrics=['accuracy'])

#fit the data, we run 1000 times over the same data
model.fit(x_train,y_train, epochs= 1000)

#evaluate accuracy, it's 97+% 
model.evaluate(x_test,y_test)