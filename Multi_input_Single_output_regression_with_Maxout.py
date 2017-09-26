
# coding: utf-8

# Python 3.5.2
# Test code to do regression with Maxout function with Tensorflow
# (Multi input/Single output)

# In[ ]:


import tensorflow as tf
import numpy as np
import random


# Create training data set 

# In[ ]:


#Target model
def target_mode(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
    cal_res = x1*x2+(x3-3)*x4+x5*x6+x8*1000/x9-20*x10
    return cal_res

#Training data
x_inp_1 = np.random.random([3000,10])*10-5
y_inp_1 = [] 
for i in x_inp_1:
    y_inp_1.append(target_mode(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9]))
y_inp_1 = np.array([y_inp_1])
y_inp_1 = y_inp_1.transpose()

#Varidation data
x_val_1 = np.random.random([3000,10])*10-5
y_val_1 = []
for i in x_val_1:
    y_val_1.append(target_mode(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9]))
y_val_1 = np.array([y_val_1])
y_val_1 = y_val_1.transpose()


# Define Tensorflow model

# In[ ]:


# Function to define weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

# "Maxout" activation Function 
def max_out(input_tensor,output_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] % output_size == 0:
        return tf.transpose(tf.reduce_max(tf.split(input_tensor,output_size,1),axis=2))
    else:
        raise ValueError("Output size or input tensor size is not fine. Please check it. Reminder need be zero.")

# Function to create calculation batch 
def bach_creator(trainig_data_input,trainig_data_output,batch_size):
    if len(trainig_data_input) != len(trainig_data_output):
        pass
        print("Length of input parameter is not same")
    else:
        index_list = random.sample(range(len(trainig_data_input)),batch_size)
        return_input = []
        return_output = []
        for i in index_list:
            return_input.append(trainig_data_input[i])
            return_output.append(trainig_data_output[i])
    return [return_input,return_output]

#Function to layer definition
def nn_layer(input_data, input_size, output_size, maxout_size, layer_name, maxout_flg):
    with tf.name_scope(layer_name):
        if maxout_flg == True:
            weight = weight_variable([input_size,output_size])
            biase = bias_variable([output_size])
            output = max_out(tf.add(tf.matmul(input_data, weight),biase),maxout_size)
        else:
            weight = weight_variable([input_size,output_size])
            biase = bias_variable([output_size])
            output = tf.add(tf.matmul(input_data, weight),biase)
    return output

# Placeholder for training data
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 10])
    y_ = tf.placeholder(tf.float32, [None, 1])

# Define each layer
y1 = nn_layer(x, 10, 40, 5, 'Layer1', True)
y2 = nn_layer(y1, 5, 20, 5, 'Layer2', True)
y3 = nn_layer(y2, 5, 20, 5, 'Layer3', True)
y4 = nn_layer(y3, 5, 1, None, 'Layer4', False)

with tf.name_scope('Loss'):
    loss = tf.reduce_sum(tf.matmul(tf.transpose(y4 - y_), y4 - y_))
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.name_scope('Person_conf'):
    pearson_cof = tf.contrib.metrics.streaming_pearson_correlation(y4, y_)[1]

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#Training 
for step in range(100000):
    
    if step % 2000 == 0:
        res =  sess.run([pearson_cof], feed_dict={x:x_inp_1, y_:y_inp_1})
        print("Step:",step, "  Person coefficinet:", res[0])
    else:
        input_data = bach_creator(x_inp_1,y_inp_1,20)
        sess.run(train_step, feed_dict={x:input_data[0], y_:input_data[1]})

print("done")


# In[ ]:


#Validation of model
res =  sess.run([pearson_cof], feed_dict={x:x_val_1, y_:y_val_1})
print(res[0])
sess.close()

