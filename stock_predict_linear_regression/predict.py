import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#hidden layer
rnn_unit=128
#feature 
input_size=8
output_size=1
lr=0.0006    
#csv_file = 'stock3005.csv' 
csv_file = 'stock2317.csv'
#csv_file = 'stock2330.csv' 
f=open(csv_file) 
df=pd.read_csv(f)
data=df.iloc[:,1:10].values
#train data end row, I use 700 rows as train data and 30 row as test data
data_train_rows = 630 

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }


def prediction(time_step=5):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint("./")
        saver.restore(sess, module_file) 
        print("model restored");
        test_predict=[]
        for step in range(len(test_x)):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[input_size]+mean[input_size]
        test_predict=np.array(test_predict)*std[input_size]+mean[input_size]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])
        avg_diff=np.average(np.abs(test_predict-test_y[:len(test_predict)]))

        print ("avg_diff=",avg_diff,", acc=",acc)

        plt.figure()

        fields = ['date']
        df = pd.read_csv(csv_file, skipinitialspace=True, usecols=fields)
        x_data = np.array(df.date)[data_train_rows:]
        

        plt.xticks(list(range(len(test_predict))), x_data, rotation=90, fontsize=10)
        plt.plot(list(range(len(test_predict))), test_predict, 'ro')
        #plt.plot(x_data, test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  'bo')
        plt.show()

def train_lstm(batch_size=10,time_step=5,train_begin=1,train_end=data_train_rows):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)

    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('model.ckpt')    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)

        for i in range(400):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 20==0:
                print("save model",saver.save(sess,'model.ckpt',global_step=i))
        print("model saved")

def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell=tf.contrib.rnn.BasicRNNCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def get_train_data(batch_size=10,time_step=5,train_begin=1,train_end=data_train_rows):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    #normalized_train_data=data_train
    train_x,train_y=[],[] 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:input_size]
       y=normalized_train_data[i:i+time_step,input_size,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

def get_test_data(time_step=5,test_begin=data_train_rows):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std
    #normalized_test_data = data_test
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
       y=normalized_test_data[i*time_step:(i+1)*time_step,input_size]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
  

    # for step in range(len(test_x)):
    #       print ("test_x[step]=",test_x[step])  

    return mean,std,test_x,test_y

train_lstm()
#prediction() 
