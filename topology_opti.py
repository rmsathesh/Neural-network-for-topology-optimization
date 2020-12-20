import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, BatchNormalization, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import h5py
import math
import glob
#%%
import os
file_loc = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_loc)

#%%
def create_model():
    inputs = Input(shape=(40, 40, 2), name='in_tensor')
    
    #block 1
    conv2d_1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv2d_2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv2d_1)
    maxpool_1 = MaxPool2D(pool_size=(2, 2))(conv2d_2)
    
    #block 2
    conv2d_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')( maxpool_1)
    conv2d_4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2d_3)
    maxpool_2 = MaxPool2D(pool_size=(2, 2))(conv2d_4)

    #block 3
    conv2d_5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')( maxpool_2)
    conv2d_6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2d_5)
    maxpool_3 = MaxPool2D(pool_size=(2, 2))(conv2d_6)

    #block 4
    conv2d_7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')( maxpool_3)
    conv2d_8 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2d_7)
    
    #block 5_upsample
    up_1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv2d_8)
    concat_1 = concatenate([up_1, conv2d_6], axis=3)
    conv2d_9 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat_1)
    
    #block 6_upsample
    up_2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv2d_9)
    concat_2 = concatenate([up_2, conv2d_4], axis=3)
    conv2d_10 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat_2)

    #block 7_upsample
    up_3 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv2d_10)
    concat_3 = concatenate([up_3, conv2d_2], axis=3)
    conv2d_11 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(concat_3)
    
    out = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv2d_11)
    
    model = Model(inputs, out)
    
    return model
#%%
def custom_loss(ytrue, ypred):
    loss_1 = tf.losses.log_loss(ytrue, ypred, reduction=tf.losses.Reduction.MEAN)
    loss_2 = tf.reduce_mean(tf.square(tf.reduce_mean(ytrue-ypred, axis=(1, 2, 3))))
    return loss_1 + loss_2
       
#%%
def data_aug(x, y):
    batch_size = x.shape[0]
    
    #flip left_right
    idx = np.random.random(batch_size)>0.5
    x[idx] = x[idx,:,::-1]
    y[idx] = y[idx,:,::-1]
    
    #flip top_down
    idx = np.random.random(batch_size)>0.5
    x[idx] = x[idx,::-1]
    y[idx] = y[idx,::-1]
    
    #90deg rotation
    idx = np.random.random(batch_size)>0.5
    x[idx] = np.swapaxes(x[idx], 1, 2)
    y[idx] = np.swapaxes(y[idx], 1, 2)
    
    return x, y
    

def generate_train(batch_size=32, file=None, idx=None,n_iter=5):
    while True:
        train_count = len(idx)
        batches = int(train_count / batch_size)
        remain = train_count % batch_size
        if remain:
            batches += 1
        for i in range(0, batches):
            if i==batches-1:
                batch_idx = idx[i*batch_size:]
            else:
                batch_idx = idx[i*batch_size:i*batch_size + batch_size]
            batch_idx = sorted(batch_idx)
            
            x1 = file['inputs'][batch_idx,:,:,n_iter]
            x2 = file['inputs'][batch_idx,:,:,n_iter-1]
            x = np.stack((x1, x1-x2), axis=-1)
            y = file['targets'][batch_idx]
            x, y = np.array(x), np.array(y)
            x, y = data_aug(x, y)
            yield x, y
            
def generate_valid(batch_size=32, file=None, idx=None, n_iter=5):
    while True:
        train_count = len(idx)
        batches = int(train_count / batch_size)
        remain = train_count % batch_size
        if remain:
            batches += 1
        for i in range(0, batches):
            if i==batches-1:
                batch_idx = idx[i*batch_size:]
            else:
                batch_idx = idx[i*batch_size:i*batch_size + batch_size]
            batch_idx = sorted(batch_idx)
            
            x1 = file['inputs'][batch_idx,:,:,n_iter]
            x2 = file['inputs'][batch_idx,:,:,n_iter-1]
            x = np.stack((x1, x1-x2), axis=-1)
            y = file['targets'][batch_idx]
            yield np.array(x), np.array(y)  
            
#%%
def train_model(model, file_name, validation_ratio=0.2, batch_size=32):
    with h5py.File(file_name, 'r') as hfd:
        n_samples = hfd['inputs'].shape[0]
        sample_id = np.arange(n_samples)
        np.random.shuffle(sample_id)
        train_id = sample_id[0:int((1 - validation_ratio)*n_samples)]
        valid_id = sample_id[int((1 - validation_ratio)*n_samples):]
        train_generator = generate_train(batch_size=batch_size, file=hfd, idx=train_id, n_iter=10)
        valid_generator = generate_valid(batch_size=batch_size, file=hfd, idx=valid_id, n_iter=10)
        
        steps = math.ceil(len(train_id) / batch_size)
        val_steps = math.ceil(len(valid_id) / batch_size)
        
        model.fit_generator(generator=train_generator, validation_data=valid_generator,
                            steps_per_epoch=steps, validation_steps=val_steps, epochs=10)
        
#%%
model = create_model()
print(model.summary())
opti = Adam(learning_rate=0.001)
model.compile(optimizer=opti, loss=custom_loss, metrics=['accuracy'])
file_name = 'dataset.h5'

#%%
train_model(model, file_name)

#%%
model.save('topo_opti_Unet_3.h5')

#%%

def evaluate(model, n_samples=10, f_loc=None, n_iter=5):
    
    y_pred = []
    y_act = []
    k=21
    for c in random.sample(glob.glob(f_loc + '/*'), n_samples):
        temp = np.load(c)['arr_0']
        temp = temp.transpose((1, 2, 0))
        x1 = temp[:,:,n_iter]
        x2 = temp[:,:,n_iter-1]
        x = np.stack((x1, x1-x2), axis=-1)
        x = np.expand_dims(x, 0)
        pred = np.round(model.predict(x)[0])
        y_pred.append(pred)
        thres = np.mean(temp, axis=(0, 1))
        y = (temp>thres).astype('float32')[:,:,[-1]]
        y_act.append(y)
        yplot = np.concatenate([y[:,:,0], pred[:,:,0], np.absolute(y[:,:,0]-pred[:,:,0])], axis=1)
        plt.imshow(yplot)
        img_name = 'results_2/image_' + str(k)
        k+=1
        plt.savefig(img_name)
        plt.show()
    y_pred = np.array(y_pred)
    y_act = np.array(y_act)
    
    return y_act, y_pred
    
        
#%%
floc = r'dataset'
val = evaluate(model, n_samples=20,f_loc=floc, n_iter=10)

#%%
      
            
     
            
        


    

    
