import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import models
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import math
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
n_workers = 10
np.random.seed(0)
random.seed(0)

params = {}
params["SCRIPT"] = "train_ST-Conv1D.py"
params["path_data"] = []
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/")
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-04 16-58-16/")

params["EPOCHS"] = 301
params["MODEL"] = "model_ST-Conv1D"
params["ENERGY_NORM_TYPE"] = "standardize"
params["LENGTH_SEQUENCE"] = 3
params["LENGTH_PAST"] = 0
params["MERGED_MODEL"] = True
params["LENGTH_SUM"] = 3

params["LOSS"] = 'mse'
params["LEARNING_RATE"] = 1e-4
params["BATCH_SIZE"] = 32
params["MAX_NO_IMPROVEMENT_EPOCHS"]= 50
params["FREQ_VAL"] = 1

params["TEST_PERC"] = 0.2
params["TERRAIN_IDS"] = []
#[12,8,7,0,22,5,14,13,4,9,11,10,1,3,15,16,17]

params["FRAC_TRAIN"] = 1
params["REMOVE_ROUGH"] = False
params["MAX_VAR_ROUGH"] = 2.25

params["LOG_DIR"] = "./Exp00/log_ST_conv1D/"
params["INPUT_FEATURES"] = ["wheel_trace"]
params["WHEEL_TRACE_SHAPE"] = (78,40)
params["REMOVE_CENTRAL_BAND"] = 0

params["path_sum_indices"] = []
params["path_sum_indices"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/Merged_Data/sum_indices.csv")
params["path_sum_indices"].append("../Dataset_Collection/Datasets/EExp_2021-08-04 16-58-16/Merged_Data/sum_indices.csv")


TIME_NAME = False
LOG_NAME_NAME = "conv_sum"

class DataSequence(keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    modified from: https://stackoverflow.com/questions/51843149/loading-batches-of-images-in-keras-from-pandas-dataframe
    """
    def __init__(self, df, df_indexes, batch_size,  mode='train'):
        self.df = df
        self.df_indexes = df_indexes
        self.batch_size = batch_size
        self.mode = mode  # shuffle when in train mode
    
        self.on_epoch_end()

    def __len__(self):
        # compute number of batches to yield
        if self.mode=='train':
            return int(math.ceil(len(self.df_indexes) / float(self.batch_size)))
        elif self.mode == 'validation':
            return int(math.ceil(len(self.df_indexes) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.df_indexes))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))            

    def __getitem__(self, index):        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        for b, idx in enumerate(indexes):
            k = int(self.df_indexes.iloc[idx].k)
            dataset = int(self.df_indexes.iloc[idx].dataset)
            sequence = self.df[self.df.dataset==dataset].iloc[k-params["LENGTH_PAST"]:k+params["LENGTH_SEQUENCE"]]
            
            
            xg_string = sequence.loc[:,params["INPUT_FEATURES"]].values
            for i in range(len(xg_string)):
                xg = np.array([float(v) for v in xg_string[i][0].split(' ')]).reshape(params["WHEEL_TRACE_SHAPE"])
                if not i:
                    xg_seq = np.expand_dims(xg,axis=0)
                else:
                    xg_seq = np.concatenate([xg_seq,np.expand_dims(xg,axis=0)],axis=0)
            
            if params["MERGED_MODEL"]:     
                zrel = sequence.loc[:,["zrel"]].values
                # Merge the shots in a single terrain trace
                xG = xg_seq[params["LENGTH_PAST"]]
                for p in range(params["LENGTH_PAST"]):
                    xG = np.concatenate([xg_seq[params["LENGTH_PAST"]-1-p,:16],xG],axis=0)
                for p in range(1, params["LENGTH_SEQUENCE"]):
                    xG = np.concatenate([xG,xg_seq[p,-16:]+zrel[p]-zrel[0]], axis = 0)
            else:
                xG = xg_seq 
            
            yE = np.sum(sequence.drop(sequence.iloc[:params["LENGTH_PAST"]+params["LENGTH_SEQUENCE"]-params["LENGTH_SUM"]].index)["energy"].values, axis=0)
            #yE = np.sum(sequence.iloc[params["LENGTH_PAST"]:]["energy"].values, axis = 0)
            
            if not b:
                X = np.expand_dims(xG, axis = 0)
                Y = np.expand_dims(yE, axis = 0)
            else:
                X = np.concatenate([X, np.expand_dims(xG, axis = 0)], axis = 0)
                Y = np.concatenate([Y, np.expand_dims(yE, axis = 0)], axis = 0)
        
        if params["REMOVE_CENTRAL_BAND"]:
            if params["MERGED_MODEL"]: 
                xg_l = X[:,:,:(params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2]
                xg_r = X[:,:,-(params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2:]
            else:
                xg_l = X[:,:,:,:(params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2]
                xg_r = X[:,:,:,-(params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2:]
            X = np.concatenate([xg_l,xg_r],axis=-1)
        
        
        if self.mode == "prediction":
            return X
        else:
            return X, Y
        

        
def main():
    if not params["TERRAIN_IDS"]:
        id_files = [".csv"]
    else:
        id_files = ["data_{}.csv".format(idt) for idt in params["TERRAIN_IDS"]]
    df = pd.DataFrame()
    for i, path_data in enumerate(params["path_data"]):
        files = os.listdir(path_data)
        for file in files:
            if any([p in file for p in id_files]):
                dfi = pd.read_csv(path_data+file)    
                ## Removing some of data:
                # Samples without failures
                try:
                    dfi = dfi[dfi.goal==1]
                except:
                    pass
                # Samples without initial acceleration
                dfi = dfi[dfi.segment!=0]
                try:
                    # Samples without low mean speed
                    dfi = dfi[dfi.mean_speed>0.87]
                    # Samples without low initial speed
                    dfi = dfi[dfi.initial_speed>0.88]
                except:
                    # Samples without low mean speed
                    dfi = dfi[dfi.mean_speed_long>0.87]
                    # Samples without low initial speed
                    dfi = dfi[dfi.initial_speed_long>0.88]
                if params["REMOVE_ROUGH"]:
                    # Samples without rough pitch/roll variations
                    dfi = dfi.loc[(dfi.var_pitch_est <=params["MAX_VAR_ROUGH"]) | (dfi.var_roll_est <=params["MAX_VAR_ROUGH"])]
                # dfi["std_pitch_est"] = dfi["var_pitch_est"].pow(0.5)
                # dfi["std_roll_est"] = dfi["var_roll_est"].pow(0.5)
                # dfi["curvature"] = dfi["curvature"]*100
                # dfi["curvature_tm1"] = dfi["curvature_tm1"]*100
                
                try:
                    dfi = dfi.drop(columns=['wheel_types'])
                except:
                    pass
                dfi["energy"] = dfi["energy"].clip(lower = 0.0)
                
                dfi["dataset"] = [i]*len(dfi)
                df = pd.concat([df,dfi])
    
    df_sum_indices = pd.DataFrame()
    for i, path_idx in enumerate(params["path_sum_indices"]):         
        dfi = pd.read_csv(path_idx).drop_duplicates()
        dfi["dataset"] = [i]*len(dfi)
        df_sum_indices = pd.concat([df_sum_indices,dfi], ignore_index=True)
    
    
    if not params["TERRAIN_IDS"]:
        terrain_ids = list(df["terrain_id"].drop_duplicates().values)
    else:
        terrain_ids = params["TERRAIN_IDS"]
    for terrain_id in terrain_ids:
        params["TERRAIN_ID"] = terrain_id
        dfi = df[df["terrain_id"]==params["TERRAIN_ID"]]
        dfx = df_sum_indices[df_sum_indices["terrain_id"]==params["TERRAIN_ID"]]
        
        df_train, df_val = train_test_split(dfx,test_size = params["TEST_PERC"])
        if params["FRAC_TRAIN"] < 1:
            df_train = df_train.sample(frac=params["FRAC_TRAIN"])
        
        df_y = pd.DataFrame()
        for i in range(len(params["path_sum_indices"])):
            ddfx = df_train[df_train.dataset==i]
            idx_train = []
            for k in ddfx.k.values:
                idx_train.extend(k+z for z in range(params["LENGTH_SEQUENCE"])) 
            ddfx = df_val[df_val.dataset==i]
            idx_val = []
            for k in ddfx.k.values:
                idx_val.extend(k+z for z in range(params["LENGTH_SEQUENCE"]))
            ddfi = dfi[dfi.dataset==i]
            ddf_y = ddfi.iloc[idx_train].drop_duplicates().energy
            df_y = pd.concat([df_y,ddf_y])
        
        # Standardize Energy
        if params["ENERGY_NORM_TYPE"] == 'standardize':
            mean_y = df_y.mean().values
            std_y = df_y.std().values
            dfi["energy"] = (dfi["energy"]-mean_y)/std_y
            params["energy_mean"] = mean_y
            params["energy_std"] = std_y
            del df_y
        elif params["ENERGY_NORM_TYPE"] == 'normalize':
            min_y = df_y.min().values
            max_y = df_y.max().values
            int_y = max_y-min_y
            dfi["energy"] = (dfi["energy"]-min_y)/int_y
            params["energy_min"] = min_y
            params["energy_max"] = max_y
            params["energy_int"] = int_y
            del df_y
        
        print("Samples: {}".format(len(dfi)))
        print("Train: {}".format(len(df_train)))
        print("Test: {}".format(len(df_val)))
        
        
        if TIME_NAME:
            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            LOG_NAME = "{}/".format(current_time)
        else:
            LOG_NAME = "{}_{}/".format(int(terrain_id),LOG_NAME_NAME)
        
        model = models.get_model(params, summary=True)
        
        if not os.path.exists(params["LOG_DIR"]+LOG_NAME):
            os.makedirs(params["LOG_DIR"]+LOG_NAME) 
        file = open(params["LOG_DIR"]+LOG_NAME+"log_params.txt","w+")
        for key, val in params.items():
            file.write("{}: {}\n".format(key,val))
        file.close()
        
        # Initialise data sequences and callbacks
        seq_train = DataSequence(dfi, df_train, params["BATCH_SIZE"], mode='train')
        seq_val = DataSequence(dfi, df_val, params["BATCH_SIZE"], mode='validation')
        
        del df_train, df_val
        
        callbacks = [ keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=params["MAX_NO_IMPROVEMENT_EPOCHS"]//params["FREQ_VAL"],
                                                    mode='min', verbose=1),
                      keras.callbacks.TensorBoard(log_dir=params["LOG_DIR"] + LOG_NAME,
                                                    update_freq="epoch",
                                                    histogram_freq=0),
                      keras.callbacks.CSVLogger(params["LOG_DIR"] + LOG_NAME + '\\log.csv',
                                                  append=True),
                      keras.callbacks.ModelCheckpoint(
                          params["LOG_DIR"] + LOG_NAME + 'model_best.hdf5',
                          monitor='val_loss', verbose=1, save_best_only=True,
                          save_weights_only=True, mode='auto', save_freq='epoch')]
        
        model.compile(optimizer=keras.optimizers.RMSprop(lr=params["LEARNING_RATE"]), 
                      loss=params["LOSS"], metrics=[tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))])
        model.fit(seq_train, 
                  validation_data=seq_val,
                  validation_freq=params["FREQ_VAL"],
                  use_multiprocessing=bool(max(0,n_workers-1)),
                  workers=n_workers,
                  epochs=params["EPOCHS"], callbacks=callbacks)
    


if __name__ == "__main__":
    main()
