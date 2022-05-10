import pandas as pd
import random
import numpy as np
import os

np.random.seed(0)
random.seed(0)

path_data = "./Datasets/Exp_2022-04-09 10-04-58/"
#path_data = "./Datasets/Exp_2021-08-04 16-58-16/"

if not os.path.exists(path_data+"Merged_Data/"):
    os.mkdir(path_data+"Merged_Data/")
    
path_output_file = path_data + "Merged_Data/" + "sum_indices.csv"

N_ITERATIONS = 1
LENGTH_SEQUENCE = 3
LENGTH_PAST = 0

    
def main():
    id_files = [".csv"]
    df = pd.DataFrame()
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
                
                dfi["mean_speed_long"] = dfi.mean_speed
                dfi["initial_speed_long"] = dfi.initial_speed
                
            except:
                # Samples without low mean speed
                dfi = dfi[dfi.mean_speed_long>0.87]
                # Samples without low initial speed
                dfi = dfi[dfi.initial_speed_long>0.88]
            try:
                dfi = dfi.drop(columns=['wheel_types'])
            except:
                pass
            dfi["energy"] = dfi["energy"].clip(lower = 0.0)
            df = pd.concat([df,dfi])
        
    data = []
    terrain_ids = list(df["terrain_id"].drop_duplicates().values)
    for terrain_id in terrain_ids:
        dfi = df[df["terrain_id"]==terrain_id]
        iterations = int(len(dfi)*N_ITERATIONS)
        for i in range(iterations):
            if not i%100:
                print("Terrain id: {} --> {}%".format(terrain_id,i/iterations*100))
            while True:
                k = random.sample(list(range(len(dfi)-LENGTH_SEQUENCE+1)),1)[0]
                flag = False
                for j in range(LENGTH_PAST):
                    if dfi.iloc[k].segment != dfi.iloc[k-(j+1)].segment + j +1:
                        flag = True
                        break
                if flag:
                    continue
                for j in range(1, LENGTH_SEQUENCE):
                    if dfi.iloc[k].segment != dfi.iloc[k+j].segment-j:
                        flag = True
                        break
                if not flag:
                    data.append({"terrain_id": terrain_id, "k": k})
                    break
        pd.DataFrame(data).drop_duplicates().to_csv(path_output_file, index=False)
    
                
        
    


if __name__ == "__main__":
    main()

