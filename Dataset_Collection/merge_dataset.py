import pandas as pd
import os



path_data = "./Datasets/Exp_2022-04-09 10-04-58/"
#path_data = "./Datasets/Exp_2021-08-04 16-58-16/"

path_dataset_plane = path_data + "Merged_Data/" + "plane_data.csv"

if not os.path.exists(path_data+"Merged_Data/"):
    os.mkdir(path_data+"Merged_Data/")
    
df = pd.DataFrame()
files = os.listdir(path_data)
for file in files:
    if ".csv" in file:
        dfi = pd.read_csv(path_data+file)
        df = pd.concat([df,dfi], ignore_index=True)
        
try:
    df = df.drop(["wheel_trace"], axis = 1)
except:
    pass
try:
    df = df.drop(["wheel_types"], axis = 1)
except:
    pass
df.to_csv(path_dataset_plane)

        
