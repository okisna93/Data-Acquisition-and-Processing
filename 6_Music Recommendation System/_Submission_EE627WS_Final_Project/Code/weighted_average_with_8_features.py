#standard imports used for data manipulation etc
#This file does the same thing as weighted_average.py but it had more features present
import numpy   
import pandas as pd 
 
#uses the 82 features present
df = pd.read_csv("82_features.csv")
df["UserID"] = df["UserID"].astype(str)
df["TrackID"] = df["TrackID"].astype(str)
df["AlbumId"] = df["AlbumId"].astype(str)
df["Genre1"] = df["Genre1"].astype(str)
df["Genre2"] = df["Genre2"].astype(str)
df["Genre3"] = df["Genre3"].astype(str)
df["Genre4"] = df["Genre4"].astype(str)
df["Genre5"] = df["Genre5"].astype(str)
df["Genre6"] = df["Genre6"].astype(str)

df2 = pd.DataFrame(columns = ['UserID',"TrackId"])
# print(df2)  
# print(df["UserID"])
# print(df["TrackID"])
df2["TrackID"]=df["UserID"] + "_" + df["TrackID"]

df["Predictor"] = df["Predictor"].astype(int)
# print(df["Predictor1"])
df["Predictor1"] = df["Predictor1"].astype(int)
weight1 = 0.95
weight2 = 0.15 

df2["Predictor"]= (weight1*df["Predictor"]+weight2*df["Predictor1"])/2
print(df2)
 
df2.to_csv("0.95_0.15_output14.csv",index=None)    