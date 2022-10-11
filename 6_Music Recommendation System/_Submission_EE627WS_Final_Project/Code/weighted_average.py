#standard imports used for data manipulation etc
import numpy
import pandas as pd   

#read the csv file in 
df = pd.read_csv("output2_copy.csv")
df["UserID"] = df["UserID"].astype(str)
df["TrackID"] = df["TrackID"].astype(str)

#Make empty dataframe with columns = 'TrackID',"Predictor"
df2 = pd.DataFrame(columns = ['TrackID',"Predictor"])
# print(df2)  
# print(df["UserID"])
# print(df["TrackID"])
df2["TrackID"]=df["UserID"] + "_" + df["TrackID"]

df["Predictor"] = df["Predictor"].astype(int)
# print(df["Predictor1"])
df["Predictor1"] = df["Predictor1"].astype(int)

#weights are values that can be adjusted based on user preference
weight1 = 0.95
weight2 = 0.15 

#df will update it predictor values based on weight applied to df Predictor and Predictor1 values
df2["Predictor"]= (weight1*df["Predictor"]+weight2*df["Predictor1"])/2
print(df2)

#Will produce a csv file with the weighted averages    
df2.to_csv("0.95_0.15_output14.csv",index=None)    