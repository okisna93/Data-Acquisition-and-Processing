import numpy
import pandas as pd

df = pd.read_csv("output2_copy.csv")
df["UserID"] = df["UserID"].astype(str)
df["TrackID"] = df["TrackID"].astype(str)
df2 = pd.DataFrame(columns = ['TrackID',"Predictor"])
# print(df2)  
# print(df["UserID"])
# print(df["TrackID"])
df2["TrackID"]=df["UserID"] + "_" + df["TrackID"]

df["Predictor"] = df["Predictor"].astype(int)
# print(df["Predictor1"])
df["Predictor1"] = df["Predictor1"].astype(int)
df2["Predictor"]= (df["Predictor"]+df["Predictor1"])/2
print(df2)

df2.to_csv("output3.csv",index=None)  



# print(df['UserID'])
# print(df['TrackID'].values)
# average = (df['Predictor'].values+df['Predictor2'].value)/2
# print(average)  

# print(df)
# print(df["UserID"])



# df2 = pd.DataFrame(str(userid_albumid["UserID"])+"_"+str(userid_albumid["TrackID"]))

# str(df["UserID"]) + "_" + str(df["TrackID"])
# print(userid_albumid)

# data = pd.DataFrame(["TrackId","Predictor"])
# print(data)


# print()
# average = []
# for d in dataframe: 
#     # average = (int(d[1])+int(d[3]))/2 
#     print(dataframe["Predictor"].values)

# print(average)



