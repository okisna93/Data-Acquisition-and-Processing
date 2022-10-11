import pandas as pd

# df = pd.read_csv("OUTPUT6.txt", sep='|',header=None ,names=['UserID','TrackId','AlbumId','ArtistId','Track','Genre1'])
# #dataframe.columns = ['TrackId','AlbumId','ArtistId','Optional GenreId_1']
# # dataframe.to_csv("OUTPUT_3.csv",index=None)

# df["UserID"] = df["UserID"].astype(str)
# df["TrackID"] = df["TrackId"].astype(str)
# df2 = pd.DataFrame(columns = ['TrackId',"Predictor"])

# # print(df["UserID"])
# # print(df["TrackID"])
# df2["TrackId"]=df["UserID"] + "_" + df["TrackID"]
# df2["Predictor"] = df["AlbumId"]+df["ArtistId"]+df["Track"]+df["Genre1"]

# # # print(df["Predictor1"])
# # # df["Predictor1"] = df["Predictor1"].astype(int)
# # # df2["Predictor"]= (df["Predictor"]+df["Predictor1"])/2
# # # print(df2)

# df2.to_csv("OUTPUT6.csv",index=None) 

data=pd.read_csv('82_features.csv')
data
