import numpy
import pandas as pd

#Reading the data and creating output file to save our results
#dataDir='C:/Users/justi/Desktop/EE-627-WS1-Final_Project/data_in_matrixForm'
file_name_test='testTrack_hierarchy.txt'
file_name_train='trainIdx2_matrix.txt'
output_file= 'OUTPUT11.txt' 

fTest= open(file_name_test, 'r')
fTrain=open(file_name_train, 'r')
Trainline= fTrain.readline()
fOut = open(output_file, 'w')

# vectors that contains trackID, albumID, artistID, Genres and lastUserID
trackID_vec=[0]*6
albumID_vec=[0]*6
artistID_vec=[0]*6
genreID_vec1=[0]*6
genreID_vec2=[0]*6
genreID_vec3=[0]*6
genreID_vec4=[0]*6
genreID_vec5=[0]*6
genreID_vec6=[0]*6
lastUserID=-1

user_rating_inTrain=numpy.zeros(shape=(6,9))

for line in fTest:
    #Extracting the rates from the data
    arr_test=line.strip().split('|')
    userID= arr_test[0]
    trackID= arr_test[1]
    albumID= arr_test[2]
    artistID=arr_test[3]

    if len(arr_test)>4:
        genreID1=arr_test[4]
    else:
        genreID1=0

    if len(arr_test)>5:
        genreID2=arr_test[5]
    else:
        genreID2=0

    if len(arr_test)>6:
        genreID3=arr_test[6]
    else:
        genreID3=0

    if len(arr_test)>7:
        genreID4=arr_test[7]
    else:
        genreID4=0

    if len(arr_test)>8:
        genreID5=arr_test[8]
    else:
        genreID5=0

    if len(arr_test)>9:
        genreID6=arr_test[9]
    else:
        genreID6=0

    if userID!= lastUserID:
        ii=0
        user_rating_inTrain=numpy.zeros(shape=(6,9))

    # Creating a path to save the rates for assigned vector

    trackID_vec[ii]=trackID
    albumID_vec[ii]=albumID
    artistID_vec[ii]=artistID
    genreID_vec1[ii]=genreID1
    genreID_vec2[ii]=genreID2
    genreID_vec3[ii]=genreID3
    genreID_vec4[ii]=genreID4
    genreID_vec5[ii]=genreID5
    genreID_vec6[ii]=genreID6
	

    # Loop for saving the rates to assigned empty vectors    
    ii=ii+1
    lastUserID=userID
    if ii==6 : #
#         while (Trainline):
        for Trainline in fTrain:
#             Trainline=fTrain.readline()
            arr_train = Trainline.strip().split('|')
            trainUserID=arr_train[0]
            trainItemID=arr_train[1]
            trainRating=arr_train[2]
            

            if trainUserID< userID:
                continue
                
            if trainUserID== userID:
                for nn in range(0, 6):
                    if trainItemID==albumID_vec[nn]:
                        user_rating_inTrain[nn, 0]=trainRating
                    if trainItemID==artistID_vec[nn]:
                        user_rating_inTrain[nn, 1]=trainRating
                    if trainItemID==trackID_vec[nn]:
                        user_rating_inTrain[nn,2]=trainRating
                    if trainItemID==genreID_vec1[nn]:
                        user_rating_inTrain[nn,3]=trainRating
                    if trainItemID==genreID_vec2[nn]:
                        user_rating_inTrain[nn,4]=trainRating
                    if trainItemID==genreID_vec3[nn]:
                        user_rating_inTrain[nn,5]=trainRating
                    if trainItemID==genreID_vec4[nn]:
                        user_rating_inTrain[nn,6]=trainRating
                    if trainItemID==genreID_vec5[nn]:
                        user_rating_inTrain[nn,7]=trainRating
                    if trainItemID==genreID_vec6[nn]:
                        user_rating_inTrain[nn,8]=trainRating

            if trainUserID> userID:
                # Writing the rates to the output file we created earlier
                for nn in range(0, 6):
                   # genretotal=user_rating_inTrain[nn,3]+user_rating_inTrain[nn,4]+user_rating_inTrain[nn,5]+user_rating_inTrain[nn,6]+user_rating_inTrain[nn,7]+user_rating_inTrain[nn,8]
                    outStr=str(userID) + '|' + str(trackID_vec[nn])+ '|' + str(user_rating_inTrain[nn,0]) + '|' + str(user_rating_inTrain[nn, 1])+'|' + str(user_rating_inTrain[nn, 2])+'|' +str(user_rating_inTrain[nn,3])+'|' +str(user_rating_inTrain[nn,4])+'|' +str(user_rating_inTrain[nn,5])+'|' +str(user_rating_inTrain[nn,6])+'|' +str(user_rating_inTrain[nn,7])+'|' +str(user_rating_inTrain[nn,8])
                    fOut.write(outStr + '\n')
                break
fTest.close()
fTrain.close()

#Turning Output file to the .csv file to use in our other applications
df = pd.read_csv("OUTPUT11.txt", sep='|',header=None ,names=['UserID','TrackId','AlbumId','ArtistId','Track','Genre1','Genre2','Genre3','Genre4','Genre5','Genre6'])
df.to_csv("8features.csv")
#dataframe.columns = ['TrackId','AlbumId','ArtistId','Optional GenreId_1']
# dataframe.to_csv("OUTPUT_3.csv",index=None)

# df["UserID"] = df["UserID"].astype(str)
# df["TrackID"] = df["TrackId"].astype(str)
# df2 = pd.DataFrame(columns = ['TrackId',"Predictor"])

# print(df["UserID"])
# print(df["TrackID"])
#df2["TrackId"]=df["UserID"] + "_" + df["TrackID"]
#df2["Predictor"] = df["AlbumId"]+df["ArtistId"]+df["Track"]+df["Genre1"]+df['Genre2']+df["Genre3"]+df['Genre4']+df["Genre5"]+df['Genre6']+df["Genre7"]+df['Genre8']

# # print(df["Predictor1"])
# # df["Predictor1"] = df["Predictor1"].astype(int)
# # df2["Predictor"]= (df["Predictor"]+df["Predictor1"])/2
# # print(df2)

#df2.to_csv("OUTPUT6.csv",index=None) 