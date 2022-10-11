#standard imports used for data manipulation etc
import pandas as pd          
import numpy as np               

#Manually had to add the names of the files with respected location
#These files are the one that were hande into Kaggle and had Kaggle score with them
files =['Best_Data/decision_tree_0.82108.csv', 
    'Best_Data/gradient_boosted_tree_0.84398.csv',
    'Best_Data/logical_regression_0.84507.csv',
    'Best_Data/OUTPUT_1_0.84991.csv',
    'Best_Data/OUTPUT_3_0.83257.csv',
    'Best_Data/OUTPUT_5_0.80648.csv',
    'Best_Data/OUTPUT_11_0.79575.csv',    
    'Best_Data/OUTPUT_14_0.78478.csv',  
    'Best_Data/random_forest_1_0.82108.csv',
    'Best_Data/pca_weighted_output_3_0.86259.csv',
    'Best_Data/pca_weighted_output_4_0.86679.csv' 
    ]

#These are the public score found on the Kaggle Competition 
scores = [0.82108,0.84398,0.84507,0.84991,0.83257,
0.80648,0.79575,0.78478,0.82108,0.86259,0.86679]             
#This will be a dataframe that will contain all the predictor scores from the files seen in line 7-18  
all_df = []

#The code commented out below was used to double check if certain files works or did not worked: 
# print("All Data is being written")
# new_result = pd.read_csv('Best_Data/random_forest_1_0.82108.csv', names=['TrackID', 'Predictor'], dtype={1:np.int64}, header=0)    
# print("All Data is done being written")
# print(new_result)
   
#pred is used so all the columns with predictor score are not called the same thing
pred = 0    

#This code will run through all the result files and 
# add content so the written file will look like this
# User_ID Predictor0 Predictor1 ... Predictorx  
for result in files:      
    # print(result)  
    columns = ['TrackID', 'Predictor' + str(pred)]    
    new_result = pd.read_csv(result, names=columns, dtype={1:np.int64}, header=0)    

    try:
        # all of the predictor values will he joined into one file
        all_df = all_df.join(new_result.set_index('TrackID'), on='TrackID')
    except:
        # The first predictor value will lead to it being read
        all_df = pd.read_csv(result)  
    
    #Need to increase this number so not column says Predictor0 
    pred += 1    

#This will turn add_df into a csv file that can be reference for later use    
all_df.to_csv("Best_Data/all_data_11_values.csv", index=False) 

#This will create a matrix with Predictor first column 
S = np.array((all_df.iloc[:, 1] * 2 - 1)) 
# print(S)

#This is done to update matrix S with all Predictor columns 
#Values will be -1 and 1 now
for i in range(2,all_df.shape[1]):
    S = np.c_[S, (all_df.iloc[:, i] * 2 - 1)]

#Use shape to make sure the right amount of values present 
# print(S.shape)      

#Will give the length of matrix S
length = len(S)

#The rest of the code from lines 72-86 
#will follow the math explained in class -> Lecture EE627A_ensemble.pdf slides:13-16

St_x = []

St_x = [length * (2*J-1) for J in scores]

St_s = np.dot(S.T, S).astype('float') + np.eye(S.shape[1]) * (10 ** -6)
   
St_s_inv = np.linalg.inv(St_s)
# print(St_s_inv)

a_LS = np.dot(St_s_inv,St_x)
# print(a_LS)    

S_ensemble = np.dot(S,a_LS)
# print(S_ensemble)  
S_length = len(S_ensemble)
# print(S_ensemble_len)

final_predictions = np.zeros(S_length)

#This loop will get top 3 tracks 
for j in range(S_length // 6):   
    # The threshold will be third element in array 
    threshold = np.sort(S_ensemble[j * 6 : j * 6 + 6])[2]    # sort the 6 values for each user and grab the third element
    for user in range(6):
        if S_ensemble[j * 6 + user] > threshold:  
            final_predictions[j * 6 + user] = 1

final_df = pd.DataFrame(all_df.iloc[:,0]) 
final_df['Predictor'] = np.array(final_predictions, dtype=int)
# print(final_df)   

print("Ensembling Data is being written")
final_df.to_csv('Best_Data/Ensemble_Predictions_Test_11_values.csv', index=False)
print("Ensembling Data is done being written")                                      