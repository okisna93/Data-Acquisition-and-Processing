#install and get all of the necessary files/api's that are being used within the program
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!wget -q https://downloads.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz 

!tar -xvf spark-3.2.1-bin-hadoop3.2.tgz

!pip install pyspark
!pip install -q findspark

#import all neccessary libraries
from pyspark.sql import SparkSession #Establishes Dataframe and SQL functionality
from pyspark.sql.functions import col  #used to grab data based upon column names
from pyspark.sql.functions import lit  #used to add new columns to the dataframe
from pyspark.ml.classification import LogisticRegression  #classification models
from pyspark.ml.feature import StringIndexer, VectorAssembler #helps to prepare the data
from pyspark.ml import Pipeline       
import pandas as pd #used to apply dataframes
import numpy as np #used for arrays/martices
import matplotlib.pyplot as plt #used for graphing data

#read in the data from our csv files and create a dataframe
truth_cols = ['userID','trackID','ground_truth']
truth_df = pd.read_csv('3_new_data_for_HW9.txt',sep='|',names=truth_cols)
truth_df

cols_of_scores = ['userID','trackID',"album_score","artist_score"]
scores_df = pd.read_csv('4_output1.txt',sep='|',names=cols_of_scores)
scores_df

rating_df = truth_df.merge(scores_df, on=["userID","trackID"]).fillna(0)#merge the files
rating_df

rating_df.to_csv("5_scores.csv",index=None)

spark = SparkSession.builder.appName("hw9").getOrCreate() #initialize the spark session

spark

ratings_df = spark.read.csv("5_scores.csv", header=True, inferSchema=True) #grab the data type of each column
ratings_df

ratings_df.count()

col_ratings = ratings_df.columns
col_ratings

pd.DataFrame(ratings_df.take(6000),columns=col_ratings).groupby('ground_truth').count()

ratings_df.printSchema()

ratings_df

ratings_df = ratings_df.withColumn('ground_truth',ratings_df['ground_truth'].cast('string')) # cast values to string so we can use StringIndexer()

ratings_df

features=['album_score','artist_score']
stages=[]
inputs = features 

Vassembler = VectorAssembler(inputCols=inputs,outputCol='features')
stages+=[Vassembler]

stages

label_column = 'ground_truth'
label_string = StringIndexer(inputCol=label_column,outputCol="label")
stages+=[label_string]

pipeline = Pipeline(stages=stages)#initialize the pipeline
pmodel = pipeline.fit(ratings_df)#fit the model
train_df = pmodel.transform(ratings_df)#transform the input dataframe with the model

sel_col = ['label','features'] + col_ratings 
train_df = train_df.select(sel_col)
train_df.printSchema()

pd.DataFrame(train_df.take(5),columns=train_df.columns).transpose()#display first 5 lines of data

train_df, test_df = train_df.randomSplit([0.7,0.3],seed=2022)#train the data with a 70:30 split

print("training dataset count: " + str(train_df.count()))
print("test dataset count: " + str(test_df.count()))

prediction_df = spark.read.csv('4_output1.txt', sep='|', inferSchema=True)

prediction_df.count()

prediction_df = prediction_df.withColumnRenamed("_c0", "userID").withColumnRenamed("_c1", "trackID").withColumnRenamed("_c2", "albumScore").withColumnRenamed("_c3", "artistScore")

prediction_columns = prediction_df.columns
prediction_columns

prediction_df = prediction_df.withColumn('prediction', lit('0'))

pd.DataFrame(prediction_df.take(5), columns=prediction_df.columns).transpose()

prediction_df.printSchema()

feature_col = ['albumScore', 'artistScore']
stages = []
assembler_inputs = feature_col
assem = VectorAssembler(inputCols=assembler_inputs, outputCol='features')    # merges multiple columns into a vector column
stages += [assem]

label_col = 'prediction'
label_string_idx = StringIndexer(inputCol=label_col, outputCol='label')
stages += [label_string_idx]

pred_pipeline = Pipeline(stages=stages)#initialize the pipeline                       
pred_pipeline_model = pred_pipeline.fit(prediction_df)#fit the model
prediction_df = pred_pipeline_model.transform(prediction_df)#transform the input dataframe with the model

selected_col = ['label', 'features'] + prediction_columns
prediction_df = prediction_df.select(selected_col)
prediction_df.printSchema()

pd.DataFrame(prediction_df.take(5), columns=prediction_df.columns).transpose()

logreg = LogisticRegression(featuresCol='features',labelCol='label',maxIter=100)#initialize the logistic regression model
logreg_model = logreg.fit(train_df) #fit the training data to the model
logreg_model.coefficients

beta_coeff = np.sort(logreg_model.coefficients)#sort a copy of the array
beta_coeff

training_sum = logreg_model.summary

roc = training_sum.roc.toPandas()

roc
#plot of the ROC curve
plt.plot(roc.FPR,roc.TPR)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Logistic Regression')
plt.show()
print("Training Set AUC = " + str(training_sum.areaUnderROC))

pred = logreg_model.transform(test_df)    # transform test_df with logistic regression model
pred.select('userID', 'trackID', 'label', 'probability', 'rawPrediction', 'prediction').show(12)

sort_preds = pred.select('userID', 'trackID', 'label', 'probability', 'rawPrediction', 'prediction').sort(col('userID').asc(), col('probability').desc())
sort_preds.show(6)    

logreg_preds = logreg_model.transform(prediction_df)    # transform prediction_df with logistic regression model
logreg_preds.select('userID', 'trackID', 'probability', 'rawPrediction', 'prediction').show(12)

sort_logreg_preds = logreg_preds.select('userID', 'trackID', 'probability', 'rawPrediction', 'prediction').sort(col('userID').asc(), col('probability').desc())
sort_logreg_preds.show(6)

pd_sort_logreg_preds = sort_logreg_preds.toPandas().fillna(0.0)    # create a pandas df

pd_sort_logreg_preds

columns_to_write = ['userID', 'trackID']
pd_sort_logreg_preds.to_csv('logisticregression.csv', index=False, header=None, columns=columns_to_write)    # write to csv (without headers)

f_logreg_pred = open('logisticregression.csv')                       
f_logreg_final_preds = open('lr_final_predictions.csv', 'w')

# Initialize some values
user_id = -1
track_id = [0] * 6  

# Go through each line of the predictions file
for line in f_logreg_pred:
    arr_out = line.strip().split(',')     
    user_id_out = arr_out[0]             
    track_id_out = arr_out[1]            
    
    if user_id_out != user_id:             
        i = 0                                 
        
    track_id[i] = track_id_out         
        
    i = i + 1                    
    user_id = user_id_out   
    
    if i == 6:                               
        predictions = np.ones(shape=(6)) # initialize numpy array 
        for index in range(0, 3):            
            predictions[index] = 0           # set first 3 values 
        
        #final predictions for the 6 track predictions 
        for ii in range(0, 6):         
            out_str = str(user_id_out) + '_' + str(track_id[ii]) + ',' + str(int(predictions[ii]))
            f_logreg_final_preds.write(out_str + '\n')

f_logreg_pred.close()                     
f_logreg_final_preds.close()  