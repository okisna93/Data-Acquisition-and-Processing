clear all

data=csvread('82_features.csv',1.0);%read the csv, provided by the team
Cols=data(:,[4 5 6 7 8 9 10 11]);%grab the columns with predictor data
[coeff,score,latent] = pca(Cols);%use the PCA function to get the coeff score and latent
figure(1);
bar(latent)%bar graph of the component contributions
xlabel('Principal Component');
ylabel('Value');

figure(2);
labels = {'Component 1','Component 2','Component 3','Component 4','Component 5','Component 6','Component 7','Component 8'};
pie(latent)%create a pie chart of the component contributions
legend(labels,'Location','south','Orientation','horizontal');

figure(3)
scatter(score(:,[1]),score(:,[2]))%create a scatter plot of the scores

for i=4:1:11
    
    weights(i-3)=latent(i-3)/sum(latent);%calculate the weights
    
end

for i=1:1:8
    
    Cols(:,i)=Cols(:,i)*weights(i);%multiply the values by their weights

end

for i=1:1:120000
    
    Output(i)=sum(Cols(i,:));% sum the values in each row
    
end
Output=rot90(Output,3);%rotate the output array so that it is vertical rather than horizontal

csvwrite('Output.csv', Output);%write the outputs to a csv file, append the predictor values to the track_id afterwards
