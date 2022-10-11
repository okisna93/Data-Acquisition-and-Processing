clear all

data=csvread('Output.csv',0);%read the data csv file
Cols=data(:,1);%grab the only column of data

sum=0;

for i=1:1:120000
    sum=sum+Cols(i);%sum the users 6 scores
    if(mod(i,6)==0)
        thresh(i/6)=sum/6;%after getting the 6th score get the average
        sum=0;%reset the sum
    end
end

j=1;
for i=1:1:120000
    if Cols(i)>thresh(j) %if the score is above the thresh then its a 1
        Output(:,i)=1;
    else %below its a 0
        Output(:,i)=0;
    end
    
    if(mod(i,6)==0)%iterate the thresh index after each user
        j=j+1;
    end
    
end

Output=rot90(Output,3);%rotate the output so it is vertical rather than horizontal


