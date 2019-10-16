# ML2019
## Team: Yann Meier, Kopiga Rasiah, Rayan Daod Nathoo
### Deadline: October 28th, 2019
Open project1_description.pdf for more informations

###### 1. Raw data.
We started by predicting the model on raw data using least squares. The accuracy was 0.745.
###### 2. Data grouping according to the undefined feature (split_in_group_1).
We separated the data into 6 groups, where each group have the same number of features that are undefined (that have the value -999). We droped constant features, as they dont give any further information for predicting a model. Note that we do not use standardization and wont use it unless specified, as it gave a less accurate accuray. The regression used was again LS and we got an accuracy of 0.757 \o/.
###### 3. Data grouping according to the feature PRI_JET_NUM. (split_in_group_2).
We noticed that the feature PRI_JET_NUM took four values : {0,1,2,3}. Therefore we grouped the data according to this variable, which gave us (obviously) four groups. Accuracy was at 0.75.
###### 4. Feature expansion.
We augmented the features by adding x to the power of **up to** a given degree, for x: a feature of the data. Adding only the square of each feature gave a better result. Accuracy with (split_in_group_1) : 0.785. Accuracy with (split_in_group_2) : 0.771. 
##### 5. Next work.
We need to test all above with other type of regression.
We can augment the feature according to some combination of features, or only add feature to the power of a degree

