# house-price-predictor

![Opening Image](https://github.com/gerardoanuglo/house-price-predictor/blob/main/images/opening_image.jpeg)

## Introduction

Objective: Create a model to accurately predict house prices in Ames Iowa based on the features most important in gauging house prices.  
  
Data Source: [Kaggle's](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) House Prices Competition Dataset
  
The data already came seperated into training and testing sets. Each dataset has 80 features, but the training dataset has an extra column with the target variable. A total of 2,919 houses spilt evenly into each dataset. 

## Exploring the Data

First, I checked the correlation of features I expected to have a strong correlation with the target variable ("SalePrice"). Features such as LotArea, Neighborhood, HouseStyle, OverallQual, YearBuilt, KitchenQual, TotRmsAbvGrd, GrLivArea, GarageCars, and YearBuilt.

OverallQual, KitchenQual, and GrLivArea showed a strong positive correlation with Salesprice. The other features mentioned above have a weak to no correlation to sales price.

<img width="405" alt="Screen Shot 2022-11-08 at 6 11 06 PM" src="https://user-images.githubusercontent.com/85320743/200719964-5bb6b566-e00a-4434-ab26-129460c6117d.png">

<img width="405" alt="Screen Shot 2022-11-08 at 6 11 28 PM" src="https://user-images.githubusercontent.com/85320743/200720026-0fa940ad-ddb6-4bcd-8fb0-efcf7effe528.png">

<img width="408" alt="Screen Shot 2022-11-08 at 6 11 47 PM" src="https://user-images.githubusercontent.com/85320743/200720088-51a5d48b-b782-46f4-9bb2-2bf74c8b4c1b.png">

There are 4 data points I'm concerned with. They are the houses with a GrLiveArea above 4000 sqft. The two with low sales price are outliers as they have a abnormally highly amount of square footage with a low sale price. I will be removing these outliers later. 

<img width="403" alt="Screen Shot 2022-11-08 at 6 12 15 PM" src="https://user-images.githubusercontent.com/85320743/200720170-ceac89f4-782f-4996-ba25-535757fa35c3.png">

"Total Rooms Above Ground" has a weak positive correlation to SalePrice. Rooms include bedrooms, kitchen rooms, dinning rooms, living rooms, and even laundry rooms. I find this weak correlation surprising because in my experience the houses with more rooms tend to have a higher selling price. This is the case for many houses, but as seen in the graph above there are houses which don't follow this trend. It can be for poor condition of rooms, size of rooms, or total square footage.

Now lets check the correlation among all features. This step is important because it will identify other features highly correlated to SalePrice that we didn't initially expect to be important to SalePrice. This correlation matrix can also show us features highly correlated to one another. For these cases I will remove the redundant features and keep the features that best capture the specific characteristic of the house. 

<img width="669" alt="Screen Shot 2022-11-08 at 6 12 59 PM" src="https://user-images.githubusercontent.com/85320743/200720268-3bb04290-1609-4134-ad3d-7ffaec14b404.png">

SalePrice is highly correlated with OverallQual, GrLiVArea, GarageCars/GarageArea. Moderatley correlated with Year Built, YearRemodAdd, Total BsmtSF, and FullBath.

TotalBsmtSF and 1stFlrSF are highly correlated to each other. GarageYrBlt, GarageCars, and GarageArea are highly correlated as well. I will keep one feature that expresses the most important information about the general item while deleting the non essential features. In the case for the Garage, I will keep GarageCars since it measures the size of the garage based on the number of cars that can fit in the area. I will delete GarageArea since it is a similar feature to GarageCars and I will also delete GarageYrBlt becuase of it's low correlation to SalePrice. The same logic will be applied to the Basement features.


## Preprocessing the Data

### Dealing With Null Values

<img width="555" alt="Screen Shot 2022-11-08 at 2 47 52 PM" src="https://user-images.githubusercontent.com/85320743/200692637-afb86daf-ae68-4222-baf1-abdf7babf18b.png">

There are 19 collumns that have null values. I will delete any column with more than 15% percent of null values. Columns "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQU", and "LotFrontage" have to many null values for me to try any techniques to fill the missing data.

From the remaining 13 features, there are two groups of features with the same amount of null values. After looking at each group, they express information about the garage or the basement. As explained earlier I will keep one key feature from each group that expresses the most important information about the item of the house. 

As for the Electrical column we will remove the row with a null value.

To be quiet honest, my first attempt at creating a regression model I was struggling with the performace. The accuracy for my testing data was at 51%. After doing some research I realized I needed to make sure my target variable was normal. By normalizing my target variable I could develope a more accurate regression model.

### Checking Target Variable for Normality

<img width="467" alt="Screen Shot 2022-11-08 at 3 13 44 PM" src="https://user-images.githubusercontent.com/85320743/200696170-b63379e7-8812-4d93-ba28-3186d02f0bd6.png">

SalePrice is not normal. It shows a normal distribution skewed to the right. We will transform SalePrice to enforce normality.

<img width="502" alt="Screen Shot 2022-11-08 at 3 15 40 PM" src="https://user-images.githubusercontent.com/85320743/200696424-20f59d4f-426b-40c1-ad72-ab977a20b032.png">

We hava a normal distribution curve and the line in the probability plot has become more straight, thus meaning we have a normal target variable. Now when I compute my predicted y values, I'll exponentialize my values to accurately represent house prices to scale. 

### Feature Selection
I chose the top ten features correlated with SalePrice. I did this by obtaining the correlation coefficient for each feature, sorting the features in descending order of their correlation coefficient, and finally filtering the dataframe for only the top 10 features.

<img width="1002" alt="Screen Shot 2022-11-08 at 4 00 47 PM" src="https://user-images.githubusercontent.com/85320743/200702394-d72dcf80-c330-4f3e-9222-9b6fad1185ed.png">

### Creating Train Test Spilts

With SalePrice as y and the rest of my cleaned data as x, I will create my training and testing data. We spilt the data now so both sets have no discrepancies from the preprocessing steps. After the model has been processed by using the training set, I'll test the model by making predictions against the testing set.

<img width="609" alt="Screen Shot 2022-11-08 at 4 04 29 PM" src="https://user-images.githubusercontent.com/85320743/200703296-366cb404-c047-47de-997a-5d290c105ee0.png">

### Standardizing the Data

I will use the function StandardScaler to rescale the features so they have a mean of 0 and a variance of 1. The goal is to bring down all features to a common scale without distorting the differences in the range of the values.

<img width="468" alt="Screen Shot 2022-11-08 at 4 14 04 PM" src="https://user-images.githubusercontent.com/85320743/200704487-5413d73d-db8e-4de3-9b66-f1d139d10657.png">

## Creating the Model

The first model is a Decision Tree Regressor Model. I will create this model using Sci-Kit Learn. Once I create the model I will calculate the accuracy score for predictions, using the score() function. 

<img width="632" alt="Screen Shot 2022-11-08 at 4 21 09 PM" src="https://user-images.githubusercontent.com/85320743/200705244-febe6052-64c6-42c0-842c-b303bc9fd03b.png">

67 percent is not bad, considering I used one decision tree. Lets see how a ensemble regression model performs. I will try the Random Forest Regressor Model, which combines predictions from a specified number of decision trees and then averages the predictions to make a more accurate predicition.

<img width="601" alt="Screen Shot 2022-11-08 at 5 02 11 PM" src="https://user-images.githubusercontent.com/85320743/200710449-07c1cd45-1ab4-4149-a068-a8a4184eef30.png">

A accuracy score for the testing set of about 85%, which is pretty good! The model only considered the top 10 features correlated to SalePrice. Now lets see if adding a few more features will improve the model. 

## Optimizing the Random Forest Regression Model

This next model went through the same process as before, it just has the top 15 features instead of the 10.

<img width="750" alt="Screen Shot 2022-11-08 at 5 09 04 PM" src="https://user-images.githubusercontent.com/85320743/200711426-0312bd3a-d252-468b-b9ae-a7bd1909152e.png">

Slightly better at 86% test score! If I had more time I would tweak my model using different iterations of model parameters to optimize test score accuracy. 

## Computing My Predictions For The Original Test Dataset

Remember, up until this point the testing data for the models has been a subset from the original training dataset. Now I will compute the predicted house prices for the original testing dataset. I will preprocess the data the exact same way and use the most accurate model to predict the house prices. 

<img width="993" alt="Screen Shot 2022-11-08 at 5 18 23 PM" src="https://user-images.githubusercontent.com/85320743/200712629-b1c0ab9c-fc62-4467-8a38-8b7ef455f181.png">

After computing the predicted house prices, I expontalized the values to accurately represent house prices to scale. 

<img width="612" alt="Screen Shot 2022-11-08 at 5 20 14 PM" src="https://user-images.githubusercontent.com/85320743/200712867-c9d04d5d-3314-4e98-8cd5-2a4164ac2891.png">

Lastly, I will create a csv file containing the predicted house values and their corresponding house Id. 

<img width="663" alt="Screen Shot 2022-11-08 at 5 21 35 PM" src="https://user-images.githubusercontent.com/85320743/200713036-0397cdb0-674a-43a0-991d-b644aed32cf5.png">

Thank you for taking the time and looking at my notebook.
