## Team Project : Walmart Trip Type Classification
Use market basket analysis to classify shopping trips
- https://www.kaggle.com/c/walmart-recruiting-trip-type-classification

### Team Name - 10 to 10
- Nam DH
- Jang HW
- Kim HS

## 1. Objective
classifying customer trips using a transactional dataset of the items they've purchased to create the best shopping experience for every customer

모든 고객에게 최상의 쇼핑 경험을 제공하기 위해 고객들의 구매 데이터를 사용하여 쇼핑유형을 분류하는 것을 목적으로 한다

## 2. Dataset Description
To give a few hypothetical examples of trip types: a customer may make a small daily dinner trip, a weekly large grocery trip, a trip to buy gifts for an upcoming holiday, or a seasonal trip to buy clothes.

- Train : 640,754 rows × 7 columns
- Test : 653,646 rows × 6 columns

Features|Types of Variable|Feature Description|Unique Value
--------|------------|------------|----------
TripType|Categorical|a categorical id representing the type of shopping trip the customer made|38
VisitNumber|Categorical|an id corresponding to a single trip by a single customer|95674
DepartmentDescription|Categorical|a high-level description of the item's department|68
FinelineNumber|Categorical|a more refined category for each of the products, created by Walmart|5196
Upc|Categorical|the UPC number of the product purchased|97715
Weekday|Categorical|the weekday of the trip|7
Scancount|Numerical|the number of the given item that was purchased|-

* "Fineline" refers to a group of items within a department which show similar sales patterns. Prices, ingredient or materials, peak season, size, color, country of origin - there are thousands of ways that we can classify products. Finelines are determined by sales patterns.

## 3. Evaluation
- the multi-class logarithmic loss

## 4. Modeling
- train data

Model|Accuracy|log loss
-----|--------|--------
logistic|0.749|0.865
Random Forest|0.558|1.831
**light GBM**|**0.90**|**0.670177**

## 5. Result
<img src="kaggle_result.jpg">

- Kaggle score
  - Public 1047명 중 201위 19.2%
  - Private 1047명 중 206위 19.67%
