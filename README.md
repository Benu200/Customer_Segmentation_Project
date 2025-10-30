# Customer_Segmentation_Project
This project aims to segment customers into distinct groups based on their Age, Annual Income, and Spending Score using the K-Means Clustering algorithm.
By identifying patterns and similarities among customers, businesses can better understand consumer behavior, design targeted marketing strategies, and enhance decision-making for improved customer satisfaction and profitability.

# What is Customer Segmentation
Customer Segmentation is the process of division of customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.
Companies that deploy customer segmentation are under the notion that every customer has different requirements and require a specific marketing effort to address them appropriately. Companies aim to gain a deeper approach of the customer they are targeting. Therefore, their aim has to be specific and should be tailored to address the requirements of each and every individual customer. Furthermore, through the data collected, companies can gain a deeper understanding of customer preferences as well as the requirements for discovering valuable segments that would reap them maximum profit. This way, they can strategize their marketing techniques more efficiently and minimize the possibility of risk to their investment.
The technique of customer segmentation is dependent on several key differentiators that divide customers into groups to be targeted. Data related to demographics, geography, economic status as well as behavioral patterns play a crucial role in determining the company direction towards addressing the various segments

# What is K-Means Algorithm
While using the k-means clustering algorithm, the first step is to indicate the number of clusters (k) that we wish to produce in the final output. The algorithm starts by selecting k objects from dataset randomly that will serve as the initial centers for our clusters. These selected objects are the cluster means, also known as centroids. Then, the remaining objects have an assignment of the closest centroid. This centroid is defined by the Euclidean Distance present between the object and the cluster mean. We refer to this step as “cluster assignment”. When the assignment is complete, the algorithm proceeds to calculate new mean value of each cluster present in the data. After the recalculation of the centers, the observations are checked if they are closer to a different cluster. Using the updated cluster mean, the objects undergo reassignment. This goes on repeatedly through several iterations until the cluster assignments stop altering. The clusters that are present in the current iteration are the same as the ones obtained in the previous iteration.

# Objective
The main goal of this project is to group customers with similar purchasing and income characteristics to:
Understand customer behavior and preferences
Personalize marketing campaigns for each segment
Improve business strategies and product positioning
Enhance customer relationship management (CRM)

# Problem Statement
Businesses often have a large amount of customer data but lack insights into which customers spend more, which are price-sensitive, or which could be targeted for premium offers.
This project solves that problem by using unsupervised learning (K-Means) to find natural groupings (segments) among customers based on their demographic and behavioral attributes.

# Dataset Information
The dataset (customer_segmentation_dataset.csv) contains the following features:

CustomerID	:      Unique identifier for each customer,
Gender	   :       Gender of the customer,
Age	              Age of the customer,
Annual Income	:    Annual income in currency value,
Spending Score	:  Score assigned based on customer spending behavior,
Profession	  :    Occupation of the customer,
City	       :     City of residence,
Tenure (Years)	:   Duration of customer relationship in years

# Steps
# 1. Data Preprocessing
Loaded and cleaned the dataset
Handled missing values
Removed extra spaces from column names
Encoded categorical variables (like Gender)
Converted columns with commas to numeric types

# 2. Exploratory Data Analysis (EDA)
Visualized distributions using Histograms and Boxplots
Checked correlations using a Heatmap
Identified relationships between income and spending behavior

# 3. Feature Scaling
Standardized numerical columns using StandardScaler to ensure equal importance in clustering.

# 4. Finding Optimal Clusters
Applied Elbow Method and Silhouette Score to determine the ideal number of clusters.
Found that 3 clusters gave the best balance between compactness and separation.


<img width="723" height="524" alt="Screenshot 2025-10-30 105623" src="https://github.com/user-attachments/assets/63b2d509-5ee1-4ae6-bb90-cc3bb6c6b7ba" />

Based on the Elbow Method, inertia decreases rapidly until k = 4, after which the curve flattens. Therefore, the optimal number of clusters is selected as 4.

<img width="707" height="528" alt="image" src="https://github.com/user-attachments/assets/e8e3818b-1ff6-473d-927e-3e95caa183b1" />

The Silhouette Score was calculated for values of k ranging from 2 to 10.
The highest average Silhouette Score was obtained at k = 2, indicating that two clusters provide the best separation and compactness among the data points.


# 5. Model Building (K-Means Clustering)
Implemented K-Means with n_clusters = 3
Predicted and assigned cluster labels to each customer

# 6. Dimensionality Reduction (PCA)
Used Principal Component Analysis (PCA) to visualize clusters in 2D space.

# 7. Cluster Profiling
Calculated mean values of numerical features for each cluster
Identified key behavioral differences between customer groups

# Results & Insights
Cluster	Customer Type	Description
Cluster 0	🟢 High-Value Customers	High income, high spending — target for premium products
Cluster 1	🟣 Moderate Spenders	Average income, balanced spending
Cluster 2	🟡 Budget Customers	Low income, low spending — focus on discounts and offers

# Technologies Used
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

# Evaluation Metrics
Elbow Method (for optimal clusters)
Silhouette Score (for cluster quality)
PCA Visualization (for interpretability)

