import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as  np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


# Title of the dashboard
st.title('Data Visualization Dashboard On Customer Segmentation Data')

# Load dataset
df = pd.read_csv('orders_export.csv')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Sidebar - Filter selection
option = st.sidebar.selectbox('Select column to filter', df.columns)
filter_value = st.sidebar.text_input('Enter value to filter')

# Filter data based on user input
if filter_value:
    df = df[df[option].astype(str) == filter_value]

# Show filtered data
st.write("Filtered Data:", df)

# Histogram
st.subheader('Histogram')
column = st.selectbox('Select column for histogram', df.select_dtypes(include=['int', 'float']).columns)
fig, ax = plt.subplots()
df[column].hist(ax=ax, bins=20)
st.pyplot(fig)



columns_to_exclude = [
    'Tax 2 Name', 'Tax 2 Value', 'Tax 3 Name', 'Tax 3 Value',
    'Tax 4 Name', 'Tax 4 Value', 'Tax 5 Name', 'Tax 5 Value',
    'Lineitem discount', 'Receipt Number', 'Duties', 'Next Payment Due At'
]

# Drop the columns
orders_export_data = df.drop(columns=columns_to_exclude)

# Select only numeric columns
df_numeric = orders_export_data.select_dtypes(include=['int64', 'float64'])

# Display subheading and plot in Streamlit
st.subheader("Pie Chart for Selected Attributes")

# Select categorical columns
categorical_columns = [
    'Financial Status',
    'Fulfillment Status',
    'Accepts Marketing',
    'Currency',
    'Shipping Method',
    'Lineitem fulfillment status',
    'Billing Company',
    'Shipping Company',
    'Payment Method',
    'Vendor',
    'Risk Level',
    'Source',
    'Tax 1 Name',
    'Payment Terms Name'
]


# Dropdown to select a categorical column
selected_column = st.selectbox(
    'Select a categorical column to display a pie chart',
    options=categorical_columns
)

if selected_column:
    # Calculate value counts for the selected column
    value_counts = orders_export_data[selected_column].value_counts()
    
    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=140)
    plt.title(f'Pie Chart of {selected_column}')
    
    st.pyplot(plt)
else:
    st.info('Please select a categorical column to display a pie chart.')


# Correlation Heatmap
st.subheader('Correlation Heatmap')

# Compute correlation matrix
corr = df_numeric.corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corr, ax=ax, annot=True, cmap='twilight', fmt='.2f',center=0)
st.pyplot(fig)

# Observations
st.subheader('Observations')

st.markdown("""
- The "Correlation Heatmap of Numerical Variables" provides a visual representation of the correlations between numerical attributes.
- **Strength of Correlation:** The heatmap color intensity indicates the strength of correlation between pairs of numerical attributes. Darker colors represent stronger correlations (positive or negative), while lighter colors represent weaker or no correlations.
- **Positive and Negative Correlation:** Positive correlations are indicated by colors moving towards the darker end of the spectrum (towards red), while negative correlations are indicated by colors moving towards the lighter end of the spectrum (towards blue).
- **Direction of Correlation:** A lighter cell signifies that as one attribute increases, the other tends to increase as well (positive correlation). A darker blue cell signifies that as one attribute increases, the other tends to decrease (negative correlation).
- **Correlation Values:** The numerical values within each cell represent the correlation coefficient between the corresponding attributes. These coefficients range from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.
""")


#  descriptive statistics of the DataFrame
st.subheader('Descriptive statistics of the DataFrame')

# Generate the styled DataFrame
styled_df = df_numeric.describe().T.style.set_properties(**{
    'background-color': '#800000',
    'color': 'white',
    'border-color': 'white',
     'font-size': '8pt'
})

# Convert the styled DataFrame to HTML and display it in Streamlit
st.markdown(styled_df.to_html(), unsafe_allow_html=True)


# "Correlation with Lineitem price"
st.subheader("Correlation with Lineitem price")
# Assuming orders_export_data is already loaded as a DataFrame
correlation_data = abs(df_numeric.corr()['Lineitem price']).sort_values()[:-1]

# Plot the pie chart
plt.figure(figsize=(10, 10))
pct = [0.1] * len(correlation_data)  # Equal explode values for each slice
textprops = {'color': "blue", "fontsize": 12}
wedgeprops = {'edgecolor': "black", 'linewidth': 1, 'antialiased': True}

correlation_data.plot.pie(explode=pct, autopct="%2.1f%%", shadow=True, startangle=90,
                          wedgeprops=wedgeprops, textprops=textprops)

plt.axis("off")
plt.title("Correlation with Lineitem price")
plt.legend(bbox_to_anchor=(1.9, 0.8), ncol=2, facecolor="#9B7DDE", framealpha=0.9, shadow=True, edgecolor="blue", fancybox=True)

# Display the plot in Streamlit
st.pyplot(plt)


# Observations
st.subheader('Observations')

st.markdown("""
- **The "Correlation with Lineitem price" pie chart** provides information about the correlation between various attributes and the "Lineitem price" attribute.
- **Attributes with Higher Positive Correlation:** The larger segments of the pie chart represent attributes that have a higher positive correlation with the "Lineitem price." These attributes have a stronger linear relationship with the "Lineitem price" and tend to increase or decrease in conjunction with changes in the price.
- **Attributes with Lower Positive Correlation:** The smaller segments of the pie chart represent attributes with a lower positive correlation with the "Lineitem price." While they still have a positive correlation, the relationship is not as strong as with the larger segments.
- **Attributes with Negative Correlation:** There are no negative correlation segments in the pie chart. This suggests that none of the attributes in the dataset have a strong negative linear relationship with the "Lineitem price."
- **Correlation Strength:** The size of each segment corresponds to the strength of the correlation between the attribute and the "Lineitem price." Larger segments indicate attributes with a higher correlation strength.
- **Correlation Magnitude:** The percentage value displayed inside each segment represents the magnitude of the correlation coefficient. This provides a quantitative measure of the strength of the correlation between each attribute and the "Lineitem price."
- **Attribute Importance:** The attributes represented by larger segments are more influential in affecting changes in the "Lineitem price." Attributes with higher positive correlation can potentially be used to predict or explain variations in the price.
""")


# Display subheading and plot in Streamlit
st.subheader("Histograms of Selected Attributes")

# Define the number of columns per row in the subplot grid
n = 3

# Select the columns you want to plot
selected_columns = [
    'Subtotal', 'Shipping', 'Taxes', 'Total', 'Discount Amount',
    'Lineitem quantity', 'Lineitem price', 'Lineitem compare at price',
    'Lineitem requires shipping', 'Lineitem taxable', 'Refunded Amount',
    'Id', 'Tax 1 Value'
]

# Calculate the number of rows needed for the subplot grid
num_rows = math.ceil(len(selected_columns) / n)

# Set the figure size based on the number of rows and columns
plt.figure(figsize=[10, 4 * num_rows])

# Create subplots
for c in range(len(selected_columns)):
    plt.subplot(num_rows, n, c + 1)
    if orders_export_data[selected_columns[c]].dtype != 'object':
        sns.histplot(data=orders_export_data, x=selected_columns[c], kde=True, color='#FFB6C1')
        plt.title(selected_columns[c])
        plt.xlabel('')  # Remove x-label for better visualization

# Adjust the layout and display the plot
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Observations
st.subheader("Observations")

st.markdown("""
- **Distribution Shape:** The histograms display the shape of the distribution for each selected numerical variable. You can observe whether the distribution is symmetric (normal), skewed to the left (negatively skewed), or skewed to the right (positively skewed).
- **Central Tendency:** The position of the peak in the histogram indicates the central tendency of the distribution, such as the mean or median.
- **Spread and Variability:** The width and spread of the distribution provide information about the variability of the data. Narrow distributions have lower variability, while wider distributions have higher variability.
- **Outliers:** Outliers, if present, can be identified as data points that fall far from the main bulk of the distribution. You can detect potential outliers by observing data points that are isolated from the main cluster.
- **Skewness:** The direction and degree of skewness can be determined from the histogram shape. Positive skewness indicates a tail extending to the right, while negative skewness indicates a tail extending to the left.
- **Kurtosis:** The height and sharpness of the peak in the histogram give an indication of the kurtosis (peakedness) of the distribution. High kurtosis indicates a more peaked distribution, while low kurtosis indicates a flatter distribution.
- **Bimodality and Multimodality:** Multiple peaks in the histogram suggest bimodal or multimodal distributions, where the data may have multiple modes or clusters.
- **Data Ranges:** The x-axis of each histogram shows the range of values for the selected numerical variable.
- **Normality and Transformations:** Histograms help identify whether the data follows a normal distribution. Deviations from normality can guide decisions about applying data transformations.
- **Data Characteristics:** You can gather insights into the inherent characteristics of the selected numerical variables, such as order amounts, quantities, taxes, etc.
""")

#Log, Square Root, and Cube Root Transformations
st.subheader("Log, Square Root, and Cube Root Transformations")
# Define the number of columns per row in the subplot grid
n = 3

# Select the numerical columns you want to transform
numerical_columns = [
    'Subtotal', 'Shipping', 'Taxes', 'Total', 'Discount Amount',
    'Lineitem quantity', 'Lineitem price', 'Lineitem compare at price',
    'Refunded Amount', 'Id', 'Tax 1 Value'
]

# Log Transformation
st.subheader("Log Transformation Histograms")
plt.figure(figsize=[10, 4 * math.ceil(len(numerical_columns) / n)])
for c in range(len(numerical_columns)):
    plt.subplot(math.ceil(len(numerical_columns) / n), n, c + 1)
    log_transformed = np.log(orders_export_data[numerical_columns[c]] + 1)  # Adding 1 to avoid log(0)
    sns.histplot(log_transformed, kde=True, color='#CFBFF3')
    plt.title('Log Transformed - {}'.format(numerical_columns[c]))
    plt.xlabel('')

plt.tight_layout()
st.pyplot(plt)

# Square Root Transformation
st.subheader("Square Root Transformation Histograms")
plt.figure(figsize=[10, 4 * math.ceil(len(numerical_columns) / n)])
for c in range(len(numerical_columns)):
    plt.subplot(math.ceil(len(numerical_columns) / n), n, c + 1)
    sqrt_transformed = np.sqrt(orders_export_data[numerical_columns[c]])
    sns.histplot(sqrt_transformed, kde=True, color='#94F2E7')
    plt.title('Sqrt Transformed - {}'.format(numerical_columns[c]))
    plt.xlabel('')

plt.tight_layout()
st.pyplot(plt)

# Cube Root Transformation
st.subheader("Cube Root Transformation Histograms")
plt.figure(figsize=[10, 4 * math.ceil(len(numerical_columns) / n)])
for c in range(len(numerical_columns)):
    plt.subplot(math.ceil(len(numerical_columns) / n), n, c + 1)
    cube_root_transformed = np.cbrt(orders_export_data[numerical_columns[c]])
    sns.histplot(cube_root_transformed, kde=True, color='#EEF3AA')
    plt.title('Cube Root Transformed - {}'.format(numerical_columns[c]))
    plt.xlabel('')

plt.tight_layout()
st.pyplot(plt)

# Observations
st.subheader("Observations")

st.markdown("""
- **Log Transformation Histograms:** The first set of histograms displays the distributions of numerical variables after applying the Log transformation. Log transformations can help address skewed distributions and compress the range of values for variables that exhibit exponential growth. Variables with high positive skewness in their original distribution may have a more symmetrical distribution after the Log transformation.
- **Square Root Transformation Histograms:** The second set of histograms shows the distributions after applying the Square Root transformation. Square Root transformations are useful for addressing positive skewness and handling data with square-root relationships.
- **Cube Root Transformation Histograms:** The third set of histograms presents the distributions after applying the Cube Root transformation. Cube Root transformations are effective for handling variables with cube-root relationships and mitigating the impact of outliers.
""")


# Aggregate data by Email for feature engineering
grouped_data = orders_export_data.groupby('Email').agg({
    'Subtotal': ['count', 'mean'],
    'Accepts Marketing': lambda x: int('yes' in x.values),
    'Source': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}).reset_index()
grouped_data.columns = ['Email', 'Total Orders', 'Average Order Value', 'Accepts Marketing', 'Most Common Source']

# Define numerical columns for clustering
numerical_columns = ['Total Orders', 'Average Order Value']

# Select relevant columns for clustering
cluster_data = grouped_data[numerical_columns]

# Standardize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Perform Agglomerative Clustering
n_clusters = 5  # Number of clusters
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
grouped_data['Cluster'] = clustering.fit_predict(cluster_data_scaled)

# Plot the clustering results
st.subheader('Agglomerative Clustering of Customers')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=grouped_data, x='Total Orders', y='Average Order Value', hue='Cluster', palette='tab10', ax=ax)
plt.title('Agglomerative Clustering of Customers')
plt.xlabel('Total Orders')
plt.ylabel('Average Order Value')
plt.legend(title='Cluster')
st.pyplot(fig)

# Display observations
st.subheader('Observations and Marketing Strategies')

st.markdown("""
**Cluster 0:**
- **Characteristics:** Customers with relatively low total orders (0 to 2) and average order value (up to 40,000).
- **Strategy:** Focus on nurturing and converting these customers into more frequent buyers. Offer incentives, discounts, or promotions for larger orders.

**Cluster 1:**
- **Characteristics:** Customers with moderate total orders (6 to 13) and average order value (up to 20,000).
- **Strategy:** Encourage these customers to increase their average order value through cross-selling and upselling. Offer product bundles, related items, or loyalty programs.

**Cluster 2:**
- **Characteristics:** Customers with moderate total orders (2 to 5) and average order value (up to 10,000).
- **Strategy:** Similar to Cluster 0, but with slightly higher average order values. Offer personalized recommendations and incentives based on their purchase history.

**Cluster 3:**
- **Characteristics:** Customers with very low total orders (only 1) but very high average order value (above 70,000).
- **Strategy:** Focus on maintaining the satisfaction of these high-value customers. Provide exceptional customer service, exclusive offers, and personalized attention to retain their loyalty.

**Cluster 4:**
- **Characteristics:** Similar to Cluster 0, with low total orders (0 to 2) and average order value (up to 40,000).
- **Strategy:** Implement strategies similar to Cluster 0.

**Marketing Strategies:**

- **For Cluster 0 and Cluster 4:** Emphasize customer retention and conversion. Implement targeted email campaigns showcasing new products, special offers, and the benefits of becoming regular customers.

- **For Cluster 1 and Cluster 2:** Focus on upselling and increasing average order value. Recommend complementary products and offer tiered discounts on larger orders.

- **For Cluster 3:** Provide personalized experiences, exclusive content, and priority service. Tailor marketing messages to their high-value purchases.
""")