from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull
import textwrap

# Load the data
data = pd.read_excel('Personality_combined_version_2_adjusted_with_maximum_or_min_social_person.xlsx', engine='openpyxl')

# Clean 'Type' column (strip spaces and convert to lowercase)
data['Type'] = data['Type'].str.strip().str.lower()

# Separate data for local and international students
local_data = data[data['Type'] == 'local']
international_data = data[data['Type'] == 'international']

# Define questions (assuming these are in the first column of your data)
questions = data.columns[1:20]  # Assuming the first column is the type (local/international)

# ===================== FUNCTION TO VISUALIZE QUESTIONS ===================== #
def vis_questions(groupname, questions, color, title):
    plt.figure(figsize=(20, 30))  # Size of the figure for 20 plots (5x4)
    
    for i in range(len(questions)):
        plt.subplot(5, 4, i + 1)  # 5 rows and 4 columns
        
        # Set title first
        wrapped_title = "\n".join(textwrap.wrap(questions[i], width=55))  # Wrap long titles
        plt.title(wrapped_title, fontsize=10, pad=10)  # Set title with smaller font size and padding
        
        # Plotting the histogram for each question
        plt.hist(data[groupname[i]], bins=14, color=color, alpha=0.5)
        
        plt.xticks(fontsize=10)  # Font size for x-axis labels
        plt.yticks(fontsize=10)  # Font size for y-axis labels

        # Adjust y-limits to ensure all response values are visible
        plt.ylim(0, data[groupname[i]].value_counts().max() * 1.2)  # Extend y-limit slightly

    plt.suptitle(f'Distribution of Responses for {title}', fontsize=18)  # Dynamic main title based on local/non-local
    plt.subplots_adjust(hspace=1, wspace=0.3)  # Adjust space between plots
    
    # Add x and y axis explanation below the entire figure
    plt.figtext(0.5, 0.02, 'X-axis: 1 being least positive and 5 being most positive. Y-axis: Frequency', 
                ha="center", fontsize=12, color='black')
    
    plt.show()

# Call the function for local students
print('Q&As Related to Local Students')
vis_questions(local_data.columns[1:20], questions, 'orange', 'Local Students')  # Adjusted to use local data

# Call the function for international students
print('Q&As Related to International Students')
vis_questions(international_data.columns[1:20], questions, 'blue', 'International Students')  # Adjusted to use international data

# ===================== CLUSTERING AND PCA ===================== #
# Save 'Type' column for later use and drop it for clustering
types = data['Type']
df_model = data.drop('Type', axis=1)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
k_fit = kmeans.fit(df_model)

# Add cluster labels to the data
df_model['Clusters'] = k_fit.labels_

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df_model)

# Create a new DataFrame with PCA results and cluster information
df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = df_model['Clusters']
df_pca['Type'] = types  # Add 'Type' column back for coloring

# Filter MSP and LSP data
msp_lsp = df_pca[df_pca['Type'].isin(['msp', 'lsp'])]

# ===================== PLOT 1: PCA PLOT WITH NEAREST NEIGHBORS ===================== #
def plot_clusters_with_background(df_pca, ax, colors):
    for cluster in df_pca['Clusters'].unique():
        points = df_pca[df_pca['Clusters'] == cluster][['PCA1', 'PCA2']].values
        if len(points) > 2:  # Convex hull needs at least 3 points
            hull = ConvexHull(points)
            poly = plt.Polygon(points[hull.vertices], alpha=0.2, color=colors[cluster])
            ax.add_patch(poly)

def plot_nearest_neighbors(point, df, ax, num_neighbors=15, circle_color='blue'):
    point = np.atleast_2d(point)
    distances = np.linalg.norm(df[['PCA1', 'PCA2']].values - point, axis=1)
    nearest_indices = np.argsort(distances)[1:num_neighbors + 1]
    
    for index in nearest_indices:
        neighbor_point = df.iloc[index][['PCA1', 'PCA2']].values
        circle = plt.Circle(neighbor_point, 0.15, color=circle_color, fill=False, linestyle='dotted', linewidth=2)
        ax.add_patch(circle)

# Create PCA plot
fig, ax = plt.subplots(figsize=(10, 10))
custom_colors = ['#f3f31e', '#b1b5f3']  # Reverse cluster colors
plot_clusters_with_background(df_pca, ax, custom_colors)

sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Type', palette='Set2', alpha=0.7, ax=ax)
sns.scatterplot(data=msp_lsp, x='PCA1', y='PCA2', hue='Type', palette='Set2', s=200, alpha=0.9, edgecolor='black', ax=ax, legend=False)

if 'msp' in msp_lsp['Type'].values:
    plot_nearest_neighbors(msp_lsp[msp_lsp['Type'] == 'msp'][['PCA1', 'PCA2']].values[0], df_pca, ax, num_neighbors=30, circle_color='red')

if 'lsp' in msp_lsp['Type'].values:
    plot_nearest_neighbors(msp_lsp[msp_lsp['Type'] == 'lsp'][['PCA1', 'PCA2']].values[0], df_pca, ax, num_neighbors=30, circle_color='blue')

plt.title('PCA Clusters with Nearest Neighbors for MSP and LSP')
plt.legend(title='Type', bbox_to_anchor=(0.8, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ===================== PLOT 2: HORIZONTAL BAR PLOT FOR MSP AND LSP NEIGHBORS ===================== #
def count_local_international_neighbors(point, df, num_neighbors=30):
    point = np.atleast_2d(point)
    distances = np.linalg.norm(df[['PCA1', 'PCA2']].values - point, axis=1)
    nearest_indices = np.argsort(distances)[1:num_neighbors + 1]
    nearest_neighbors = df.iloc[nearest_indices]['Type']
    num_local = (nearest_neighbors == 'local').sum()
    num_international = (nearest_neighbors == 'international').sum()
    return num_local, num_international

# Calculate local and international neighbors for MSP and LSP
num_local_msp, num_international_msp = count_local_international_neighbors(msp_lsp[msp_lsp['Type'] == 'msp'][['PCA1', 'PCA2']].values[0], df_pca)
num_local_lsp, num_international_lsp = count_local_international_neighbors(msp_lsp[msp_lsp['Type'] == 'lsp'][['PCA1', 'PCA2']].values[0], df_pca)

def plot_nearest_neighbors_bar(local_values, international_values):
    labels = ['MSP', 'LSP']
    bar_width = 0.35
    index = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_local = ax.barh(index, local_values, bar_width, label='Local', color='#1f77b4')
    bar_international = ax.barh(index + bar_width, international_values, bar_width, label='International', color='#ff7f0e')

    ax.set_xlabel('Number of Students')
    ax.set_title('Local and International Students among Nearest 15 to MSP and LSP')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(labels)
    ax.legend()

    for bars, counts, total in zip([bar_local, bar_international], [local_values, international_values], [15, 15]):
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)} ({width / total * 100:.1f}%)', ha='left', va='center', color='black')

    plt.tight_layout()
    plt.show()

# Call the bar plotting function for nearest neighbors
plot_nearest_neighbors_bar([num_local_msp, num_local_lsp], [num_international_msp, num_international_lsp])

# ===================== PLOT 3: BAR PLOT FOR CLUSTER LOCAL/INTERNATIONAL COUNTS ===================== #
data_filtered = df_pca[~df_pca['Type'].isin(['msp', 'lsp'])]
cluster_counts = pd.crosstab(data_filtered['Clusters'], data_filtered['Type'])[['international', 'local']]

plt.figure(figsize=(10, 6))
ax = cluster_counts.plot(kind='barh', stacked=True, figsize=(10, 6), color=['#ff7f0e', '#1f77b4'])

for i in range(len(cluster_counts)):
    total = cluster_counts.iloc[i].sum()
    
    international_count = cluster_counts.iloc[i]['international']
    international_percentage = (international_count / total) * 100
    international_position = international_count / 2
    ax.text(international_position, i, f'{international_count} ({international_percentage:.1f}%)', ha='center', va='center', color='white', fontsize=12)
    
    local_count = cluster_counts.iloc[i]['local']
    local_percentage = (local_count / total) * 100
    local_position = international_count + (local_count / 2)
    ax.text(local_position, i, f'{local_count} ({local_percentage:.1f}%)', ha='center', va='center', color='white', fontsize=12)

plt.title('Counts of International and Local Students in Each Cluster (MSP vs LSP)')
plt.xlabel('Number of Students')
plt.ylabel('Clusters')
plt.legend(title='Student Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.yticks(ticks=[0, 1], labels=['MSP', 'LSP'])

plt.tight_layout()
plt.show()
