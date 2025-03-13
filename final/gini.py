###


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_citation_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_citation_w_distance')



max_citations = 20

# Filter out patents with more than 30 citations
df_ai_filtered = df_ai[df_ai['external_citations'] <= max_citations]
df_non_ai_filtered = df_non_ai[df_non_ai['external_citations'] <= max_citations]
df_ai_filtered = df_ai_filtered[df_ai_filtered['pub_year']<=2020]
df_non_ai_filtered = df_non_ai_filtered[df_non_ai_filtered['pub_year']<=2020]

# Set up figure and axes
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# AI Patents Normalized Histogram
ax[0].hist(df_ai_filtered['external_citations'], bins=np.arange(0, max_citations+1), color='blue', alpha=0.7, edgecolor='black', density=True)
ax[0].set_title("Normalized Citation Histogram: AI Patents")
ax[0].set_xlabel("Number of External Citations")
ax[0].set_ylabel("Normalized Frequency")

# Non-AI Patents Normalized Histogram
ax[1].hist(df_non_ai_filtered['external_citations'], bins=np.arange(0, max_citations+1), color='red', alpha=0.7, edgecolor='black', density=True)
ax[1].set_title("Normalized Citation Histogram: Non-AI Patents")
ax[1].set_xlabel("Number of External Citations")

# Adjust layout
plt.tight_layout()
plt.show()










df_citation_distance_us = df_citation_distance[df_citation_distance['disambig_country']=='US']

df_ai = df_citation_distance_us[(df_citation_distance_us['predict86_any_ai'] == 1)]
df_non_ai = df_citation_distance_us[(df_citation_distance_us['predict86_any_ai'] == 0)]

threshold = 0
year_threshold = 2020
df_ai_wcitation = df_ai[(df_ai['external_citations'] >= threshold) & (df_ai['pub_year'] <= year_threshold)]
df_non_ai_wcitation = df_non_ai[(df_non_ai['external_citations']>=threshold) & (df_non_ai['pub_year'] <= year_threshold)]
df_all_wcitation = df_citation_distance_us[(df_citation_distance_us['external_citations']>=threshold) & (df_citation_distance['pub_year']<=year_threshold)]

def gini_coefficient(citations):
    """Compute Gini coefficient for a given array of citation counts."""
    sorted = np.sort(citations)
    n = len(citations)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sorted) / (n * np.sum(sorted))

gini_by_year = df_ai_wcitation.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()
gini_by_year_non_ai = df_non_ai_wcitation.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()
gini_all = df_all_wcitation.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()

gini_by_year.columns = ['Year', 'Gini_Coefficient']

gini_by_year_non_ai.columns = ['Year', 'Gini_Coefficient']

gini_all.columns = ['Year', 'Gini_Coefficient']

# Plot Gini coefficient for AI vs. Non-AI patents over time
plt.figure(figsize=(10, 5))
plt.plot(gini_by_year['Year'], gini_by_year['Gini_Coefficient'], marker='o', linestyle='-', label="AI Patents", color="blue")
plt.plot(gini_by_year_non_ai['Year'], gini_by_year_non_ai['Gini_Coefficient'], marker='s', linestyle='--', label="Non-AI Patents", color="red")
plt.plot(gini_all['Year'], gini_all['Gini_Coefficient'], marker='^', linestyle='-', label="All Patents", color="green")

plt.xlabel("Publication Year")
plt.ylabel("Gini Coefficient of Patent Citations")
plt.title("Inequality in Patent Citations Over Time")
plt.legend()
plt.grid(True)
plt.show()









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define target countries
target_countries = ['US', 'DE', 'CN', 'CA', 'KR', 'JP', 'TW']

# Filter dataset for AI patents in specified countries
df_ai_countries = df_citation_distance[
    (df_citation_distance['disambig_country'].isin(target_countries))
    & (df_citation_distance['predict86_any_ai'] == 1)
]

# COULD FILTER FOR AI OR NOT

# Consider patents with at least one citation
df_ai_wcitation = df_ai_countries[df_ai_countries['external_citations'] > 0]

# Gini coefficient function
def gini_coefficient(citations):
    """Compute Gini coefficient for a given array of citation counts."""
    sorted_citations = np.sort(citations)
    n = len(citations)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sorted_citations) / (n * np.sum(sorted_citations))

# Create 3-year bins, stopping at 2021
def assign_3yr_group(year):
    if year <= 2021:
        return f"{year // 3 * 3}-{year // 3 * 3 + 2}"

# Apply binning
df_ai_wcitation['year_group'] = df_ai_wcitation['pub_year'].apply(assign_3yr_group)

# Compute Gini coefficient for each 3-year group and country
gini_by_group_country = df_ai_wcitation.groupby(['year_group', 'disambig_country'])['external_citations'].apply(gini_coefficient).reset_index()

# Rename columns
gini_by_group_country.columns = ['Year_Group', 'Country', 'Gini_Coefficient']

# Plot Boxplot comparing Gini coefficients for AI patents across countries
plt.figure(figsize=(12, 6))

# Plot boxplot
plt.boxplot([gini_by_group_country[gini_by_group_country['Country'] == country]['Gini_Coefficient'] for country in target_countries], 
            labels=target_countries)

# Add labels and title
plt.xlabel("Country")
plt.ylabel("Gini Coefficient of Patent Citations")
plt.title("Comparison of Gini Coefficients of AI Patents across Countries")
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
