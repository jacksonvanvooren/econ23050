### Graphs and Analysis for Citation Shares and Concentrations ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------- #

df_citation_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_citation_w_distance')

# Calculate distribution of patents for AI vs. non-AI
max_citations = 20
# AI
df_ai_filtered = df_ai[df_ai['external_citations'] <= max_citations]
df_ai_filtered = df_ai_filtered[df_ai_filtered['pub_year']<=2020]
# Non-AI
df_non_ai_filtered = df_non_ai[df_non_ai['external_citations'] <= max_citations]
df_non_ai_filtered = df_non_ai_filtered[df_non_ai_filtered['pub_year']<=2020]

# AI Patents Normalized Histogram
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
ax[0].hist(df_ai_filtered['external_citations'], bins=np.arange(0, max_citations+1), color='blue',
           alpha=0.7, edgecolor='black', density=True)
ax[0].set_title("Normalized Citation Histogram: AI Patents")
ax[0].set_xlabel("Number of External Citations")
ax[0].set_ylabel("Normalized Frequency")

# Non-AI Patents Normalized Histogram
ax[1].hist(df_non_ai_filtered['external_citations'], bins=np.arange(0, max_citations+1), color='red', alpha=0.7, edgecolor='black', density=True)
ax[1].set_title("Normalized Citation Histogram: Non-AI Patents")
ax[1].set_xlabel("Number of External Citations")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------- #

# Gini helper
def gini_coefficient(citations):
    """Compute Gini coefficient for a given array of citation counts."""
    sorted = np.sort(citations)
    n = len(citations)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sorted) / (n * np.sum(sorted))

# ---------------------------------------------------------------------------------- #

# Gini for AI patents in various countries
target_countries = ['US', 'DE', 'CN', 'CA', 'KR', 'JP', 'TW']
df_ai_countries = df_citation_distance[
    (df_citation_distance['disambig_country'].isin(target_countries))
    & (df_citation_distance['predict86_any_ai'] == 1)]

# Create 3-year bins, stopping at 2021 for truncation
def assign_3yr_group(year):
    if year <= 2021:
        return f"{year // 3 * 3}-{year // 3 * 3 + 2}"
df_ai_countries['year_group'] = df_ai_countries['pub_year'].apply(assign_3yr_group)

# Compute Gini coefficient for each 3-year group and country
gini_by_group_country = df_ai_countries.groupby(['year_group', 'disambig_country'])['external_citations'].apply(gini_coefficient).reset_index()
gini_by_group_country.columns = ['Year_Group', 'Country', 'Gini_Coefficient']

# Plot Boxplot comparing Gini coefficients for AI patents across countries
plt.figure(figsize=(12, 6))
plt.boxplot([gini_by_group_country[gini_by_group_country['Country'] == country]['Gini_Coefficient'] for country in target_countries], 
            labels=target_countries)
plt.xlabel("Country")
plt.ylabel("Gini Coefficient of Patent Citations")
plt.title("Comparison of Gini Coefficients of AI Patents across Countries")
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------- #

# Calculate Gini coefficients in the US over time

# Sort to US AI vs. non-AI patents, and all patents
df_citation_distance_us = df_citation_distance[df_citation_distance['disambig_country']=='US']
df_ai = df_citation_distance_us[(df_citation_distance_us['predict86_any_ai'] == 1)]
df_non_ai = df_citation_distance_us[(df_citation_distance_us['predict86_any_ai'] == 0)]

df_ai = df_ai[df_ai['pub_year'] <= 2020]
df_non_ai = df_non_ai[df_non_ai['pub_year'] <= 2020]
df_all = df_citation_distance_us[df_citation_distance['pub_year']<= 2020]

# Group by year using forward citations
gini_by_year_ai = df_ai.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()
gini_by_year_non_ai = df_non_ai.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()
gini_by_year_all = df_all.groupby('pub_year')['forward_citations'].apply(gini_coefficient).reset_index()

gini_by_year_ai.columns = ['Year', 'Gini_Coefficient']
gini_by_year_non_ai.columns = ['Year', 'Gini_Coefficient']
gini_by_year_all.columns = ['Year', 'Gini_Coefficient']

# Plot Gini coefficient for AI vs. Non-AI vs. all patents over time
plt.figure(figsize=(10, 5))
plt.plot(gini_by_year_ai['Year'], gini_by_year_ai['Gini_Coefficient'], marker='o', linestyle='-', label="AI Patents", color="blue")
plt.plot(gini_by_year_non_ai['Year'], gini_by_year_non_ai['Gini_Coefficient'], marker='s', linestyle='--', label="Non-AI Patents", color="red")
plt.plot(gini_by_year_all['Year'], gini_by_year_all['Gini_Coefficient'], marker='^', linestyle='-', label="All Patents", color="green")

plt.xlabel("Publication Year")
plt.ylabel("Gini Coefficient of Patent Citations")
plt.title("Inequality in Patent Citations Over Time")
plt.legend()
plt.grid(True)
plt.show()
