### Calculates top decile patents and runs regressions of impact on predictors ###

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------------- #

# Load distance and counts dataset
df_citation_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_citation_w_distance')

# Filter to data from 2000-2020 to avoid citation bias with gini calculations by state
df_filtered = df_citation_distance[df_citation_distance['pub_year'] <= 2020]

# ---------------------------------------------------------------------------------- #

# Calculate top decile patents of external citation counts by year
# Calculate the top decile cutoff for each year
top_decile_cutoffs = df_filtered.groupby('pub_year')['external_citations'].quantile(0.9)

# Create the dependent variable (top decile indicator)
df_filtered['top_decile'] = df_filtered.apply(
    lambda row: 1 if row['external_citations'] >= top_decile_cutoffs[row['pub_year']] else 0, axis=1)

# Find share of AI patents in the top decile
# Group by year and calculate the total number of top decile patents and the number of top decile AI patents
top_decile_by_year = df_citation_distance[df_citation_distance['top_decile'] == 1] \
    .groupby('pub_year').agg(
        total_top_decile=('top_decile', 'size'),  # Total top decile patents
        ai_top_decile=('predict86_any_ai', lambda x: (x == 1).sum())  # AI patents in top decile
    )

# Calculate the share of AI patents in the top decile for each year
top_decile_by_year['ai_share'] = top_decile_by_year['ai_top_decile'] / top_decile_by_year['total_top_decile']

# Plot
plt.figure(figsize=(12, 6))
plt.plot(top_decile_by_year.index, top_decile_by_year['ai_share'], marker='o', color='b')

plt.xlabel('Year')
plt.ylabel('Share of AI Patents in Top Decile')
plt.title('Share of AI Patents in Top Citation Decile by Year')
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------- #

# Create controls at the state level
# Total patent counts by state
state_patent_counts = df_filtered.groupby('disambig_state')['patent_id'].nunique().reset_index()
state_patent_counts.columns = ['state', 'total_patents']
df_filtered = df_filtered.merge(state_patent_counts[['state', 'total_patents']],
                                left_on='disambig_state', right_on='state', how='left')

# ---------------------------------------------------------------------------------- #

# Gini coefficient by state
def gini_coefficient(citations):
    """Compute Gini coefficient for a given array of citation counts."""
    sorted = np.sort(citations)
    n = len(citations)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sorted) / (n * np.sum(sorted))

us_states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

df_states = df_filtered[df_filtered['disambig_state'].isin(us_states)]
gini_by_group_state = df_states.groupby(['disambig_state'])['external_citations'].apply(gini_coefficient).reset_index()
gini_by_group_state.columns = ['disambig_state', 'gini']

gini_by_group_state['gini'] = gini_by_group_state['gini'] * 100
df_filtered = df_filtered.merge(gini_by_group_state, how='left', on='disambig_state')

# ---------------------------------------------------------------------------------- #

# Spillover bias by state
# Load exported data from home_bias.py
spillover_bias = pd.read_csv('spillover_bias_df.csv')
df_filtered = df_filtered.merge(spillover_bias, how='left', left_on='disambig_state', right_on='state')
df_filtered['home_bias'] = df_filtered['index'] * 100

# ---------------------------------------------------------------------------------- #

# Run regressions

# Drop NA's and clean
df_citation_distance_reg = df_filtered.dropna()

# Patent characteristics
df_citation_distance_reg['patent_age'] = 2025 - df_citation_distance_reg['pub_year']
df_citation_distance_reg['log_total_patents'] = np.log(df_citation_distance_reg['total_patents'])

# State level controls
# Log distance undefined (some rows did not properly sort out self citations)
df_citation_distance_reg = df_citation_distance_reg[(df_citation_distance_reg['mean_distance'] != 0) & (df_citation_distance_reg['mean_distance'].notna())]
df_citation_distance_reg['log_mean_distance'] = np.log(df_citation_distance_reg['mean_distance'])
df_citation_distance_reg['ai_x_home'] = df_citation_distance_reg['home_bias'] * df_citation_distance_reg['predict86_any_ai']
df_citation_distance_reg['ai_x_dist'] = df_citation_distance_reg['log_mean_distance'] * df_citation_distance_reg['predict86_any_ai']

# Dependent variable
y = df_citation_distance_reg['top_decile']

# Simple model with just AI as a predictor
X_1 = df_citation_distance_reg[['predict86_any_ai']]
X_1 = sm.add_constant(X_1)

# Patent information including its age and citing to cited distance
X_2 = df_citation_distance_reg[['predict86_any_ai', 'log_mean_distance', 'patent_age']]
X_2 = sm.add_constant(X_2)

# Adding in state level controls
X_3 = df_citation_distance_reg[['predict86_any_ai', 'log_mean_distance', 'patent_age', 'gini', 'home_bias', 'log_total_patents']]
X_3 = sm.add_constant(X_3)

# Logits
model_1 = sm.Logit(y, X_1).fit()
model_2 = sm.Logit(y, X_2).fit()
model_3 = sm.Logit(y, X_3).fit()

print(model_1.summary())
print(model_2.summary())
print(model_3.summary())

# ---------------------------------------------------------------------------------- #
# Top 5 most represented states in the top decile of external citations

# Filter for U.S. patents with at least one citation and top decile flag
state_top_decile_counts = df_filtered[df_filtered['top_decile'] == 1] \
    .groupby('disambig_state')['top_decile'].count() \
    .sort_values(ascending=False)

# Select the top 4 states with the most patents in the top decile
top_5_states = state_top_decile_counts.head(5).index

# Filter data to include only top 4 states
df_top_5_states = df_filtered[df_filtered['disambig_state'].isin(top_5_states)]

# Group by publication year and state, and count the number of patents in the top decile for these states
state_decile_counts_top_5 = df_top_5_states[df_top_5_states['top_decile'] == 1] \
    .groupby(['pub_year', 'disambig_state'])['top_decile'].count() \
    .unstack(fill_value=0)

# Group by year and get the total number of top decile patents in the US
total_top_decile_patents_per_year = df_filtered[df_filtered['top_decile'] == 1] \
    .groupby('pub_year').size()

# Normalize the data into a proportion
state_decile_counts_top_5_norm = state_decile_counts_top_5.div(total_top_decile_patents_per_year, axis=0)

# Plot
plt.figure(figsize=(12, 6))
state_decile_counts_top_5_norm.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.xlabel('Year')
plt.ylabel('Proportion of Top Decile Patents')
plt.title('Top 5 States Represented in the Top Citation Decile (Normalized by Total Top Decile Patents)')
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
