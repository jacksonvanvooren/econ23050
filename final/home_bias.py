### Code for home bias, spillover, and distances ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import scipy.stats as stats
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------------- #

# Load datasets
df_citation_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_citation_w_distance')
df_external_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_external_distance.csv')

# ---------------------------------------------------------------------------------- #

# Define a threshold for citation count
threshold = 0
df_ai = df_citation_distance[df_citation_distance['predict86_any_ai']==1]
df_non_ai = df_citation_distance[df_citation_distance['predict86_any_ai']==0]

# Filter the datasets based on citation count threshold
df_ai_wcitation = df_ai[df_ai['forward_citations'] >= threshold]
df_non_ai_wcitation = df_non_ai[df_non_ai['forward_citations'] >= threshold]
df_all_wcitation = df_citation_distance[df_citation_distance['forward_citations'] >= threshold]

# Calculate mean citation distance per year per group
mean_distance_by_year_ai = df_ai_wcitation.groupby('pub_year')['mean_distance'].mean().reset_index()
mean_distance_by_year_non_ai = df_non_ai_wcitation.groupby('pub_year')['mean_distance'].mean().reset_index()
mean_distance_by_year_all = df_all_wcitation.groupby('pub_year')['mean_distance'].mean().reset_index()

# Rename columns for consistency
mean_distance_by_year_ai.columns = ['Year', 'Mean_Citation_Distance']
mean_distance_by_year_non_ai.columns = ['Year', 'Mean_Citation_Distance']
mean_distance_by_year_all.columns = ['Year', 'Mean_Citation_Distance']

mean_distance_by_year_ai = mean_distance_by_year_ai[mean_distance_by_year_ai['Year'] <= 2023]
mean_distance_by_year_non_ai = mean_distance_by_year_non_ai[mean_distance_by_year_non_ai['Year'] <= 2023]
mean_distance_by_year_all = mean_distance_by_year_all[mean_distance_by_year_all['Year'] <= 2023]

# Plot mean citation distance for AI vs. Non-AI patents over time
plt.figure(figsize=(10, 5))
plt.plot(mean_distance_by_year_ai['Year'], mean_distance_by_year_ai['Mean_Citation_Distance'],
         marker='o', linestyle='-', label="AI Patents", color="blue")
plt.plot(mean_distance_by_year_non_ai['Year'], mean_distance_by_year_non_ai['Mean_Citation_Distance'],
         marker='s', linestyle='--', label="Non-AI Patents", color="red")
plt.plot(mean_distance_by_year_all['Year'], mean_distance_by_year_all['Mean_Citation_Distance'],
         marker='^', linestyle='-.', label="All Patents", color="green")

plt.xlabel("Publication Year")
plt.ylabel("Mean Citation Distance")
plt.title("Mean Citation Distance Over Time (AI vs. Non-AI Patents)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------- #

# Calculate in-state vs. out-of-state citation props in each state (normalized)
df_external_distance_ai = df_external_distance[df_external_distance['predict86_any_ai_x'] == 1]

# Group by state and citation pairs
state_citation_counts = df_external_distance_ai.groupby(['disambig_state_y', 'disambig_state_x']).size().reset_index(name='count')
state_total_citations = state_citation_counts.groupby('disambig_state_y')['count'].sum().reset_index(name='total_citations')

# Merge total citations onto state counts
state_citation_counts = state_citation_counts.merge(state_total_citations, on='disambig_state_y')

# Function to calculate in-state and out-of-state proportions
def total_citations_props(state):
    df_state = state_citation_counts[state_citation_counts['disambig_state_y'] == state]
    in_state_count = df_state[df_state['disambig_state_x'] == state]['count'].sum()
    out_of_state_count = df_state[df_state['disambig_state_x'] != state]['count'].sum()

    total_count = in_state_count + out_of_state_count
    in_state_prop = in_state_count / total_count
    out_of_state_prop = out_of_state_count / total_count

    # Normalize the proportions by the state's total citation volume
    total_citations = state_citation_counts[state_citation_counts['disambig_state_y'] == state]['total_citations'].iloc[0]
    in_state_prop_normalized = in_state_count / total_citations
    out_of_state_prop_normalized = out_of_state_count / total_citations

    return in_state_prop, out_of_state_prop, in_state_prop_normalized, out_of_state_prop_normalized

# Create the list of state citation proportions
state_citations = []
for state in state_citation_counts['disambig_state_y'].unique():
    in_state_prop, out_of_state_prop, in_state_prop_normalized, out_of_state_prop_normalized = total_citations_props(state)
    state_citations.append({'state': state, 'in_state_prop': in_state_prop, 'out_of_state_prop': out_of_state_prop,
                            'in_state_prop_normalized': in_state_prop_normalized, 'out_of_state_prop_normalized': out_of_state_prop_normalized})
state_citations_df = pd.DataFrame(state_citations)

# Filter to only U.S. states
us_states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

state_citations_df = state_citations_df[state_citations_df['state'].isin(us_states)]
state_citations_df.sort_values(by="out_of_state_prop_normalized", ascending=False, inplace=True)

# Plot with normalized proportions
plt.figure(figsize=(12, 6))
plt.bar(state_citations_df['state'], state_citations_df['out_of_state_prop_normalized'], label="Out-of-State Citations", color="royalblue")
plt.bar(state_citations_df['state'], state_citations_df['in_state_prop_normalized'], bottom=state_citations_df['out_of_state_prop_normalized'], label="In-State Citations", color="orange")

plt.xlabel("State")
plt.ylabel("Normalized Proportion of Citations")
plt.title("Normalized In-State vs Out-of-State Citations by State")
plt.xticks(rotation=90)
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------- #

# Calculate home bias (spillover) index for all patents

# Count and sum in state and out of state citations. Here it is how many times a filed patent gets cited by future ones from the same state.
def state_citation_props(state):
    # Calculate how many times each states cites itself and others
    state_citation_counts = df_external_distance.groupby(['disambig_state_y', 'disambig_state_x']).size().reset_index(name='count')
    
    df_state = state_citation_counts[state_citation_counts['disambig_state_x']==state]
    out_of_state_count = 0
    in_state_count = 0
    for _ , row in df_state.iterrows():
        if row['disambig_state_x'] != row['disambig_state_y']:
            out_of_state_count += row['count']
        else:
            in_state_count += row['count']
    out_of_state_prop = out_of_state_count / (out_of_state_count + in_state_count)
    in_state_prop = in_state_count / (out_of_state_count + in_state_count)
    return in_state_prop, out_of_state_prop

# Calculate spillover
def spillover_index(state):
    state_citation_counts = df_external_distance.groupby(['disambig_state_y', 'disambig_state_x']).size().reset_index(name='count')
    in_state_obs, _ = state_citation_props(state)
    
    # Compute Counterfactual Model (Random Citation Model) - count how many times each state cites the patent
    total_state_citation_counts = df_external_distance.groupby('disambig_state_x').size().reset_index(name='total_citations')

    # Normalize to get the expected probability of citing any state
    total_state_citation_counts['citation_prob'] = total_state_citation_counts['total_citations'] / total_state_citation_counts['total_citations'].sum()
    
    # Get the expected probability of citing a patent if citations were random
    if state in total_state_citation_counts['disambig_state_x'].values:
        in_state_exp = total_state_citation_counts[total_state_citation_counts['disambig_state_x'] == state]['citation_prob'].values[0]
    else:
        in_state_exp = 0
    spillover_bias = in_state_obs - in_state_exp
    print(f"Spillover Bias Score for {state}: {spillover_bias:.3f}")
    return spillover_bias

spillover_bias_data = []
for state in us_states:
    spillover_bias_data.append({'state': state, 'index': spillover_index(state)})
spillover_bias_df = pd.DataFrame(spillover_bias_data)

# Develop map using geopandas
# Load US shapefile for states
path = "/Users/jacksonvanvooren/Downloads/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
gdf = gpd.read_file(path)

# Merge spillover bias data with GeoDataFrame
merged = gdf.merge(spillover_bias_df, left_on='STUSPS', right_on='state', how='inner')
fig, ax = plt.subplots(1, 1, figsize=(20, 15))

# Exclude Alaska and Hawaii, I was struggling with scale when trying to include them
merged = merged[~merged['STUSPS'].isin(['AK', 'HI'])]
merged.plot(column='index', ax=ax, legend=True,
            legend_kwds={'label': "Home Bias by State",
                         'orientation': "horizontal",
                         'fraction': 0.03, 'pad': 0.04})

ax.set_axis_off()
ax.set_title('Home Bias by U.S. State', fontsize=20)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()


# ---------------------------------------------------------------------------------- #

# Dp the same spillover analysis for AI patents

ai_df_external_distance = df_external_distance[df_external_distance['predict86_any_ai_x']==1]

# Same as above, just filtered to AI patents now
def ai_state_citation_props(state):
    ai_state_citation_counts = ai_df_external_distance.groupby(['disambig_state_y', 'disambig_state_x']).size().reset_index(name='count')
    df_state = ai_state_citation_counts[ai_state_citation_counts['disambig_state_x']==state]
    out_of_state_count = 0
    in_state_count = 0
    for _ , row in df_state.iterrows():
        if row['disambig_state_x'] != row['disambig_state_y']:
            out_of_state_count += row['count']
        else:
            in_state_count += row['count']
    out_of_state_prop = out_of_state_count / (out_of_state_count + in_state_count)
    in_state_prop = in_state_count / (out_of_state_count + in_state_count)
    return in_state_prop, out_of_state_prop

# Same as above, just filtered to AI patents now
def ai_spillover_index(state):
    ai_state_citation_counts = ai_df_external_distance.groupby(['disambig_state_y', 'disambig_state_x']).size().reset_index(name='count')
    in_state_obs, _ = ai_state_citation_props(state)

    ai_total_state_citation_counts = ai_df_external_distance.groupby('disambig_state_x').size().reset_index(name='total_citations')
    ai_total_state_citation_counts['citation_prob'] = ai_total_state_citation_counts['total_citations'] / ai_total_state_citation_counts['total_citations'].sum()
    if state in ai_total_state_citation_counts['disambig_state_x'].values:
        in_state_exp = ai_total_state_citation_counts[ai_total_state_citation_counts['disambig_state_x'] == state]['citation_prob'].values[0]
    else:
        expected_ca_in_state_prob = 0
    spillover_bias = in_state_obs - in_state_exp
    print(f"Spillover Bias Score for {state}: {spillover_bias:.3f}")
    return spillover_bias

ai_spillover_bias_data = []
for state in us_states:
    ai_spillover_bias_data.append({'state': state, 'index': ai_spillover_index(state)})
ai_spillover_bias_df = pd.DataFrame(ai_spillover_bias_data)

# Develop map using the exact same code, just with AI patents
path = "/Users/jacksonvanvooren/Downloads/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
gdf = gpd.read_file(path)

merged = gdf.merge(ai_spillover_bias_df, left_on='STUSPS', right_on='state', how='inner')
fig, ax = plt.subplots(1, 1, figsize=(20, 15))  # Increase figure size
merged = merged[~merged['STUSPS'].isin(['AK', 'HI'])]
merged.plot(column='index', ax=ax, legend=True,
            legend_kwds={'label': "Home Bias by State",
                         'orientation': "horizontal",
                         'fraction': 0.03, 'pad': 0.04})

ax.set_axis_off()
ax.set_title('Home Bias in AI Patents by U.S. State', fontsize=20)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

# In retrospect I would have made these functions that took in different datasets to not repeat code

# Conduct a paired t-test for difference of means
ai_spillover = ai_spillover_bias_df['index']
spillover = spillover_bias_df['index']

t_stat, p_value = stats.ttest_rel(ai_spillover, spillover)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference is statistically significant (AI spillover bias is less than general spillover bias).")
else:
    print("There is no significant difference between AI spillover bias and general spillover bias.")
