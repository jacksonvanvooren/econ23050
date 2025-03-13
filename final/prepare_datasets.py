### Data Preprocessing from Raw Patent Data ###

# Import packages
import pandas as pd
from datetime import datetime
from haversine import haversine, Unit

# ---------------------------------------------------------------------------------- #
# Process raw patent information and AI data
# Define chunk size for memory issues
chunk_size = 100000

# AI patents by USPTO AI Patents Dataset
file_path = "/Users/jacksonvanvooren/Desktop/Econ23050Project/Data/ai_model_predictions.csv"  
chunks = []
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunks.append(chunk)
    del chunk
df_ai_model = pd.concat(chunks, ignore_index=True)
del chunks

# Assignee data from PatentsView
file_path = "/Users/jacksonvanvooren/Desktop/Econ23050Project/Data/g_assignee_disambiguated.tsv"
chunks = []
for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size):
    chunks.append(chunk)
    del chunk
df_assignee = pd.concat(chunks, ignore_index=True)
del chunks

# Application data from PatentsView
file_path = "/Users/jacksonvanvooren/Desktop/Econ23050Project/Data/g_application.tsv"
chunks = []
for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size):
    chunks.append(chunk)
    del chunk
df_application = pd.concat(chunks, ignore_index=True)
del chunks

# Location data from PatentsView
file_path = "/Users/jacksonvanvooren/Desktop/Econ23050Project/Data/g_location_disambiguated.tsv"
chunks = []
for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size):
    chunks.append(chunk)
    del chunk
df_location = pd.concat(chunks, ignore_index=True)
del chunks

# Merges all PatentsView Datasets
# df_ai_model must be merged on application identifier, all others on granted patent identifier
df_application_merge = df_ai_model.merge(df_application, how='inner', left_on='appl_id', right_on='application_id')
df_assignee_merge = df_application_merge.merge(df_assignee, how='inner', on='patent_id')
df_location_merge = df_assignee_merge.merge(df_location, how='left', on='location_id')

# Keep relevant columns
columns_to_keep = ['patent_id', 'pub_dt', 'disambig_assignee_organization',
       'disambig_state', 'disambig_country', 'latitude',
       'longitude', 'predict86_any_ai']
df = df_location_merge[columns_to_keep]

# Extract the year from 'pub_dt' column
df['pub_dt'] = pd.to_datetime(df['pub_dt'])
df['pub_year'] = df['pub_dt'].dt.year
df = df.drop_duplicates(subset=['patent_id'], keep='first')

# Limit dataset from memory issues
df_ai_patent = df[(df['pub_year'] >= 2000) & (df['pub_year'] <= 2023)]

# ---------------------------------------------------------------------------------- #

# Process citation data, this one takes a long time
file_path = "/Users/jacksonvanvooren/Desktop/Econ23050Project/Data/g_us_patent_citation.tsv.zip"
chunks = []
for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size, usecols=['patent_id', 'citation_patent_id'], compression="zip"):
    chunks.append(chunk)
    del chunk
citation_df = pd.concat(chunks, ignore_index=True)
del chunks

# Integer string mismatches in the raw data
df_ai_patent['patent_id'] = df_ai_patent['patent_id'].astype(str)
citation_df['patent_id'] = citation_df['patent_id'].astype(str)
citation_df['citation_patent_id'] = citation_df['citation_patent_id'].astype(str)

# Get citing and cited patent identifiers
df_patent_ids = set(df_ai_patent['patent_id'])
df_citation_patent_ids = citation_df[citation_df['citation_patent_id'].isin(df_patent_ids)]

# Merge patent information for citing patents
df_citation_merge = df_ai_patent.merge(df_citation_patent_ids, left_on='patent_id', right_on='citation_patent_id', how='left')
df_citation_merge_2 = df_citation_merge.merge(df_ai_patent, left_on='patent_id_y', right_on='patent_id', how='inner')
columns_to_keep = ['patent_id_x', 'pub_year_x', 'disambig_assignee_organization_x',
                   'disambig_state_x', 'disambig_country_x', 'predict86_any_ai_x',
                   'patent_id_y', 'pub_year_y', 'disambig_assignee_organization_y',
                   'disambig_state_y', 'disambig_country_y', 'predict86_any_ai_y']
df_citation_merge_2 = df_citation_merge_2[columns_to_keep]

# Count forward citations and merge
# df_citation_merge_2 does NOT account for self citations where the company cites itself
columns_to_keep = ['patent_id_x']
df_citation_forwardcounts = df_citation_merge_2[columns_to_keep]
df_grouped_forward = df_citation_forwardcounts.groupby('patent_id_x').size().reset_index(name='forward_citations')
df_forward = df_ai_patent.merge(df_grouped_forward, left_on='patent_id', right_on='patent_id_x', how='left')
df_forward['forward_citations'] = df_forward['forward_citations'].fillna(0) # Set NaNs to 0
df_forward.drop(columns=['patent_id_x'], inplace=True)

# Count external citations and merge
# Filter out self citations to get external counts
df_citation_merge_3 = df_citation_merge_2[df_citation_merge_2['disambig_assignee_organization_x'] != df_citation_merge_2['disambig_assignee_organization_y']]
columns_to_keep = ['patent_id_x', 'pub_year_x', 'disambig_assignee_organization_x', 'disambig_state_x',
                   'disambig_country_x', 'predict86_any_ai_x', 'patent_id_y', 'pub_year_y',
                   'disambig_assignee_organization_y', 'disambig_state_y', 'disambig_country_y', 'predict86_any_ai_y']
df_citation_merge_3 = df_citation_merge_3[columns_to_keep]
columns_to_keep = ['patent_id_x']
df_citation_external_counts = df_citation_merge_3[columns_to_keep]
df_grouped_external = df_citation_external_counts.groupby('patent_id_x').size().reset_index(name='external_citations')
df_external = df_forward.merge(df_grouped_external, left_on='patent_id', right_on='patent_id_x', how='left')
df_external['external_citations'] = df_external['external_citations'].fillna(0)
df_external.drop(columns=['patent_id_x'], inplace=True)
df_external['self_citations'] = df_external['forward_citations'] - df_external['external_citations'] # Self citation count

# Select relevant columns
# This dataset is huge and computationally difficult to deal with
columns_to_keep = ['patent_id_x', 'pub_year_x', 'disambig_state_x', 'disambig_country_x', 'predict86_any_ai_x', 'patent_id_y', 'pub_year_y', 'disambig_state_y', 'disambig_country_y', 'predict86_any_ai_y']
df_citation_merge_3 = df_citation_merge_3[columns_to_keep]

# Calculate the Haversine distance between citing and cited patents
df_citation_merge_3['distance_citation'] = df_citation_merge_3.apply(
    lambda row: haversine(
        (row['latitude_x'], row['longitude_x']),
        (row['latitude_y'], row['longitude_y']),
        unit=Unit.MILES
    ),
    axis=1
)

# More selecting columns and removing empty rows
columns_to_keep=['patent_id_x', 'pub_year_x', 'disambig_assignee_organization_x',
       'disambig_state_x', 'disambig_country_x', 'predict86_any_ai_x', 'patent_id_y', 'pub_year_y', 'disambig_assignee_organization_y',
       'disambig_state_y', 'disambig_country_y', 'predict86_any_ai_y', 'distance_citation']
df_citation_merged_4 = df_citation_merge_3[columns_to_keep]
df_citation_merged_5 = df_citation_merged_4.dropna(subset=['distance_citation'])

# Export to CSV
# This dataset has rows with cited and citing patent information, so there are n rows for a patent that receives n citations
# I've uploaded df_external_distance.csv into the Github
df_citation_merged_5.to_csv('df_external_distance.csv', index=False)

# Aggregate by patent identifiers to add mean distances
mean_distance_df = df_citation_merged_5.groupby('patent_id_x')['distance_citation'].mean().reset_index()
mean_distance_df = mean_distance_df.rename(columns={'distance_citation': 'mean_distance'})
df_citations_distance = df_external.merge(mean_distance_df, left_on='patent_id', right_on='patent_id_x', how='left')
df_citations_distance.drop(columns=['patent_id_x'], inplace=True)

# Export to CSV
# This dataset has one row per unique patent identifier, with counts of citations and mean citing to cited distances
# I've uploaded df_citation_w_distance to the Github
df_citations_distance.to_csv('df_citation_w_distance', index=False)
