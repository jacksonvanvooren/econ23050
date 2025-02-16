import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_patent = pd.read_csv("/Users/jacksonvanvooren/Downloads/g_patent.tsv", sep="\t")
df_assignee = pd.read_csv("/Users/jacksonvanvooren/Downloads/g_assignee_disambiguated.tsv", sep="\t")

chunksize = 100000
chunks = []
for chunk in pd.read_csv("/Users/jacksonvanvooren/Downloads/g_patent_abstract.tsv", sep="\t", chunksize=chunksize):
    chunks.append(chunk)
df_abstract = pd.concat(chunks, ignore_index=True)

# Merge patent and company data
df_merged = df_patent.merge(df_assignee, on='patent_id', how='inner')

# Filter to incumbent vs. new entrant corporations
incumbent_keywords = ['International Business Machines', 'Control Data',
                      'Digital Equipment', 'Data General', 'Memorex',
                      'Ampex', 'Pertec', 'Burroughs', 'Fujitsu', 'Hitachi',
                      'Univac']
df_incumbents = df_merged[df_merged['disambig_assignee_organization'].str.contains('|'.join(incumbent_keywords),
                                                                                   case=False, na=False)]

entrant_keywords = ['Seagate', 'Conner', 'Quantum', 'Shugart', 'Micropolis',
                    'Miniscribe', 'Prairietek', 'Wang', 'Prime', 'NCR', 'Nixdorf',
                    'Apple', 'Commodore', 'Compaq', 'Tandy', 'Sun Microsystems']
df_new_entrants = df_merged[df_merged['disambig_assignee_organization'].str.contains('|'.join(entrant_keywords),
                                                                                     case=False, na=False)]

# Company data (no abstracts yet)
# Group by year and count number of patents
df_incumbents['year'] = pd.to_datetime(df_incumbents['patent_date']).dt.year
df_new_entrants['year'] = pd.to_datetime(df_new_entrants['patent_date']).dt.year
incumbents_time_series = df_incumbents.groupby('year').size().reset_index(name='incumbent_patent_count')
new_entrants_time_series = df_new_entrants.groupby('year').size().reset_index(name='new_entrant_patent_count')
time_series_data = pd.merge(incumbents_time_series, new_entrants_time_series, on='year', how='outer').fillna(0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_series_data['year'], time_series_data['incumbent_patent_count'], label='Incumbents')
plt.plot(time_series_data['year'], time_series_data['new_entrant_patent_count'], label='New Entrants')
plt.xlabel('Year')
plt.ylabel('Number of Patents')
plt.title('Number of Patents Filed by Disk Manufacturers')
plt.legend()
plt.show()

# Company data with filtered abstracts
df_abstract.set_index('patent_id', inplace=True)

filtered_incumbents = df_incumbents.join(df_abstract[['patent_abstract']], on='patent_id', how='inner')
filtered_new_entrants = df_new_entrants.join(df_abstract[['patent_abstract']], on='patent_id', how='inner')

keywords = ["disk", "disc", "rigid", "drive", "storage", "magnetic"]
pattern = '|'.join(keywords)

filtered_incumbents = filtered_incumbents[filtered_incumbents['patent_abstract'].str.contains(pattern, case=False, na=False)]
filtered_new_entrants = filtered_new_entrants[filtered_new_entrants['patent_abstract'].str.contains(pattern, case=False, na=False)]

incumbents_time_series = filtered_incumbents.groupby('year').size().reset_index(name='incumbent_patent_count')
new_entrants_time_series = filtered_new_entrants.groupby('year').size().reset_index(name='new_entrant_patent_count')

time_series_data = pd.merge(incumbents_time_series, new_entrants_time_series, on='year', how='outer').fillna(0)
time_series_data = time_series_data[time_series_data['year'] <= 2010]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_series_data['year'], time_series_data['incumbent_patent_count'], label='Incumbents')
plt.plot(time_series_data['year'], time_series_data['new_entrant_patent_count'], label='New Entrants')
plt.xlabel('Year')
plt.ylabel('Number of Patents')
plt.title('Number of Patents Filed by Incumbents vs New Entrants')
plt.legend()
plt.show()
