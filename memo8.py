import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# US data
patent_data = pd.read_pickle("patent_data.pkl")
df_usa = patent_data[patent_data['disambig_country'] == 'US']

# India data
patent_data_india = patent_data[patent_data['disambig_country'] == 'IN']
df_india = df_assignee_india.merge(df_patent, on='patent_id', how='left')
df_india = df_india[['patent_id', 'patent_date', 'patent_title', 'disambig_assignee_organization', 
                     'disambig_city', 'disambig_state', 'disambig_country']]
df_india = df_india[df_india['patent_date'].notna()]

# Patent growth percent changes year over year
df_usa['patent_date'] = pd.to_datetime(df_usa['patent_date'])
df_india['patent_date'] = pd.to_datetime(df_india['patent_date'])

df_usa['filing_year'] = df_usa['patent_date'].dt.year
df_india['filing_year'] = df_india['patent_date'].dt.year

patent_trends_usa = df_usa.groupby('filing_year').size().reset_index(name='num_patents_usa')
patent_trends_india = df_india.groupby('filing_year').size().reset_index(name='num_patents_india')

patent_trends_usa['pct_change_usa'] = patent_trends_usa['num_patents_usa'].pct_change() * 100
patent_trends_india['pct_change_india'] = patent_trends_india['num_patents_india'].pct_change() * 100
patent_trends = patent_trends_usa.merge(patent_trends_india, on='filing_year', how='outer')

# See companies that stopped filing patents
threshold = 5
recent_year = 2024  # Adjust based on the most recent year in the dataset

# Find the last filing year for each company in the USA
last_filing_year_usa = df_usa.groupby('disambig_assignee_organization')['filing_year'].max().reset_index(name='last_filing_year')
last_filing_year_usa['stopped_filing'] = last_filing_year_usa['last_filing_year'] < (recent_year - threshold)

# Find the last filing year for each company in India
last_filing_year_india = df_india.groupby('disambig_assignee_organization')['filing_year'].max().reset_index(name='last_filing_year')
last_filing_year_india['stopped_filing'] = last_filing_year_india['last_filing_year'] < (recent_year - threshold)

total_usa_companies = last_filing_year_usa.shape[0]
stopped_filing_usa = last_filing_year_usa['stopped_filing'].sum()

total_india_companies = last_filing_year_india.shape[0]
stopped_filing_india = last_filing_year_india['stopped_filing'].sum()

proportion_stopped_filing_usa = stopped_filing_usa / total_usa_companies if total_usa_companies > 0 else 0
proportion_stopped_filing_india = stopped_filing_india / total_india_companies if total_india_companies > 0 else 0

print(f"Proportion of companies in the USA that stopped filing patents: {proportion_stopped_filing_usa:.2%}")
print(f"Proportion of companies in India that stopped filing patents: {proportion_stopped_filing_india:.2%}")

# Make plots (one raw changes and one rolling average)
rolling_window = 5

patent_trends['rolling_avg_usa'] = patent_trends['pct_change_usa'].rolling(window=rolling_window, min_periods=1).mean()
patent_trends['rolling_avg_india'] = patent_trends['pct_change_india'].rolling(window=rolling_window, min_periods=1).mean()

patent_trends_filtered = patent_trends[(patent_trends['filing_year'] >= 2007) & 
                                       (patent_trends['filing_year'] <= 2023)]

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# First plot: Year-over-Year % Change
axes[0].plot(patent_trends_filtered['filing_year'], patent_trends_filtered['pct_change_usa'], 
             label='USA (Annual % Change)', color='blue', marker='o', linestyle='dashed', alpha=0.7)
axes[0].plot(patent_trends_filtered['filing_year'], patent_trends_filtered['pct_change_india'], 
             label='India (Annual % Change)', color='green', marker='x', linestyle='dashed', alpha=0.7)

axes[0].set_title('Year-over-Year Patent Growth: USA vs India', fontsize=14)
axes[0].set_ylabel('Percent Change in Patents', fontsize=12)
axes[0].legend()
axes[0].grid(True)

# Second plot: 5-Year Rolling Average
axes[1].plot(patent_trends_filtered['filing_year'], patent_trends_filtered['rolling_avg_usa'], 
             label='USA (5-year Avg)', color='blue', linewidth=2)
axes[1].plot(patent_trends_filtered['filing_year'], patent_trends_filtered['rolling_avg_india'], 
             label='India (5-year Avg)', color='green', linewidth=2)

axes[1].set_title('5-Year Rolling Average Patent Growth: USA vs India', fontsize=14)
axes[1].set_xlabel('Filing Year', fontsize=12)
axes[1].set_ylabel('Smoothed Percent Change', fontsize=12)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
