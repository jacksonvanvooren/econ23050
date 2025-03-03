import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load dataset (I've pickled this, happy to provide data cleaning code if interested)
patent_data = pd.read_pickle("patent_data.pkl")

# Extract year and calculate age for normalization
current_year = datetime.now().year
patent_data['year'] = pd.to_datetime(patent_data['patent_date']).dt.year
patent_data['age'] = current_year - patent_data['year'] + 1  # Avoids division by zero

# External (forwards) citations
patent_data['fw_citation_rate'] = patent_data['forward_citation_count'] / patent_data['age']
yearly_citation_rate = patent_data.groupby('year')['fw_citation_rate'].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_citation_rate.index, yearly_citation_rate.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Normalized Forward Citations per Year")
plt.title("Normalized Forward Citation Rate of AI Patents Over Time")
plt.grid(True)
plt.show()

# Backwards citations
patent_data['bw_citation_rate'] = patent_data['us_backward_citation_count'] / patent_data['age']
yearly_citation_rate = patent_data.groupby('year')['bw_citation_rate'].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_citation_rate.index, yearly_citation_rate.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Normalized Backwards Citations per Year")
plt.title("Normalized Backwards Citation Rate of AI Patents Over Time")
plt.grid(True)
plt.show()
