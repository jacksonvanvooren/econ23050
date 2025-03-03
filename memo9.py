import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load dataset (I've pickled this - happy to provide data cleaning code)
patent_data = pd.read_pickle("patent_data.pkl")

# Extract year from patent_date
current_year = datetime.now().year
patent_data['year'] = pd.to_datetime(patent_data['patent_date']).dt.year
patent_data['age'] = current_year - patent_data['year'] + 1  # Avoids division by zero

# Forward citations (greater than 5)
fw_cited_patents = patent_data[patent_data['forward_citation_count'] > 5]
fw_cited_patents['fw_citation_rate'] = fw_cited_patents['forward_citation_count'] / fw_cited_patents['age']
yearly_fw_citation_rate = fw_cited_patents.groupby('year')['fw_citation_rate'].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_fw_citation_rate.index, yearly_fw_citation_rate.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Normalized Forward Citations per Year")
plt.title("Normalized Forward Citation Rate of AI Patents Over Time")
plt.grid(True)
plt.show()

# Backward citations (greater than 5)
bw_cited_patents = patent_data[patent_data['us_backward_citation_count'] > 5]
bw_cited_patents['bw_citation_rate'] = bw_cited_patents['us_backward_citation_count'] / bw_cited_patents['age']
yearly_bw_citation_rate = bw_cited_patents.groupby('year')['bw_citation_rate'].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_bw_citation_rate.index, yearly_bw_citation_rate.values, marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Normalized Backward Citations per Year (Excluding Zero-Citation Patents)")
plt.title("Normalized Backward Citation Rate of AI Patents Over Time")
plt.grid(True)
plt.show()
