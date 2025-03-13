### This code uses my datasets to generate graphs used in the methodology/dataset portions of the paper ###
### Presents already known trends in the literature using my datasets ###

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------------- #
df_citation_distance = pd.read_csv('/Users/jacksonvanvooren/Desktop/df_citation_w_distance.csv')

# Show share of AI patents (number AI/total) over time
ai_share_by_year = df_citation_distance.groupby('pub_year').agg(
    total_patents=('predict86_any_ai', 'size'),  # Total patents per year
    ai_patents=('predict86_any_ai', 'sum')  # Sum of AI patents per year
)

ai_share_by_year['ai_share'] = ai_share_by_year['ai_patents'] / ai_share_by_year['total_patents']

plt.figure(figsize=(10, 6))
plt.plot(ai_share_by_year.index, ai_share_by_year['ai_share'], marker='', color='black', linestyle='-', linewidth=2)
plt.title('Share of AI Patents Over Time')
plt.xlabel('Year')
plt.ylabel('Share of AI Patents')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------- #

# Heatmap of US patent density over time
def create_us_patent_heatmap(df, year):
    # Filter for continental US coordinates only (excluding Alaska and Hawaii)
    continental_us_bounds = {
        'min_lat': 24.0, 'max_lat': 50.0,
        'min_lon': -125.0, 'max_lon': -66.0
    }
    
    us_df = df[
        (df['latitude'] >= continental_us_bounds['min_lat']) & 
        (df['latitude'] <= continental_us_bounds['max_lat']) &
        (df['longitude'] >= continental_us_bounds['min_lon']) & 
        (df['longitude'] <= continental_us_bounds['max_lon'])
    ]
    
    print(f"Plotting {len(us_df)} patents within continental US bounds")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Get US background
    us_shapefile = '/Users/jacksonvanvooren/Downloads/ne_110m_admin_0_countries 2/ne_110m_admin_0_countries.shp'
    world = gpd.read_file(us_shapefile)
    usa = world[world['NAME'] == 'United States of America']
    
    usa.plot(ax=ax, color='lightgray', edgecolor='white')
    ax.set_xlim(continental_us_bounds['min_lon'], continental_us_bounds['max_lon'])
    ax.set_ylim(continental_us_bounds['min_lat'], continental_us_bounds['max_lat'])
    
    x = us_df['longitude']
    y = us_df['latitude']
    xi, yi = np.mgrid[
        continental_us_bounds['min_lon']:continental_us_bounds['max_lon']:100j, 
        continental_us_bounds['min_lat']:continental_us_bounds['max_lat']:100j
    ]
    
    # Perform KDE for density
    positions = np.vstack([xi.ravel(), yi.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, xi.shape)
    
    custom_cmap = plt.cm.get_cmap('YlOrRd', 20)
    levels = np.linspace(z.min(), z.max(), 21)
    contour = ax.contourf(xi, yi, z, levels=levels, cmap=custom_cmap, alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Patent Density')
    
    # Try to add state boundaries for better reference (been issues getting rid of Canada)
    try:
        # Get state shapefile
        states_shapefile = '/Users/jacksonvanvooren/Downloads/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
        states = gpd.read_file(states_shapefile)
        us_states = states[states['admin'] == 'United States of America']
        us_states.boundary.plot(ax=ax, linewidth=0.5, color='gray')
    except Exception as e:
        print(f"Could not add state boundaries: {e}")
        print("Continuing without state boundaries")
        ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title(f'US Patent Density Heatmap for {year}', fontsize=16)
  
    stats_text = f"Total Patents: {len(us_df):,}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    plt.savefig('us_patent_heatmap.png', dpi=300)
    plt.show()
    
    return fig, ax

# Loop through year and generate a heatmap for each
for year in range(2000,2024):
    df_citation_distance_year = df_citation_distance[df_citation_distance['pub_year']==year]
    create_us_patent_heatmap(df_citation_distance_year, year)
