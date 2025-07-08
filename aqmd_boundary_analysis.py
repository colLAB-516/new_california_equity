import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import urllib.request
import zipfile
import os

# Step 1: Download and load AQMD boundaries
def download_aqmd_boundaries():
    """Download California Air District boundaries from CARB"""
    url = "https://ww3.arb.ca.gov/ei/gislib/boundaries/ca_air_district.zip"
    local_filename = "ca_air_district.zip"
    
    if not os.path.exists(local_filename):
        print("Downloading AQMD boundaries...")
        urllib.request.urlretrieve(url, local_filename)
        print("Download complete.")
    
    # Load the shapefile
    aqmd_gdf = gpd.read_file(local_filename)
    return aqmd_gdf

# Step 2: Download Census Tract boundaries for fallback
def download_census_tracts():
    """Download Census tract boundaries for spatial fallback"""
    try:
        # Try to get CA census tracts from Census Bureau
        census_url = "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_06_tract.zip"
        local_filename = "ca_census_tracts.zip"
        
        if not os.path.exists(local_filename):
            print("Downloading Census tract boundaries...")
            urllib.request.urlretrieve(census_url, local_filename)
            print("Census tract download complete.")
        
        tract_gdf = gpd.read_file(local_filename)
        tract_gdf['GEOID'] = tract_gdf['GEOID'].astype(str)
        return tract_gdf
    except:
        print("Could not download Census tracts - will use lat/long only")
        return None

# Step 3: Parse coordinates and create spatial points
def create_project_geometries(df, lat_long_col='Lat Long', census_col='CensusTract'):
    """
    Create Point geometries using lat/long when available, census tract centroids as fallback
    """
    df_work = df.copy()
    
    # Parse lat/long (format: "38.583288, -121.533046")
    coords = df_work[lat_long_col].str.strip()
    lat_lon_split = coords.str.split(',', expand=True)
    df_work['latitude'] = pd.to_numeric(lat_lon_split[0].str.strip(), errors='coerce')
    df_work['longitude'] = pd.to_numeric(lat_lon_split[1].str.strip(), errors='coerce')
    
    # Clean census tract data
    df_work['census_tract_clean'] = df_work[census_col].astype(str).str.strip()
    
    print(f"Projects with lat/long: {df_work[['latitude', 'longitude']].dropna().shape[0]}")
    print(f"Projects with census tract: {df_work['census_tract_clean'].notna().sum()}")
    
    # Create geometries for projects with lat/long
    has_coords = df_work.dropna(subset=['latitude', 'longitude'])
    geometry_coords = [Point(xy) for xy in zip(has_coords['longitude'], has_coords['latitude'])]
    
    # Try to get census tract centroids for projects without lat/long
    tract_gdf = download_census_tracts()
    
    if tract_gdf is not None:
        # Get centroids for census tracts (project to appropriate CRS first)
        tract_centroids = tract_gdf.copy()
        # Project to California Albers Equal Area for accurate centroids
        tract_centroids_proj = tract_centroids.to_crs('EPSG:3310')
        tract_centroids_proj['centroid'] = tract_centroids_proj.geometry.centroid
        # Convert back to WGS84
        tract_centroids_proj = tract_centroids_proj.to_crs('EPSG:4326')
        tract_lookup = dict(zip(tract_centroids['GEOID'], tract_centroids_proj['centroid']))
        
        # Fill missing coordinates with census tract centroids
        missing_coords = df_work[df_work[['latitude', 'longitude']].isna().any(axis=1)]
        
        print(f"Attempting to fill {len(missing_coords)} missing coordinates using census tracts...")
        
        # Debug: check tract ID formats
        sample_project_tracts = df_work['census_tract_clean'].dropna().head(5).tolist()
        sample_shapefile_tracts = list(tract_lookup.keys())[:5]
        print(f"Sample project tract IDs: {sample_project_tracts}")
        print(f"Sample shapefile tract IDs: {sample_shapefile_tracts}")
        
        geometries_all = []
        df_final = []
        
        # Add projects with direct lat/long
        for idx, row in has_coords.iterrows():
            row_copy = row.copy()
            row_copy['coord_source'] = 'lat_long'
            geometries_all.append(Point(row['longitude'], row['latitude']))
            df_final.append(row_copy)
        
        # Add projects using census tract centroids
        filled_from_census = 0
        for idx, row in missing_coords.iterrows():
            # Clean the census tract ID - your data has format like '6019001201.0'
            tract_raw = str(row['census_tract_clean']).split('.')[0]  # Remove decimal part
            # Ensure it's 11 digits (standard GEOID format)
            if pd.notna(tract_raw) and tract_raw != 'nan' and len(tract_raw) >= 10:
                # Your tracts start with '60' (California FIPS code), others might be '6'
                if len(tract_raw) == 10:
                    tract = '0' + tract_raw  # Pad to 11 digits
                else:
                    tract = tract_raw
                    
                if tract in tract_lookup:
                    centroid = tract_lookup[tract]
                    row_updated = row.copy()
                    row_updated['latitude'] = centroid.y
                    row_updated['longitude'] = centroid.x
                    row_updated['coord_source'] = 'census_tract'
                    geometries_all.append(centroid)
                    df_final.append(row_updated)
                    filled_from_census += 1
            
        print(f"Successfully filled {filled_from_census} coordinates from census tracts")
        df_final = pd.DataFrame(df_final)
        
    else:
        # Fallback to lat/long only
        df_final = has_coords.copy()
        df_final['coord_source'] = 'lat_long'
        geometries_all = geometry_coords
    
    # Create GeoDataFrame
    projects_gdf = gpd.GeoDataFrame(df_final, geometry=geometries_all, crs='EPSG:4326')
    
    print(f"Final dataset: {len(projects_gdf)} projects with geometries")
    print(f"Coordinate sources: {projects_gdf['coord_source'].value_counts()}")
    
    return projects_gdf

# Step 4: Map projects to AQMDs
def map_projects_to_aqmd(projects_gdf, aqmd_gdf):
    """Spatial join projects with AQMD boundaries"""
    
    # First, check what columns are available in the AQMD data
    print(f"AQMD shapefile columns: {list(aqmd_gdf.columns)}")
    
    # Ensure both use same CRS
    aqmd_gdf = aqmd_gdf.to_crs('EPSG:4326')
    
    # Find the district name column - try common variations
    name_columns = [col for col in aqmd_gdf.columns if any(x in col.upper() for x in ['DIST', 'NAME', 'AQMD'])]
    print(f"Potential name columns: {name_columns}")
    
    if name_columns:
        district_col = name_columns[0]  # Use the first match
    else:
        # Use all columns except geometry for now
        district_col = [col for col in aqmd_gdf.columns if col != 'geometry'][0]
    
    print(f"Using district column: {district_col}")
    
    # Spatial join
    projects_with_aqmd = gpd.sjoin(projects_gdf, aqmd_gdf[[district_col, 'geometry']], 
                                   how='left', predicate='within')
    
    # Clean up column names
    projects_with_aqmd = projects_with_aqmd.rename(columns={district_col: 'AQMD'})
    
    return projects_with_aqmd

# Step 5: Analyze collaboration patterns
def analyze_aqmd_collaboration(df):
    """
    Analyze both within-AQMD and cross-AQMD collaboration patterns
    """
    # Create project-level aggregations
    project_stats = df.groupby('Project ID Number').agg({
        'AQMD': ['first', 'nunique'],  # Lead AQMD and number of AQMDs
        'County': 'nunique',           # Number of counties (keep for comparison)
        'Agency Name': 'first',
        'Total Program GGRFFunding': 'first',
        'Total Project GHGReductions': 'first',
        'Total GGRFDisadvantaged Community Funding': 'first'
    }).reset_index()
    
    # Flatten column names
    project_stats.columns = [
        'Project ID Number', 'Lead_AQMD', 'n_aqmd_partners', 
        'n_county_partners', 'Agency_Name', 'Total_Funding', 
        'Total_GHG_Reductions', 'DAC_Funding'
    ]
    
    # Calculate derived variables (with division checks)
    project_stats['cost_per_ton'] = np.where(
        (project_stats['Total_GHG_Reductions'] > 0) & (project_stats['Total_Funding'] > 0),
        project_stats['Total_Funding'] / project_stats['Total_GHG_Reductions'],
        np.nan
    )
    project_stats['share_DAC'] = np.where(
        project_stats['Total_Funding'] > 0,
        project_stats['DAC_Funding'] / project_stats['Total_Funding'],
        np.nan
    )
    project_stats['log_funding'] = np.log1p(project_stats['Total_Funding'])

    # Clean up any rogue inf/-inf (paranoia)
    project_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Define collaboration levels (adjusted thresholds for larger sample sizes)
    project_stats['cross_aqmd_collab'] = (project_stats['n_aqmd_partners'] > 1).astype(int)
    project_stats['high_county_collab'] = (project_stats['n_county_partners'] > 3).astype(int)  # Lowered from 5
    
    # Focus on Southern California AQMDs (using actual names from shapefile)
    socal_aqmds = [
        'South Coast',           # South Coast AQMD 
        'Ventura',              # Ventura County APCD
        'Imperial',             # Imperial County APCD
        'Mojave Desert',        # Mojave Desert AQMD
        'San Diego'             # San Diego APCD (also SoCal)
    ]
    
    project_stats['SoCal_AQMD'] = project_stats['Lead_AQMD'].isin(socal_aqmds).astype(int)
    
    return project_stats

# Step 6: Run propensity score matching analysis
def run_aqmd_psm_analysis(project_stats, treatment_var='cross_aqmd_collab'):
    """
    Run PSM analysis using AQMD-based collaboration as treatment
    """
    # Clean data
    analysis_df = project_stats.dropna(subset=[
        'log_funding', 'Agency_Name', 'cost_per_ton', 'share_DAC', treatment_var
    ])
    
    # Remove infinite values
    analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Prepare covariates for PSM
    covariates = ['log_funding', 'Agency_Name', 'SoCal_AQMD']
    X = pd.get_dummies(analysis_df[covariates], drop_first=True).astype(float)
    y = analysis_df[treatment_var].astype(int)
    
    # Fit propensity score model
    try:
        ps_model = sm.Logit(y, sm.add_constant(X)).fit(method='lbfgs', maxiter=500, disp=0)
        analysis_df['propensity'] = ps_model.predict(sm.add_constant(X))
        
        # Perform matching
        treated = analysis_df[analysis_df[treatment_var] == 1]
        control = analysis_df[analysis_df[treatment_var] == 0]
        
        matches = []
        for idx, p in treated['propensity'].items():
            closest_idx = (control['propensity'] - p).abs().idxmin()
            matches.append((idx, closest_idx))
        
        matched_idx = [i for pair in matches for i in pair]
        matched_sample = analysis_df.loc[matched_idx]
        
        # Winsorize outcomes
        costs = matched_sample['cost_per_ton'].copy()
        costs_wins = winsorize(costs.values, limits=[0.01, 0.01])
        matched_sample['cost_per_ton_wins'] = costs_wins
        
        return matched_sample, ps_model
        
    except Exception as e:
        print(f"PSM failed: {e}")
        return None, None

# Step 7: Generate results
def generate_aqmd_results(matched_sample, treatment_var='cross_aqmd_collab'):
    """Generate and display results"""
    
    if matched_sample is None:
        print("No matched sample available")
        return
    
    treated = matched_sample[matched_sample[treatment_var] == 1]
    control = matched_sample[matched_sample[treatment_var] == 0]
    
    print(f"\n=== AQMD Collaboration Analysis Results ===")
    print(f"Treatment: {treatment_var}")
    print(f"Treated projects: {len(treated)}")
    print(f"Control projects: {len(control)}")
    print()
    
    # Cost effectiveness results
    print("Cost per Ton GHG Reduction:")
    if treatment_var == 'cross_aqmd_collab':
        print(f"  Cross-AQMD collaborative: ${treated['cost_per_ton'].mean():.2f}")
        print(f"  Single-AQMD: ${control['cost_per_ton'].mean():.2f}")
    else:
        print(f"  High county collaborative: ${treated['cost_per_ton'].mean():.2f}")
        print(f"  Low county collaborative: ${control['cost_per_ton'].mean():.2f}")
    print(f"  Difference: ${control['cost_per_ton'].mean() - treated['cost_per_ton'].mean():.2f}")
    print()
    
    # Equity results  
    print("DAC Funding Share:")
    if treatment_var == 'cross_aqmd_collab':
        print(f"  Cross-AQMD collaborative: {treated['share_DAC'].mean():.1%}")
        print(f"  Single-AQMD: {control['share_DAC'].mean():.1%}")
    else:
        print(f"  High county collaborative: {treated['share_DAC'].mean():.1%}")
        print(f"  Low county collaborative: {control['share_DAC'].mean():.1%}")
    print(f"  Difference: {treated['share_DAC'].mean() - control['share_DAC'].mean():.1%}")
    
    return treated, control

# Quick test function
def test_data_format(df, n_sample=5):
    """Test the data format and show sample coordinates"""
    print("=== Data Format Check ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("Sample Lat Long values:")
    sample_coords = df['Lat Long'].dropna().head(n_sample)
    for i, coord in enumerate(sample_coords):
        print(f"  {i+1}: '{coord}'")
    print()
    
    print(f"Missing Lat Long: {df['Lat Long'].isna().sum()} ({df['Lat Long'].isna().sum()/len(df):.1%})")
    print(f"Missing Census Tract: {df['CensusTract'].isna().sum()} ({df['CensusTract'].isna().sum()/len(df):.1%})")
    print()
    
    print("Sample Census Tract values:")
    sample_tracts = df['CensusTract'].dropna().head(n_sample)
    for i, tract in enumerate(sample_tracts):
        print(f"  {i+1}: '{tract}'")

# Main execution function
def main():
    """
    Main function to run the complete AQMD analysis
    """
    
    print("Starting AQMD Spatial Analysis...")
    
    # Download AQMD boundaries
    aqmd_gdf = download_aqmd_boundaries()
    print(f"Loaded {len(aqmd_gdf)} Air Districts")
    
    # Load CCI data
    print("\nLoading CCI data...")
    try:
        df = pd.read_csv('cci_programs_data_reduced.csv', low_memory=False)
        print(f"Loaded {len(df)} rows of CCI data")
    except FileNotFoundError:
        print("Error: cci_programs_data_reduced.csv not found in current directory")
        print("Please ensure your CCI data file is in the same directory as this script")
        return
    
    # Test data format
    print("\n" + "="*50)
    test_data_format(df)
    print("="*50)
    
    # Step 2: Create spatial geometries
    print("\nStep 2: Creating spatial geometries...")
    projects_gdf = create_project_geometries(df)
    
    # Step 3: Map to AQMDs  
    print("\nStep 3: Mapping projects to AQMDs...")
    projects_with_aqmd = map_projects_to_aqmd(projects_gdf, aqmd_gdf)
    
    # Check mapping results
    print(f"\nProjects successfully mapped to AQMDs: {projects_with_aqmd['AQMD'].notna().sum()}")
    print(f"Projects without AQMD mapping: {projects_with_aqmd['AQMD'].isna().sum()}")
    
    print("\nTop AQMDs by project count:")
    aqmd_counts = projects_with_aqmd['AQMD'].value_counts().head(10)
    for aqmd, count in aqmd_counts.items():
        print(f"  {aqmd}: {count} projects")
    
    # Step 4: Analyze collaboration patterns
    print("\nStep 4: Analyzing collaboration patterns...")
    project_stats = analyze_aqmd_collaboration(projects_with_aqmd)
    
    print(f"\nProject-level statistics:")
    print(f"  Total projects: {len(project_stats)}")
    print(f"  Cross-AQMD projects: {project_stats['cross_aqmd_collab'].sum()}")
    print(f"  High county collaboration (>3 counties): {project_stats['high_county_collab'].sum()}")
    print(f"  Southern California projects: {project_stats['SoCal_AQMD'].sum()}")
    
    # Show AQMD distribution
    print(f"\nTop Southern California AQMDs:")
    socal_subset = project_stats[project_stats['SoCal_AQMD'] == 1]
    if len(socal_subset) > 0:
        socal_counts = socal_subset['Lead_AQMD'].value_counts()
        for aqmd, count in socal_counts.items():
            print(f"  {aqmd}: {count} projects")
    
    # Step 5: Run PSM analysis for cross-AQMD collaboration
    print("\nStep 5a: Running PSM analysis for cross-AQMD collaboration...")
    matched_sample_aqmd, model_aqmd = run_aqmd_psm_analysis(project_stats, 'cross_aqmd_collab')
    
    if matched_sample_aqmd is not None:
        print("\nStep 6a: Cross-AQMD Collaboration Results:")
        treated_aqmd, control_aqmd = generate_aqmd_results(matched_sample_aqmd, 'cross_aqmd_collab')
    else:
        print("Cross-AQMD PSM analysis failed - insufficient data or convergence issues")
    
    # Step 5b: Run PSM analysis for county-based collaboration (for comparison)
    print("\nStep 5b: Running PSM analysis for county-based collaboration...")
    matched_sample_county, model_county = run_aqmd_psm_analysis(project_stats, 'high_county_collab')
    
    if matched_sample_county is not None:
        print("\nStep 6b: County-Based Collaboration Results:")
        treated_county, control_county = generate_aqmd_results(matched_sample_county, 'high_county_collab')
    else:
        print("County-based PSM analysis failed")
    
    # Additional analysis: AQMD-specific results
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS: AQMD-Specific Results")
    print("="*60)
    
    # Focus on Southern California AQMDs
    socal_projects = project_stats[project_stats['SoCal_AQMD'] == 1]
    
    if len(socal_projects) > 0:
        print(f"\nSouthern California Analysis ({len(socal_projects)} projects):")
        
        for aqmd in socal_projects['Lead_AQMD'].unique():
            aqmd_projects = socal_projects[socal_projects['Lead_AQMD'] == aqmd]
            if len(aqmd_projects) >= 5:  # Only analyze AQMDs with sufficient projects
                avg_cost = aqmd_projects['cost_per_ton'].median()  # Use median for robustness
                avg_dac = aqmd_projects['share_DAC'].mean()
                cross_aqmd_rate = aqmd_projects['cross_aqmd_collab'].mean()
                
                print(f"\n{aqmd}:")
                print(f"  Projects: {len(aqmd_projects)}")
                print(f"  Median cost per ton: ${avg_cost:.2f}")
                print(f"  Avg DAC share: {avg_dac:.1%}")
                print(f"  Cross-AQMD collaboration rate: {cross_aqmd_rate:.1%}")
    
    # Save results for further analysis
    print(f"\nSaving results...")
    project_stats.to_csv('cci_aqmd_project_stats.csv', index=False)
    projects_with_aqmd.to_csv('cci_projects_with_aqmd.csv', index=False)
    
    print(f"\nAnalysis complete! Files saved:")
    print(f"  - cci_aqmd_project_stats.csv (project-level aggregated data)")
    print(f"  - cci_projects_with_aqmd.csv (full data with AQMD mappings)")
    
    return project_stats, projects_with_aqmd

if __name__ == "__main__":
    main()