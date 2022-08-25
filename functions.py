import pandas as pd
import s3fs
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.stats import theilslopes
import matplotlib.gridspec as gridspec
import matplotlib
import cartopy.crs as ccrs
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pymannkendall as mk
from scipy.stats import pearsonr
from scipy import stats
from global_land_mask import globe
import warnings
import os
import math
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn import linear_model

# coordinates fo rthe boxes for each region
# [lower_left_lon, lower_left_lat, 
# upper_right_lon, upper_right_lat]
dict_regions = dict([
    ('global', [-180, -90, 180, 90]),
    ('europe', [-20, 37, 60, 75]),
    ('russia', [60, 45, 180, 80]),
    ('nna', [-170, 45, -55, 78]),
    ('easia', [100, -10, 150, 45]),
    ('sna', [-130, 15, -60, 45]),
    ('wasia', [60, 0, 100, 45]),
    ('nsa', [-90, -35, -30, 15]),
    ('australia', [105, -45, 165, -10]),
    ('ssa', [-80, -60, -55, -35]),
    ('nafrica', [-25, -13, 60, 37]),
    ('safrica', [0, -40, 60, -13]),
])


dict_regions_sub = dict([
    ('global', [0]),
    ('europe', [1]),
    ('russia', [2]),
    ('nna', [3]),
    ('easia', [4]),
    ('sna', [5]),
    ('wasia', [6]),
    ('nsa', [7]),
    ('australia', [8]),
    ('ssa', [9]),
    ('nafrica', [10]),
    ('safrica', [11]),
])

def coordinates_regions(dict_regions):
    df = pd.DataFrame(columns = ['Lower left lon', 'Lower left lat',
                                'Upper right lon', 'Upper right lat'])
    for key in dict_regions:
        df.loc[key,:] = dict_regions[key]
    return df

# Global functions
def loc_region(location_f, dict_regions_f):
    
    try:
        lat_f = location_f[0]
        lon_f = location_f[1]
        for key_f in dict_regions:
            if key_f != 'global':
                if (lat_f>=dict_regions_f[key_f][1])&(lon_f>=dict_regions_f[key_f][0])&(
                    lat_f<=dict_regions_f[key_f][3])&(lon_f<=dict_regions_f[key_f][2]):
                    return key_f
                else:
                    pass
    except:
        lat_f = location_f[0][0]
        lon_f = location_f[0][1]
        for key_f in dict_regions:
            if key_f != 'global':
                if (lat_f>=dict_regions_f[key_f][1])&(lon_f>=dict_regions_f[key_f][0])&(
                    lat_f<=dict_regions_f[key_f][3])&(lon_f<=dict_regions_f[key_f][2]):
                    return key_f
                else:
                    pass
                
def average_std_area_weighted(ds):
    try:
        weights = np.cos(np.deg2rad(ds.lat))
        weights.name = "weights"
    except:
        weights = np.cos(np.deg2rad(ds.latitude))
        weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    try:
        return ds_weighted.mean(("longitude", "latitude"))
    except:
        return ds_weighted.mean(("lon", "lat"))
            
    
def process_station(df_f):
    # extract location
    location_f = df_f.loc[0, ['Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 
                          'Site_Elevation(m)']]
    # Get date from Date(dd:mm:yyyy)
    df_f['Date'] = pd.to_datetime(df_f['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
    # Set date as index
    df_f.set_index('Date', inplace=True)
    # extract AOD at a specific wavelength
    variables_f = ['AOD_500nm', 'AOD_551nm']
    df_aod_f = df_f.loc[:, variables_f]
    # replace fill_value by nan
    df_aod_f.replace(to_replace=-999.0, value=np.nan, inplace=True)
    return [location_f, df_aod_f]
            
# Aeronet processing and time series calculation
def process_aeronet(n_day_f, n_month_f, n_season_f, dict_regions_f):

    # applied to df.row to return the season in [1, 2, 3, 4]
    def season(month):
        seasons_f=[m%12 // 3 + 1 for m in range(1, 13)]
        return seasons_f[month-1]

    # returns a df containing all yearly averages of AOD fulfulling the requirements
    # does the average day -> month -> season -> year
    def average_dmy(df_f, d_thresh_f, m_thresh_f, s_thresh_f):
        # if number of day with available data >= n_day, calculate the mean
        # else replace the mean by a nan value
        df_m_f = df_f.groupby(pd.Grouper(freq='M')).mean()[
            [df_f.groupby(pd.Grouper(freq='M')).size()>=d_thresh_f][0].tolist()].groupby(
            pd.Grouper(freq='M')).mean()
        # if after condition, more than one month is there
        if df_m_f.shape[0]>0:
            # add a column with the date to apply the function season
            df_m_f['Date'] = df_m_f.index
            df_m_f['season'] = df_m_f.apply(lambda row : season(row.Date.month), axis=1)

            # create df_s which contains the mean per season and per year if 
            # number of month with available data >= n_month
            df_s_f = df_m_f.groupby([df_m_f.Date.dt.year, df_m_f['season']]).mean()[
                df_m_f.groupby([df_m_f.Date.dt.year, df_m_f['season']]).size()>=m_thresh_f]

            # Calculate the yearly average only if the number of seasons that have data is >= n_season
            # df_y contains the yearly average for years satisfying the conditions
            df_y_f = df_s_f.reset_index(level=['season'])
            df_y_f = df_y_f.groupby(df_y_f.index).mean()[df_y_f.groupby(df_y_f.index).size()>=s_thresh_f]
            df_y_f = df_y_f.drop(columns=['season', 'AOD_551nm']).dropna(axis=0)
            # if after condition, at least one year is available, return the yearly time series
            if df_y_f.shape[0]>0:
                return df_y_f
            else:
                return [[], []]
        else:
            return [[], []]

    # returns [p-value, slope, intercept] using the mann-kendall trend test
    def trend_MK(df_y_f):
        try:
            results_f = mk.original_test(df_y_f.loc[df_y_f.index>=2000, 'AOD_500nm'])
            return [results_f[2], results_f[7], results_f[8]]
        except:
            return [np.nan, np.nan, np.nan]
    
    # Connect to bucket (anonymous login for public data only)
    fs = s3fs.S3FileSystem(anon=True,
          client_kwargs={
             'endpoint_url': 'https://climate.uiogeo-apps.sigma2.no/'
          })
    s3path = 'ESGF/obs4MIPs/AERONET/AeronetSunV3Lev1.5.daily/*.lev30'
    remote_files = fs.glob(s3path)

    # Iterate through remote_files to create a fileset
    fileset_f = [fs.open(file) for file in remote_files]
    
    # dataframe which contains the coordinates and AOD trend for each station that fulfill the requirements
    # this dataframe also contains the region of each station
    df_trend_f = pd.DataFrame(columns = ['lontitude', 'latitude', 'altitude', 'slope', 
                                       'p_val', 'intercept', 'nb_yrs', 'start_yr', 'region'])
    
    # count increments in case all conditions are validated
    count=0
    # Store all the time series fulfilling the requirements
    list_ts_f = []
        
    # loop in the fileset to test the conditions on the stations and compute/store
    # the trends if fulfilled
    for ifile, file in enumerate(fileset_f):
        # load in df the station data
        df_f = pd.read_csv(file, skiprows=6)
        # process the data to extract the AOD and location of the station
        location_f, df_aod_f = process_station(df_f)
        # if the station has at least one measurement of AOD, do:
        if df_aod_f[~df_aod_f['AOD_500nm'].isna()].shape[0] >= 1:
            # condition and compute the yearly average
            df_y_f = average_dmy(df_aod_f, n_day_f, n_month_f, n_season_f)
            # if no yearly average has been returned due to no condition fulfilled, average_dmy returns an empty list
            # in that case, the station is not added to the list containing a time series per valid station
            if type(df_y_f)==list:
                # skip the stations that don't fullfil the requirements
                pass
            else:
                # Store the time series in list_ts and coordinates
                list_ts_f.append([df_y_f, [location_f]])
                # Compute the trend per station if at least 6 years of measurements are available
                if df_y_f.shape[0] >= 3:
                    try:
                        p_value_f, slope_f, intercept_f = trend_MK(df_y_f)
                        # store the station and trend for comparison with MODIS and CMIP6 models
                        df_trend_f.loc[count,:] = [location_f[1], location_f[0], location_f[2], slope_f, 
                                           p_value_f, intercept_f, df_y_f.shape[0], df_y_f.index[0], 
                                           loc_region(location_f, dict_regions_f)]
                        count+=1
                    except:
                        pass
        else:
            pass
        
    return list_ts_f, df_trend_f

def figure_aeronet_location(list_ts):
    fig=plt.figure(figsize=(9,4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    for iloc, loc in enumerate(list_ts):
        ny_lon = loc[1][0][1]
        ny_lat = loc[1][0][0]
        mappable_cbar = ax.scatter(
            ny_lon, ny_lat, s=10, c='k', alpha=0.6)
    ax.set_title('AERONET stations locations')

# MODIS functions
def read_MODIS_AOD_xr():
    fs = s3fs.S3FileSystem(anon=True,
          client_kwargs={
             'endpoint_url': 'https://climate.uiogeo-apps.sigma2.no/'
          })
    fs.ls('ESGF/obs4MIPs/MODIS/MODIS6.1terra')[:10]
    s3path = 'ESGF/obs4MIPs/MODIS/MODIS6.1terra/*od550aer*.nc'
    remote_files = fs.glob(s3path)
    # Iterate through remote_files to create a fileset
    fileset = [fs.open(file) for file in remote_files]
    # Read file with xarray
    # dataset = xr.open_mfdataset(fileset, combine='by_coords', chunks=None) # This method returns dask arrays ! BAD
    list_ds = []
    for ifile, file in enumerate(fileset):
        list_ds.append(xr.open_dataset(fileset[ifile]))
    return xr.concat(list_ds, dim='time')

def yearly_mean_MODIS(data):
    # 6 months offset to have a tick at the middle of the year
    return data.resample({'time': 'Y'}, loffset = '-6M').mean()

# def mask_ocean(dataset, lon, lat):
#     # Make a grid
#     lon_grid, lat_grid = np.meshgrid(lon,lat)
#     # Get whether the points are on land using globe.is_land
#     mask = globe.is_land(lat_grid, lon_grid)
#     return dataset.where(mask, np.nan)

def figure_mean_modis_aod(ds_modis):
    fig = plt.figure(num=None, figsize=(9, 4))
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
    ds_modis['od550aer'].mean(dim='time').plot(ax = ax, cmap='Reds',
                                                  cbar_kwargs={'label': r'AOD'}, vmin=0, vmax=1)
    ax.coastlines()
    ax.set_title('Mean MODIS AOD (550nm) 2000-2018')

def mask_ocean(dataset, lon, lat):
    # Make a grid
    lon_grid, lat_grid = np.meshgrid(lon,lat)
    # Get whether the points are on land using globe.is_land
    mask = globe.is_land(lat_grid, lon_grid)
    try:
        mask_3D = np.zeros((dataset.shape))*np.nan
        
    except:
        mask_3D = np.zeros((dataset['od550aer'].shape))*np.nan
        for itime in range(0, mask_3D.shape[0]):
            mask_3D[itime, :, :] = mask
        return dataset.where(mask_3D, np.nan)
    
    for itime in range(0, mask_3D.shape[0]):
        mask_3D[itime, :, :] = mask
    return dataset.where(mask_3D, np.nan)

def nearest_coord(latorlon, latitudeorlatitude):
    return latitudeorlatitude[np.argmin(abs(latitudeorlatitude-latorlon))]

def average_std_ts_modis_region(data_modis_y_land, region_coord):
    longitude = data_modis_y_land.longitude.values
    latitude = data_modis_y_land.latitude.values
    slice_ds= data_modis_y_land.sel(latitude=slice(nearest_coord(region_coord[3], latitude), 
                                   nearest_coord(region_coord[1], latitude)), 
                                   longitude=slice(nearest_coord(region_coord[0], longitude), 
                                   nearest_coord(region_coord[2], longitude)))
    return [average_std_area_weighted(slice_ds), slice_ds.std(dim=['longitude', 'latitude'])]

def trend_meants_stdts_mk_modis_region(dict_regions, data_modis_y_land):
    list_ts_modis_region_mean = [[] for i in range(0, len(dict_regions))]
    list_ts_modis_region_std = [[] for i in range(0, len(dict_regions))]
    list_trend_modis_region = a = [[[],[],[]] for i in range(0, len(dict_regions))]
    for ind_region, key in enumerate(dict_regions.keys()):
        list_ts_modis_region_mean[ind_region], list_ts_modis_region_std[ind_region] = average_std_ts_modis_region(
            data_modis_y_land, dict_regions[key])
        try:
            results = mk.original_test(list_ts_modis_region_mean[ind_region]['od550aer'])
            list_trend_modis_region[ind_region] = [results[7], results[8], results[2]] # slope, intercept, pval
        except:
            pass
    return list_trend_modis_region, list_ts_modis_region_mean, list_ts_modis_region_std

def add_trend_MK_MODIS(data_y_land):
    data_aod = data_y_land['od550aer']
    trend = np.zeros((data_aod.latitude.shape[0], data_aod.longitude.shape[0]))*np.nan
    p_val = np.zeros((data_aod.latitude.shape[0], data_aod.longitude.shape[0]))*np.nan
    intercept = np.zeros((data_aod.latitude.shape[0], data_aod.longitude.shape[0]))*np.nan
    for ilat, lat in enumerate(data_aod.latitude):
        for ilon, lon in enumerate(data_aod.longitude):
            ts = data_aod.isel(latitude=ilat, longitude=ilon)
            try:
                results = mk.original_test(ts)
                trend[ilat, ilon] = results[7]
                intercept[ilat, ilon] = results[8]
                p_val[ilat, ilon] = results[2]
            except:
                pass
    data_y_land['trend_mk'] = (['latitude', 'longitude'],  trend)
    data_y_land['p_val_mk'] = (['latitude', 'longitude'],  p_val)
    data_y_land['intercept_mk'] = (['latitude', 'longitude'],  intercept)
    return data_y_land

def table_modis_trend_region(list_trend_modis_region, dict_regions):
    df_trend_region = pd.DataFrame(columns=['region', 'slope (/yr)', 'p_val', 
                                   'intercept', 'relative_slope (%/yr)'])
    for ikey, key in enumerate(dict_regions.keys()):
        df_trend_region.loc[ikey, :] = [key, list_trend_modis_region[ikey][0], 
            list_trend_modis_region[ikey][2], list_trend_modis_region[ikey][1], 
            list_trend_modis_region[ikey][0]/list_trend_modis_region[ikey][1]*100]
    return df_trend_region

def table_number_stations(df_trend, dict_regions):
    df_nstation_region = pd.DataFrame(columns=['Region', 'Number of stations'])
    for ikey, key in enumerate(dict_regions.keys()):
        df_nstation_region.loc[ikey, :] = [key, df_trend[df_trend.region==key].shape[0]]
    df_nstation_region.loc[0, 'Number of stations'] = df_nstation_region.loc[
        :, 'Number of stations'].sum()
    return df_nstation_region    

# def month_to_season(x):
#     for ii, i in enumerate(x):
#         if i in [12,1,2]:
#             x[ii] = 1
#         elif i in [3,4,5]:
#             x[ii] = 2
#         elif i in [6,7,8]:
#             x[ii] = 3
#         else:
#             x[ii] = 4
#     return x
# a = a.assign(season=lambda a: test(a.month))
# a.groupby(a.season)

######################### Takes to long to process with xarray but works fine!
# Function for calculating the yearly mean with conditions based on thresholds
def average_dmy_modis(ts, d_thresh, m_thresh, s_thresh):
    ts_m = ts.resample(time = '1M').count().where(ts.resample(time = '1M').count()>=d_thresh)
    ts_s = ts_m.resample(time = 'Q').mean().where(ts_m.resample(time = 'Q').count()>=m_thresh)
    ts_y = ts_s.resample(time = 'Y').mean().where(ts_s.resample(time = 'Y').count()>=s_thresh)
    return ts_y

def save_ts_gridcell(data, d_thresh, m_thresh, s_thresh):
    for ilat, lat in enumerate(data.latitude):
        for ilon, lon in enumerate(data.longitude):
            ts = data.sel(latitude=lat, longitude=lon)
            average_dmy_modis(ts, d_thresh, m_thresh, s_thresh).to_netcdf(
                path='yearlyTS_modis_Lon'+str(ilon)+'xLat'+str(ilat)+'.nc')


# version = 1: with modis trend in the regional subplots
# version = 2: only with AERONET regional trends
def figure1_regional_AOD_trends(
    list_ts, ds_modis_y_land, list_trend_modis, list_ts_modis, 
    list_std_modis, version, dict_regions, dict_regions_sub):
    
    ## Defining the figure and axes frame
    fig = plt.figure(figsize=(25,14))
    gs = gridspec.GridSpec(ncols=5, nrows=5, hspace = 0.6, wspace=0.3)

    # Central map
    ax_map = fig.add_subplot(gs[1:4,1:4], projection=ccrs.PlateCarree())
    ax_map.coastlines()
    ax_map.stock_img()


    # Colorbar for the map
    cm = plt.cm.get_cmap('seismic')
    # cm.set_bad(color='black',alpha = 1.)

    # # Defining the subplots for the trends by region
    # sub_array contains 1 when the number of subplot in the frame
    # has to be displayed, and 0 everywhere else
    sub_array = np.ones((5,5))
    sub_array[1:4, 1:4] = np.zeros((3,3))
    sub_array[0, 0] = 0
    sub_array[4, 4] = 0
    sub_array[4, 0] = 0
    sub_array[0, 4] = 0

    # axes_sub to store the axes for the trends
    axes_sub = []
    
    # style of the axes
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
               'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']

    count=0
    for sub_x in range(0,5):
        for sub_y in range(0,5):
            if sub_array[sub_x, sub_y] == 1:
                # create and append the trend subplots
                axes_sub.append(fig.add_subplot(gs[sub_x,sub_y]))
                # apply a function to make the subplots look nice
                thickax(axes_sub[count])
                if list_regions[count]=='Global':
                    axes_sub[count].set_title(list_regions[count], fontweight='bold', c='k', fontsize=16, pad=8)
                else:
                    axes_sub[count].set_title(list_regions[count], c='k', fontsize=16, pad=8)
                count+=1
    
    # Method to plot the lines that connect the subplots with the regions on the central map
    transFigure = fig.transFigure.inverted()
    def line_connect(ax, c1, c2):
        coord1 = transFigure.transform(ax.transData.transform(c1))
        coord2 = transFigure.transform(ax_map.transData.transform(c2))
        line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                       transform=fig.transFigure, linewidth=2, linestyle='--', c='k')
        fig.add_artist(line)

    line_connect(axes_sub[0], [0.2,0], [-147, 85]) # Global
    line_connect(axes_sub[1], [0.6,0], [0, 60]) # Europe
    line_connect(axes_sub[2], [0.8,0], [150, 60]) # Russia
    line_connect(axes_sub[5], [1,0.3], [-120, 25]) # S- NA
    line_connect(axes_sub[4], [0,0.3], [130, 25]) # Japan
    line_connect(axes_sub[3], [1,0.6], [-140, 60]) # N- NA
    line_connect(axes_sub[6], [0,0.3], [80, 12]) # India
    line_connect(axes_sub[8], [0,0.6], [130, -25]) # Australia
    line_connect(axes_sub[11], [0.5,1.27], [30, -35]) # South-Africa
    line_connect(axes_sub[10], [0.37,1.27], [-12, 25]) # N-Africa
    line_connect(axes_sub[9], [0.5,1.27], [-75, -55]) # S- SA
    line_connect(axes_sub[7], [1,0.5], [-70, 0]) # N- SA

    list_key_regions =  ['global','europe','russia','nna','easia','sna',
                         'wasia','nsa','australia','ssa','nafrica','safrica']

    def line_regions(ll_coord, ur_coord):
        c = 'k'
        lw = 2
        coordll = ll_coord
        coordur = ur_coord
        coordlr = [ur_coord[0], ll_coord[1]]
        coordul = [ll_coord[0], ur_coord[1]]
        line = matplotlib.lines.Line2D((coordll[0],coordul[0]),(coordll[1],coordul[1])
                                       , linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordll[0],coordlr[0]),(coordll[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordul[0]),(coordur[1],coordul[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordlr[0]),(coordur[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)

    for key_region in dict_regions:
        line_regions(dict_regions[key_region][0:2], dict_regions[key_region][2:4])

    # Plot the central map
    vmin = -9
    vmax = 9
    lon = np.arange(-179.5, 180.5, 1)
    lat = np.arange(-89.5, 90.5, 1)
    
    # calculate the relative trend of modis per grid cell over land
    ds_trend_modis = ds_modis_y_land['trend_mk']/ds_modis_y_land['intercept_mk']*100
    
    # set color for the nan values (white)
    null_bkgd_ds_trend_modis = ds_trend_modis/ds_trend_modis*4
    null_bkgd_ds_trend_modis = null_bkgd_ds_trend_modis.fillna(3.55)
    null_bkgd_ds_trend_modis.plot(ax = ax_map, cmap = 'gray', add_colorbar=False, vmin=0, vmax=4)
    
    # plot the map in the central axe
    cmap = plt.cm.RdBu.reversed()
    mappable_cbar = ds_trend_modis.plot(ax = ax_map, cmap=cmap, add_colorbar=False, 
                                       levels = np.arange(-8, 9, 1))
    ax_map.set_title('  MODIS AOD (550nm)     Trend for 2000-2018', fontsize=18)
    
    # add dots where the p-value is significant
    trend_modis_pval = ds_modis_y_land.where(
        ds_modis_y_land['p_val_mk'] < 0.05, np.nan)['p_val_mk'].values
    trend_modis_pval[~np.isnan(trend_modis_pval)] = 1
    
    # put one dot every 2 dots
    mask = np.ones(trend_modis_pval.shape)*np.nan
    for i in range(0, len(lon), 3):
        for j in range(0, len(lat), 3):
            mask[j, i] = 1.5
    trend_modis_pval = trend_modis_pval * mask
    x, y = np.meshgrid(lon, lat)
    ax_map.scatter(x, y, s=np.flip(trend_modis_pval, axis=0), transform=ccrs.PlateCarree(), c='k', alpha=1)

    def plot_ts(region, df_y, axes_sub):
        try:
            if region=='global':
                ts = df_y[df_y.index>=2000]
                y = ts['AOD_500nm']
                x = ts.index.values
                axes_sub[0].scatter(x, y, marker='d', alpha=0.15, c='k',s=10)
            else:
                ts = df_y[df_y.index>=2000]
                y = ts['AOD_500nm']
                x = ts.index.values
                axes_sub[dict_regions_sub[region][0]].scatter(x, y, 
                                                              marker='d', alpha=0.15, c='k',s=10)
        except:
            pass 
    
    # calculate yearly average within the regions
    list_ts_region = [[] for i in range(0, len(dict_regions))]
    list_ts_std_region = [[] for i in range(0, len(dict_regions))] 
    for ts, location in list_ts:
        if version == 2:
            plot_ts(loc_region(location, dict_regions), ts, axes_sub)
            plot_ts('global', ts, axes_sub)
        try:
            list_ts_region[dict_regions_sub[loc_region(location, dict_regions)][0]].append(ts)
            list_ts_region[0].append(ts)
        except:
            pass
        
    def plot_regional_yearly_ts(ts, ind, color):
        axes_sub[ind].plot(ts.index.values, ts['AOD_mean'], 'o', c=color, linewidth=2.7, 
                           markersize=6, alpha=0.65)
        axes_sub[ind].fill_between(ts.index.values, ts['AOD_mean']-ts['AOD_std'], 
                                   ts['AOD_mean']+ts['AOD_std'], 
                                   color=color, alpha=0.2)
        
    def trend_MK_region(df_region):
        try:
            results = mk.original_test(df_region['AOD_mean'])
            return [results[2], results[7], results[8]]
        except:
            return [np.nan, np.nan, np.nan] 
        
    def plot_modis_region(list_trend_modis, list_ts_modis, list_std_modis, color, ind):
        axes_sub[ind].plot(np.arange(2000, 2019), list_ts_modis[ind]['od550aer'], 'o', c=color, linewidth=2.7, 
                       markersize=6, alpha=0.3)
        axes_sub[ind].fill_between(np.arange(2000, 2019), 
                                   list_ts_modis[ind]['od550aer']-list_std_modis[ind]['od550aer'], 
                                   list_ts_modis[ind]['od550aer']+list_std_modis[ind]['od550aer'], 
                                   color=color, alpha=0.2)
        axes_sub[ind].plot(np.arange(2000, 2019), np.arange(0, 19)*list_trend_modis[ind][0]+list_trend_modis[ind][1], 
                           '--', linewidth=2.7, c=color)
        axes_sub[ind].text(2001, 0.76, 'MODIS=', c=color, fontsize=12, fontweight='bold')
        axes_sub[ind].text(2007.1, 0.76,
                           f'{list_trend_modis[ind][0]/list_trend_modis[ind][1]*100:.2f}'+
                           '%/yr ; p_val='+f'{list_trend_modis[ind][2]:.2f}', 
                                  c='k', fontsize=12)
        
    def plot_trend(slope, intercept, ind, color):
        axes_sub[ind].plot(np.arange(2000, 2021), np.arange(0, 21)*slope+intercept, '--', linewidth=2.7, c=color)
        
    def regional_yearly_ts(list_ts_region):
        list_years = [[] for i in range(0,21)]
        for ts in list_ts_region:
            for ind_year, year in enumerate(np.arange(2000, 2021)):
                aod = ts[ts.index==year]['AOD_500nm'].values
                if aod.size==0:
                    pass
                else:
                    list_years[ind_year].append(aod[0])

        df_region = pd.DataFrame(data = np.zeros((21,4))*np.nan, 
                                 columns = ['Date', 'AOD_mean', 'AOD_std', 'nb_station'])
        df_region['Date'] = np.arange(2000, 2021)
        for ind_year, year in enumerate(np.arange(2000, 2021)):
            df_region.loc[ind_year, 'AOD_mean'] = np.nanmean(list_years[ind_year])
            df_region.loc[ind_year, 'AOD_std'] = np.nanstd(list_years[ind_year])
            df_region.loc[ind_year, 'nb_station'] = len(list_years[ind_year])
        df_region.set_index('Date', inplace=True)
        return df_region
        
    color1 = 'blue'
    color2 = 'darkblue'
    list_yearly_ts_region = [[] for i in range(0, len(dict_regions))]
    for ind_region in range(0,len(list_yearly_ts_region)):
        list_yearly_ts_region[ind_region] = regional_yearly_ts(list_ts_region[ind_region])
        plot_regional_yearly_ts(list_yearly_ts_region[ind_region], ind_region, 'cornflowerblue')
        p_value, slope, intercept = trend_MK_region(list_yearly_ts_region[ind_region])
        if version == 1:
            axes_sub[ind_region].text(2000.5, 0.90, 'AERONET=', c='cornflowerblue', fontsize=12, fontweight='bold')
            axes_sub[ind_region].text(2007.1, 0.90, 
                                      f'{slope/intercept*100:.2f}'+'%/yr ; p_val='+f'{p_value:.2f}', 
                                      c='k', fontsize=12)
            plot_modis_region(list_trend_modis, list_ts_modis, list_std_modis, 'darkorange', ind_region)
            plot_trend(slope, intercept, ind_region, 'cornflowerblue')
            
        if version == 2:
            axes_sub[ind_region].text(2000.5, 0.88, 'Trend=', c='k', fontsize=12)
            if slope<0:    
                axes_sub[ind_region].text(2004.5, 0.88, f'{slope/intercept*100:.2f}'+'%/yr', c='blue', fontsize=13)
            else:
                axes_sub[ind_region].text(2004.5, 0.88, f'{slope/intercept*100:.2f}'+'%/yr', c='red', fontsize=13)                      
            axes_sub[ind_region].text(2000.5, 0.74, 'p_val=' + 
                                      f'{p_value:.2f}', c='k', fontsize=13)
#              + '; n_stations=' + f'{len(list_ts_region[ind_region]):.0f}' # number of stations
            plot_trend(slope, intercept, ind_region, color2)
            
    # Parameters of the axes
    for i in range(0, 12):
        axes_sub[i].set_yticks([0,0.25,0.5, 0.75])
        axes_sub[i].set_xticks([2000, 2010, 2020])
        axes_sub[i].set_xlim([2000, 2020])
        if version==1:
            axes_sub[i].set_ylim([0, 0.97])
            if i in [0, 3, 5, 7, 9]:
                axes_sub[i].set_ylabel('AOD', fontsize=14)
        if version==2:
            axes_sub[i].set_ylim([0, 1])
            if i in [0, 3, 5, 7, 9]:
                axes_sub[i].set_ylabel('AOD (500nm)', fontsize=13.5)

    # Parameters of the colorbar
    axins = inset_axes(ax_map,
                        width="3.8%",  
                        height="100%",
                        loc='center right',
                        borderpad=-39
                       )
    cb = fig.colorbar(mappable_cbar, cax=axins, orientation="vertical")
    cb.set_label(r'Relative AOD trend %/yr', labelpad=14, size=16, rotation=90)
    cb.ax.tick_params(labelsize=14)

    # Save the figure
    if version == 1:
        fig.savefig('figure1_aeronet_vs_modis_aodtrend_regions.jpeg', bbox_inches='tight', dpi=300)
    elif version == 2:
        fig.savefig('figure1_aeronet_aodtrend_regions.jpeg', bbox_inches='tight', dpi=300)

        
##################### SEASONALITY
    
# applied to df.row to return the season in [1, 2, 3, 4]
def season(month):
    seasons=[m%12 // 3 + 1 for m in range(1, 13)]
    return seasons[month-1]
    
# returns a df containing all yearly averages of AOD fulfulling the requirements
# does the average day -> month -> season -> year
def average_dmy_season(df, d_thresh, m_thresh):
    # if number of day with available data >= d_thresh, calculate the mean
    # else replace the mean by a nan value
    df_m = df.groupby(pd.Grouper(freq='M')).mean()[[df.groupby(
        pd.Grouper(freq='M')).size()>=d_thresh][0].tolist()].groupby(pd.Grouper(freq='M')).mean()
    # if after condition, more than one month is there
    if df_m.shape[0]>0:
        # add a column with the date to apply the function season
        df_m['Date'] = df_m.index
        df_m['season'] = df_m.apply(lambda row : season(row.Date.month), axis=1)

        # create df_s which contains the mean per season and per year if 
        # number of month with available data >= m_thresh
        df_s = df_m.groupby([df_m.Date.dt.year, df_m['season']]).mean()[
            df_m.groupby([df_m.Date.dt.year, df_m['season']]).size()>=m_thresh]
        df_s.reset_index(level=['season'], inplace=True)
        
        list_df_y_season = [[] for i in range(0,4)]
        for ind_season in range(1,5):
            df_y = df_s.loc[df_s['season']==ind_season,:]
            df_y = df_y.drop(columns=['season', 'AOD_551nm']).dropna(axis=0)
            list_df_y_season[ind_season-1] = df_y
        
        return list_df_y_season
        
    else:
        return [[], [], [], []]
    
def process_aeronet_season(n_day, n_month, dict_regions):
    
# Connect to bucket (anonymous login for public data only)
    fs = s3fs.S3FileSystem(anon=True,
          client_kwargs={
             'endpoint_url': 'https://climate.uiogeo-apps.sigma2.no/'
          })
    s3path = 'ESGF/obs4MIPs/AERONET/AeronetSunV3Lev1.5.daily/*.lev30'
    remote_files = fs.glob(s3path)

    # Iterate through remote_files to create a fileset
    fileset = [fs.open(file) for file in remote_files]
    
    # Store all the time series fulfilling the requirements for each season
    list_ts_djf = []
    list_ts_mam = []
    list_ts_jja = []
    list_ts_son = []
        
    # loop in the fileset to test the conditions on the stations and compute/store
    # the trends if fulfilled
    for ifile, file in enumerate(fileset):
        # load in df the station data
        df = pd.read_csv(file, skiprows=6)
        # process the data to extract the AOD and location of the station
        location, df_aod = process_station(df)
        # if the station has at least one measurement of AOD, do:
        if df_aod[~df_aod['AOD_500nm'].isna()].shape[0] >= 1:
            # condition and compute the yearly average
            df_y_djf, df_y_mam, df_y_jja, df_y_son = average_dmy_season(df_aod, n_day, n_month)
            # if no yearly average has been returned due to no condition fulfilled, average_dmy returns an empty list
            if type(df_y_djf)==list:
                # skip the stations that don't fullfil the requirements
                pass
            else:
                list_ts_djf.append([df_y_djf, [location]])
            if type(df_y_mam)==list:
                pass
            else:            
                list_ts_mam.append([df_y_mam, [location]])
            if type(df_y_jja)==list:
                pass
            else:
                list_ts_jja.append([df_y_jja, [location]])
            if type(df_y_son)==list:
                pass
            else:            
                list_ts_son.append([df_y_son, [location]])
        else:
            pass
    return [list_ts_djf, list_ts_mam, list_ts_jja, list_ts_son]
    
# Figure2:
def figure2_regional_seasonal_AOD_trends(
    list_ts_djf, list_ts_mam, list_ts_jja, list_ts_son, 
    dict_regions, dict_regions_sub):
    
    # style of the axes
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig = plt.figure(figsize=(18,11))

    # axes_sub to store the axes for the trends
    axes_sub = []
    # Full names of the regions
    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
                   'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']
    count=0
    for sub_x in range(0,4):
        for sub_y in range(0,3):
            # create and append the trend subplots
            axes_sub.append(plt.subplot(3,4,count+1))
            # apply a function to make the subplots look nice
            thickax(axes_sub[count])
            # title
            if list_regions[count]=='Global':
                axes_sub[count].set_title(list_regions[count], c='blue', fontsize=15, pad=8)
            else:
                axes_sub[count].set_title(list_regions[count], c='k', fontsize=15, pad=8)
            count+=1

    def calculate_season_yearly_avg(list_ts, dict_regions_sub, dict_regions):
        list_ts_region = [[] for i in range(0, len(dict_regions))] 
        for ts, location in list_ts:
            try:
                list_ts_region[dict_regions_sub[loc_region(location, dict_regions)][0]].append(ts)
                list_ts_region[0].append(ts)
            except:
                pass
        return list_ts_region
    
    list_ts_region_djf = calculate_season_yearly_avg(list_ts_djf, dict_regions_sub, dict_regions)
    list_ts_region_mam = calculate_season_yearly_avg(list_ts_mam, dict_regions_sub, dict_regions)
    list_ts_region_jja = calculate_season_yearly_avg(list_ts_jja, dict_regions_sub, dict_regions)
    list_ts_region_son = calculate_season_yearly_avg(list_ts_son, dict_regions_sub, dict_regions)

    def regional_yearly_ts(list_ts_region):
        list_years = [[] for i in range(0,21)]
        for ts in list_ts_region:
            for ind_year, year in enumerate(np.arange(2000, 2021)):
                aod = ts[ts.index==year]['AOD_500nm'].values
                if aod.size==0:
                    pass
                else:
                    list_years[ind_year].append(aod[0])

        df_region = pd.DataFrame(data = np.zeros((21,4))*np.nan, 
                                 columns = ['Date', 'AOD_mean', 'AOD_std', 'nb_station'])
        df_region['Date'] = np.arange(2000, 2021)
        for ind_year, year in enumerate(np.arange(2000, 2021)):
            df_region.loc[ind_year, 'AOD_mean'] = np.nanmean(list_years[ind_year])
            df_region.loc[ind_year, 'AOD_std'] = np.nanstd(list_years[ind_year])
            df_region.loc[ind_year, 'nb_station'] = len(list_years[ind_year])
        df_region.set_index('Date', inplace=True)
        
        return df_region
                    
    def plot_regional_yearly_ts(ts, ind, color):
        axes_sub[ind].plot(ts.index.values, ts['AOD_mean'], 'o', c=color, linewidth=2.7, 
                           markersize=8, alpha=0.55)
#         axes_sub[ind].fill_between(ts.index.values, ts['AOD_mean']-ts['AOD_std'], 
#                                    ts['AOD_mean']+ts['AOD_std'], 
#                                    color=color, alpha=0.2)
        
    def regional_yearly_ts(list_ts_region):
        list_years = [[] for i in range(0,21)]
        for ts in list_ts_region:
            for ind_year, year in enumerate(np.arange(2000, 2021)):
                aod = ts[ts.index==year]['AOD_500nm'].values
                if aod.size==0:
                    pass
                else:
                    list_years[ind_year].append(aod[0])

        df_region = pd.DataFrame(data = np.zeros((21,4))*np.nan, 
                                 columns = ['Date', 'AOD_mean', 'AOD_std', 'nb_station'])
        df_region['Date'] = np.arange(2000, 2021)
        for ind_year, year in enumerate(np.arange(2000, 2021)):
            df_region.loc[ind_year, 'AOD_mean'] = np.nanmean(list_years[ind_year])
            df_region.loc[ind_year, 'AOD_std'] = np.nanstd(list_years[ind_year])
            df_region.loc[ind_year, 'nb_station'] = len(list_years[ind_year])
        df_region.set_index('Date', inplace=True)
        return df_region
        
    def trend_MK_region(df_region):
        try:
            results = mk.original_test(df_region['AOD_mean'])
            return [results[2], results[7], results[8]]
        except:
            return [np.nan, np.nan, np.nan]
        
    def plot_trend(slope, intercept, ind, color):
        axes_sub[ind].plot(np.arange(2000, 2021), np.arange(0, 21)*slope+intercept, '-', linewidth=3, c=color)

    def calculate_season_yearly_avg_region(list_ts_region, color, color_trend, name_season, locator):
        list_yearly_ts_region = [[] for i in range(0, len(dict_regions))]
        for ind_region in range(0,12):
            list_yearly_ts_region[ind_region] = regional_yearly_ts(list_ts_region[ind_region])
#             plot_regional_yearly_ts(list_yearly_ts_region[ind_region], ind_region, color)
            p_value, slope, intercept = trend_MK_region(list_yearly_ts_region[ind_region])
            axes_sub[ind_region].text(2000.5, locator, name_season+'=', c=color_trend, fontsize=12)
            axes_sub[ind_region].text(2004, locator, 
                                      f'{slope/intercept*100:.2f}'+'%/yr ; p_val='+f'{p_value:.2f}', 
                                      c='k', fontsize=13)
            plot_trend(slope, intercept, ind_region, color_trend)
            
    calculate_season_yearly_avg_region(list_ts_region_djf, 'cornflowerblue', 'cornflowerblue', 'DJF', 1)
    calculate_season_yearly_avg_region(list_ts_region_mam, 'coral', 'coral', 'MAM', 0.88)
    calculate_season_yearly_avg_region(list_ts_region_jja, 'orange', 'orange', 'JJA', 0.76)
    calculate_season_yearly_avg_region(list_ts_region_son, 'mediumturquoise', 'mediumturquoise', 'SON', 0.64)

    # Parameters of the axes
    for i in range(0, 12):
        axes_sub[i].set_yticks([0,0.5,1])
        axes_sub[i].set_xticks([2000, 2010, 2020])
        axes_sub[i].set_xlim([2000, 2020])
        axes_sub[i].set_ylim([0, 1.1])
        if i in [0, 4, 8]:
            axes_sub[i].set_ylabel('AOD (500nm)', fontsize=14)
        else:
            axes_sub[i].set_yticklabels(['','',''])
        if i in [11, 10, 9, 8]:
            pass
        else:
            axes_sub[i].set_xticklabels(['','',''])

    plt.subplots_adjust(wspace=0.24, hspace=0.45)
    
    
    
##################### Burned Area

# Convert fraction of grid cell burnt to surface area burnt
def grid_area(longitude, latitude):
    #This creat a global grid of the approximate size of each grid cell
    def gridsize(lat1):
        lon1=200
        lat2=lat1
        lon2=lon1+1

        R = 6378.137 # // Radius of earth in km
        dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
        dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
        a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(
            lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = R * c * 100 # in meters

        return d * 1000 #; // meters

    boxlo,boxla=np.array(np.meshgrid(longitude,latitude))
    sizes=np.ones(boxlo.shape)
    grid=gridsize(boxla)

    grid_area=xr.DataArray(grid,coords={'lat':boxla[:,1],'lon':boxlo[1,:]},dims=['lat','lon'])
    lat_size=110567 #in m
    grid_area['m2']=grid_area*lat_size

    return grid_area['m2']

def process_ba():

    # Load download data and select data after 2000
    path_file = './GFED4_Glb_0.25x0.25_fire_BA__monthly.nc'
    ds = xr.open_dataset(path_file)
    ds_ba = ds['burned_area']
    ds_ba = ds_ba[ds_ba['time.year']>=2000,:,:]
    
    # Interpolate onto a 1x1grid
    longitude = np.arange(-179.5, 180.5, 1)
    latitude = np.arange(-89.5, 90.5, 1)
    ds_ba = ds_ba.interp(lon=longitude, lat=latitude, method="linear")
    
    ds_ba['burned_area_m2'] = grid_area(longitude, latitude)*ds_ba
    ds_ba['burned_area_m2'].assign_attrs({'units': '$m^2$/grid_cell'})
    
    ds_ba['burned_area_land'] = mask_ocean(ds_ba, ds_ba.lon, ds_ba.lat)
    
    return ds_ba

def figure_mean_gfed4_ba(ds_ba):
    fig = plt.figure(num=None, figsize=(9, 4))
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
    ds_ba['burned_area_m2'].mean(dim='time').plot(ax = ax, cmap='Reds',
                                                  cbar_kwargs={'label': r'Burned area (m$^2/month$)'})
    ax.coastlines()
    ax.set_title('Mean GFED4 burned area 2000-2015')
    
def surface_area_region(region_coord, longitude, latitude):
    ds_grid_area = grid_area(longitude, latitude)
    lon_grid, lat_grid = np.meshgrid(longitude,latitude)
    mask = globe.is_land(lat_grid, lon_grid)
    ds_grid_area = ds_grid_area.where(mask, 0)
    slice_region = ds_grid_area.sel(lat=slice(nearest_coord(region_coord[1], latitude), 
                                       nearest_coord(region_coord[3], latitude)), 
                                       lon=slice(nearest_coord(region_coord[0], longitude), 
                                       nearest_coord(region_coord[2], longitude)))

    return slice_region.sum(dim=('lon','lat'))


def sum_ba_region_surface(ds_ba, region_key):
    # put 0 outside of the region and 1 in the region
    mask_region = np.zeros((180, 360, 16))
    min_lon = dict_regions[region_key][0]
    min_lat = dict_regions[region_key][1]
    max_lon = dict_regions[region_key][2]
    max_lat = dict_regions[region_key][3]
    mask_region[np.where(ds_ba.lat>=min_lat)[0][0]:np.where(ds_ba.lat<=max_lat)[0][-1],
                np.where(ds_ba.lon>=min_lon)[0][0]:np.where(ds_ba.lon<=max_lon)[0][-1], :] = 1
    # multiply the mask with the yearly average time series and sum of lon and lat to get 
    # the yearly time series of the total burned area
    return (ds_ba['burned_area_m2'].groupby(
        'time.year').sum()*mask_region).sum(dim=('lon','lat'), skipna=False)

def mean_ba_region_surface(ds_ba, region_key):
    # put 0 outside of the region and 1 in the region
    mask_region = np.zeros((16, 180, 360))
    min_lon = dict_regions[region_key][0]
    min_lat = dict_regions[region_key][1]
    max_lon = dict_regions[region_key][2]
    max_lat = dict_regions[region_key][3]
    mask_region[:, np.where(ds_ba.lat>=min_lat)[0][0]:np.where(ds_ba.lat<=max_lat)[0][-1],
                np.where(ds_ba.lon>=min_lon)[0][0]:np.where(ds_ba.lon<=max_lon)[0][-1]] = 1
    # multiply the mask with the yearly average time series and sum of lon and lat to get 
    # the yearly time series of the total burned area
    return (ds_ba['burned_area_land'].groupby(
        'time.year').mean()*mask_region).mean(dim=('lon','lat'), skipna=True)

def list_ts_region_ba(ds_ba, dict_regions):
    longitude = np.arange(-179.5, 180.5, 1)
    latitude = np.arange(-89.5, 90.5, 1)
    list_ts_year_region_surface = [[] for i in range(0, 12)]
    list_ts_year_region_ratio = [[] for i in range(0, 12)]
    list_ts_year_region_mean = [[] for i in range(0, 12)]
    for ikey, key in enumerate(dict_regions.keys()):
        list_ts_year_region_surface[ikey] = sum_ba_region_surface(ds_ba, key)
        list_ts_year_region_mean[ikey] = mean_ba_region_surface(ds_ba, key)
        list_ts_year_region_ratio[ikey] = list_ts_year_region_surface[
            ikey]/surface_area_region(dict_regions[key], longitude, latitude)
    return list_ts_year_region_surface, list_ts_year_region_ratio, list_ts_year_region_mean
        
def figure3_regional_burned_area_trend(
    ds_ba, dict_regions, dict_regions_sub, list_ts_year_region, df_burned_area, list_mk_ba):
    
    # style of the axes
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    fig = plt.figure(figsize=(18,11))

    # axes_sub to store the axes for the trends
    axes_sub = []
    # Full names of the regions
    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
                   'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']
    count=0
    for sub_x in range(0,4):
        for sub_y in range(0,3):
            # create and append the trend subplots
            axes_sub.append(plt.subplot(3,4,count+1))
            # apply a function to make the subplots look nice
            thickax(axes_sub[count])
            # title
            if list_regions[count]=='Global':
                axes_sub[count].set_title(list_regions[count], c='blue', fontsize=15, pad=8)
            else:
                axes_sub[count].set_title(list_regions[count], c='k', fontsize=15, pad=8)
            count+=1
    
    def plot_regional_yearly_ts(ds, ind, color):
        try:
            axes_sub[ind].plot(ds.year, ds.values, 'o', c=color, linewidth=2.7, 
                               markersize=6.5, alpha=0.55)
        except:
            axes_sub[ind].scatter(np.arange(2000,2016), ds, c=color, 
                                  s=40, alpha=0.55)

    def plot_trend(slope, intercept, ind, color):
        axes_sub[ind].plot(np.arange(2000, 2021), np.arange(0, 21)*slope+intercept, '--', linewidth=2.7, c=color)

    def trend_MK_ba(ds):
        try:
            results = mk.original_test(ds.values)
            return [results[2], results[7], results[8]]
        except:
            return [np.nan, np.nan, np.nan]

    def calculate_yearly_avg_region_ba(list_ts_year_region, color, color_trend, locator):
        for ind_region in range(0,12):
            ds_yr = list_ts_year_region[ind_region]/list_ts_year_region[ind_region].max()
            plot_regional_yearly_ts(ds_yr, ind_region, color)
            p_value, slope, intercept = trend_MK_ba(ds_yr)
            axes_sub[ind_region].text(2000.5, locator, 'Trend=', c='k', fontsize=12)
            if slope<0:    
                axes_sub[ind_region].text(2004, locator, f'{slope/intercept*100:.2f}'+'%/yr', c='blue', fontsize=13)
            else:
                axes_sub[ind_region].text(2004, locator,f'{slope/intercept*100:.2f}'+'%/yr', c='red', fontsize=13)             
            if p_value <= 0.10:
                axes_sub[ind_region].text(2000.5, locator-0.15, 'p_val='+f'{p_value:.2f}', c='k', fontsize=13, fontweight='bold')
            else:
                axes_sub[ind_region].text(2000.5, locator-0.15, 'p_val='+f'{p_value:.2f}', c='k', fontsize=13)
            plot_trend(slope, intercept, ind_region, color_trend)
            
    def plot_trend_regional_ba(
        list_mk_ba, df_burned_area, dict_regions, list_ts_year_region,
        color, color_trend, locator):
        for ikey, key in enumerate(dict_regions.keys()):
            if key == 'global':
                ds_yr = list_ts_year_region[ikey]/list_ts_year_region[ikey].max()
                plot_regional_yearly_ts(ds_yr, ikey, color)
                p_value, slope, intercept = trend_MK_ba(ds_yr)
            else:
                ds_yr = df_burned_area[key]/df_burned_area[key].max()
                plot_regional_yearly_ts(ds_yr, ikey, color)
                p_value, slope, intercept = trend_MK_ba(ds_yr)
#                 slope, p_value, intercept = list_mk_ba[ikey-1][1:]
                
            axes_sub[ikey].text(2000.5, locator, 'Trend=', c='k', fontsize=12)
            if slope<0:    
                axes_sub[ikey].text(2004, locator, 
                                          f'{slope/intercept*100:.2f}'+'%/yr', c='blue', fontsize=13)
            else:
                axes_sub[ikey].text(2004, locator,
                                          f'{slope/intercept*100:.2f}'+'%/yr', c='red', fontsize=13)             
            if p_value <= 0.10:
                axes_sub[ikey].text(2000.5, locator-0.15, 
                                          'p_val='+f'{p_value:.2f}', c='k', fontsize=13, fontweight='bold')
            else:
                axes_sub[ikey].text(2000.5, locator-0.15,
                                          'p_val='+f'{p_value:.2f}', c='k', fontsize=13)
            plot_trend(slope, intercept, ikey, color_trend)
            
    plot_trend_regional_ba(list_mk_ba, df_burned_area, dict_regions, list_ts_year_region,
                           'blue', 'darkblue',  1.35)
#     calculate_yearly_avg_region_ba(list_ts_year_region, 'blue', 'darkblue',  1.35)

    for i in range(0, 12):
        axes_sub[i].set_yticks([0,0.5,1])
        axes_sub[i].set_xticks([2000, 2005, 2010, 2015])
        axes_sub[i].set_xlim([2000, 2015])
        axes_sub[i].set_ylim([0, 1.5])
#         if i in [0, 4, 8]:
#             axes_sub[i].set_ylabel('Annual burned area (norm)', fontsize=14)
#         else:
#             axes_sub[i].set_yticklabels(['','',''])
        if i in [11, 10, 9, 8]:
            pass
        else:
            axes_sub[i].set_xticklabels(['','','', ''])

    fig.text(0.5, 0.05, 
             'Time (year) (%)', ha='center', fontsize=16)
    fig.text(0.07, 0.5, 
             'Normalised total yearly burned area (%)', 
             va='center', rotation='vertical', fontsize=16)
    plt.subplots_adjust(wspace=0.24, hspace=0.45)

    fig.savefig('burnedarea_trend_regions.jpeg', bbox_inches='tight', dpi=100)
    
    
def figure4_corr_aod_ba(list_ts_region_burned_area_ratio, 
                        list_ts_modis_region, dict_regions, dict_regions_sub):
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig = plt.figure(figsize=(18,11))

    # axes_sub to store the axes for the trends
    axes_sub = []
    # Full names of the regions
    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
                   'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']
    count=0
    for sub_x in range(0,4):
        for sub_y in range(0,3):
            # create and append the trend subplots
            axes_sub.append(plt.subplot(3,4,count+1))
            # apply a function to make the subplots look nice
            thickax(axes_sub[count])
            # title
            if list_regions[count]=='Global':
                axes_sub[count].set_title(list_regions[count], c='k', fontsize=15, pad=8, fontweight='bold')
            else:
                axes_sub[count].set_title(list_regions[count], c='k', fontsize=15, pad=8)
            count+=1
            
    def plot_correlation_ba_aod_region(ts_aod, ts_ba, slope, intercept, r_value, p_value, ind):
        axes_sub[ind].plot(ts_ba.values, ts_aod.values, 'o', c='darkgreen', linewidth=2.7, 
                           markersize=6, alpha=0.3)
        axes_sub[ind].plot([ts_ba.values.min(), ts_ba.values.max()], 
                           [intercept+ts_ba.values.min()*slope, intercept+ts_ba.values.max()*slope], 
                           '--', c='darkgreen', linewidth=2.7) 
        if p_value < 0.065:
            axes_sub[ind_region].text(ts_ba.values.min(), 
                                      ts_aod.values.max()-(ts_aod.values.max()-ts_aod.values.min())/7, 
                                      'Slope='+f'{slope:.3f}'+'/%'+'\n' + 'R=' + 
                                  f'{r_value:.2f}' + ' ; p_val=' + f'{p_value:.2f}', c='r', fontsize=13) 
        else:
            axes_sub[ind_region].text(ts_ba.values.min(), 
                                      ts_aod.values.max()-(ts_aod.values.max()-ts_aod.values.min())/7, 
                              'Slope='+f'{slope:.3f}'+'/%'+'\n' + 'R=' + 
                          f'{r_value:.2f}' + ' ; p_val=' + f'{p_value:.2f}', c='k', fontsize=13)
        axes_sub[ind_region].set_ylim(top=ts_aod.values.max()+(ts_aod.values.max()-ts_aod.values.min())/9)

    for ind_region in range(0,12):
        ts_ba = list_ts_region_burned_area_ratio[ind_region]*100
        ts_aod = list_ts_modis_region[ind_region]['od550aer'][0:16]
        slope, intercept, r_value, p_value, std_err = stats.linregress(ts_ba.values, ts_aod)
        plot_correlation_ba_aod_region(ts_aod, ts_ba, slope, intercept, r_value, p_value, ind_region)

    fig.text(0.5, 0.05, 'Total regional yearly burned area (%)', ha='center', fontsize=17)
    fig.text(0.06, 0.5, 'MODIS AOD (550nm)', va='center', rotation='vertical', fontsize=16)
            
    plt.subplots_adjust(wspace=0.3, hspace=0.45)
    fig.savefig('correlation_burnedarea_aod_regions.jpeg', bbox_inches='tight', dpi=100)
    
    
def correlation_aod_regionalBA(ds_aod, ts_region_burned_area_surface):
    slope_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    p_val_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    r_val_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    intercept_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    for ilat, lat in enumerate(ds_aod.latitude):
        for ilon, lon in enumerate(ds_aod.longitude):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    ds_aod['od550aer'].isel(latitude=ilat, longitude=ilon)[0:16], 
                    ts_region_burned_area_surface)

                slope_array[ilon, ilat] = slope
                p_val_array[ilon, ilat] = p_value
                r_val_array[ilon, ilat] = r_value
                intercept_array[ilon, ilat] = intercept
            except:
                pass
    return slope_array, p_val_array, r_val_array, intercept_array

def yearly_mean_ba(data):
    # 6 months offset to have a tick at the middle of the year
    return data.resample({'time': 'Y'}, loffset = '-6M').mean()

def add_trend_MK_burned_area(data_ba):
#     data_ba = ds_ba['burned_area_land']
    trend = np.zeros((data_ba.lat.shape[0], data_ba.lon.shape[0]))*np.nan
    p_val = np.zeros((data_ba.lat.shape[0], data_ba.lon.shape[0]))*np.nan
    intercept = np.zeros((data_ba.lat.shape[0], data_ba.lon.shape[0]))*np.nan
    for ilat, lat in enumerate(data_ba.lat):
        for ilon, lon in enumerate(data_ba.lon):
            ts = data_ba.isel(lat=ilat, lon=ilon)
            try:
                results = mk.original_test(ts)
                trend[ilat, ilon] = results[7]
                intercept[ilat, ilon] = results[8]
                p_val[ilat, ilon] = results[2]
            except:
                pass
    data_ba['trend_mk'] = (['lat', 'lon'],  trend)
    data_ba['p_val_mk'] = (['lat', 'lon'],  p_val)
    data_ba['intercept_mk'] = (['lat', 'lon'],  intercept)
    return data_ba

def correlation_aod_regionalBA(ds_aod, ts_region_ba, key):
    slope_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    p_val_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    r_val_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    intercept_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    for ilat, lat in enumerate(ds_aod.latitude):
        for ilon, lon in enumerate(ds_aod.longitude):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    ds_aod['od550aer'].isel(latitude=ilat, longitude=ilon)[0:16], 
                    ts_region_ba)

                slope_array[ilon, ilat] = slope
                p_val_array[ilon, ilat] = p_value
                r_val_array[ilon, ilat] = r_value
                intercept_array[ilon, ilat] = intercept
            except:
                pass
            
    ds_aod['slope_linreg_'+key] = (['latitude', 'longitude'],  slope_array.T)
    ds_aod['p_val_linreg_'+key] = (['latitude', 'longitude'],  p_val_array.T)
    ds_aod['r_val_linreg_'+key] = (['latitude', 'longitude'],  r_val_array.T)
    ds_aod['intercept_linreg_'+key] = (['latitude', 'longitude'],  intercept_array.T)
    return ds_aod

def correlation_regionalba_aod(ds_modis_y, list_ts_region_burned_area, dict_regions):
    for ikey, key in enumerate(dict_regions.keys()):
        ds_modis_y_corr_ba = correlation_aod_regionalBA(
            ds_modis_y, list_ts_region_burned_area[ikey], key)
    return ds_modis_y_corr_ba

def subplot_map_r_val(ax, ds_modis_y_corr_ba, dict_regions, key):
    def line_regions(ll_coord, ur_coord, ax):
        c = 'yellow'
        lw = 2
        coordll = ll_coord
        coordur = ur_coord
        coordlr = [ur_coord[0], ll_coord[1]]
        coordul = [ll_coord[0], ur_coord[1]]
        line = matplotlib.lines.Line2D((coordll[0],coordul[0]),(coordll[1],coordul[1])
                                       , linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordll[0],coordlr[0]),(coordll[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordul[0]),(coordur[1],coordul[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordlr[0]),(coordur[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        ax.coastlines()
        
    line_regions(dict_regions[key][0:2], dict_regions[key][2:4], ax)

    cmap = plt.cm.RdBu.reversed()
    cbar_sub = ds_modis_y_corr_ba['r_val_linreg_'+key].plot(
        ax = ax, cmap='seismic', levels = np.arange(-1,1.2, 0.2), add_colorbar=False)

    # add dots where the p-value is significant
    trend_pval = ds_modis_y_corr_ba['p_val_linreg_'+key].where(ds_modis_y_corr_ba['p_val_linreg_'+key]<0.05,np.nan).values
    trend_pval[~np.isnan(trend_pval)] = 1

    # put one dot every 2 dots
    mask = np.ones(trend_pval.shape)*np.nan
    for i in range(0, len(ds_modis_y_corr_ba.longitude), 4):
        for j in range(0, len(ds_modis_y_corr_ba.latitude), 4):
            mask[j, i] = 3
    trend_pval = trend_pval * mask
    x, y = np.meshgrid(ds_modis_y_corr_ba.longitude, ds_modis_y_corr_ba.latitude)
    ax.scatter(x, y, s=trend_pval, 
                   transform=ccrs.PlateCarree(), c='yellow', alpha=0.9)
    ax.set_title(key, fontsize=15)
    
    return cbar_sub

def subplot_map_corr(ax, ds_modis_y_corr_ba, dict_regions, key):
    def line_regions(ll_coord, ur_coord, ax):
        c = 'red'
        lw = 2
        coordll = ll_coord
        coordur = ur_coord
        coordlr = [ur_coord[0], ll_coord[1]]
        coordul = [ll_coord[0], ur_coord[1]]
        line = matplotlib.lines.Line2D((coordll[0],coordul[0]),(coordll[1],coordul[1])
                                       , linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordll[0],coordlr[0]),(coordll[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordul[0]),(coordur[1],coordul[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordlr[0]),(coordur[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax.add_line(line)
        ax.coastlines()
        
    line_regions(dict_regions[key][0:2], dict_regions[key][2:4], ax)

    cmap = plt.cm.RdBu.reversed()
    cbar_sub = ds_modis_y_corr_ba['slope_linreg_'+key].plot(
        ax = ax, cmap=cmap, add_colorbar=False)

    # add dots where the p-value is significant
    trend_pval = ds_modis_y_corr_ba['p_val_linreg_'+key].where(ds_modis_y_corr_ba['p_val_linreg_'+key]<0.05,np.nan).values
    trend_pval[~np.isnan(trend_pval)] = 1

    # put one dot every 2 dots
    mask = np.ones(trend_pval.shape)*np.nan
    for i in range(0, len(ds_modis_y_corr_ba.longitude), 5):
        for j in range(0, len(ds_modis_y_corr_ba.latitude), 5):
            mask[j, i] = 2
    trend_pval = trend_pval * mask
    x, y = np.meshgrid(ds_modis_y_corr_ba.longitude, ds_modis_y_corr_ba.latitude)
    ax.scatter(x, y, s=trend_pval, 
                   transform=ccrs.PlateCarree(), c='k', alpha=0.9)
    
    ax.set_title(key, fontsize=15)
    
    return cbar_sub

def figure5_correlation_aod_regionalBA(data_ba, list_region_burned_area, ds_modis_y_corr_ba, dict_regions):
    
    ## Defining the figure and axes frame
    fig = plt.figure(figsize=(25,14))
    gs = gridspec.GridSpec(ncols=5, nrows=5, hspace = 0.6, wspace=0.3)

    # Central map
    ax_map = fig.add_subplot(gs[1:4,1:4], projection=ccrs.PlateCarree())
    ax_map.coastlines()
    ax_map.stock_img()

    # # Defining the subplots for the trends by region
    # sub_array contains 1 when the number of subplot in the frame
    # has to be displayed, and 0 everywhere else
    sub_array = np.ones((5,5))
    sub_array[1:4, 1:4] = np.zeros((3,3))
    sub_array[0, 0] = 0
    sub_array[4, 4] = 0
    sub_array[4, 0] = 0
    sub_array[0, 4] = 0

    # axes_sub to store the axes for the trends
    axes_sub = []

    # style of the axes
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
               'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']

    count=0
    for sub_x in range(0,5):
        for sub_y in range(0,5):
            if sub_array[sub_x, sub_y] == 1:
                # create and append the trend subplots
                axes_sub.append(fig.add_subplot(gs[sub_x,sub_y], projection=ccrs.PlateCarree()))
                axes_sub[count].coastlines()
                # apply a function to make the subplots look nice
                thickax(axes_sub[count])
                if list_regions[count]=='Global':
                    axes_sub[count].set_title(list_regions[count], fontweight='bold', c='k', fontsize=16, pad=8)
                else:
                    axes_sub[count].set_title(list_regions[count], c='k', fontsize=16, pad=8)
                count+=1

    # Method to plot the lines that connect the subplots with the regions on the central map
    transFigure = fig.transFigure.inverted()
    def line_connect(ax, c1, c2):
        coord1 = transFigure.transform(ax.transData.transform(c1))
        coord2 = transFigure.transform(ax_map.transData.transform(c2))
        line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                       transform=fig.transFigure, linewidth=2, linestyle='--', c='k')
        fig.add_artist(line)

    line_connect(axes_sub[0], [0.2,0], [-147, 85]) # Global
    line_connect(axes_sub[1], [0.6,0], [0, 60]) # Europe
    line_connect(axes_sub[2], [0.8,0], [150, 60]) # Russia
    line_connect(axes_sub[5], [1,0.3], [-120, 25]) # S- NA
    line_connect(axes_sub[4], [0,0.3], [130, 25]) # Japan
    line_connect(axes_sub[3], [1,0.6], [-140, 60]) # N- NA
    line_connect(axes_sub[6], [0,0.3], [80, 12]) # India
    line_connect(axes_sub[8], [0,0.6], [130, -25]) # Australia
    line_connect(axes_sub[11], [0.5,1.27], [30, -35]) # South-Africa
    line_connect(axes_sub[10], [0.37,1.27], [-12, 25]) # N-Africa
    line_connect(axes_sub[9], [0.5,1.27], [-75, -55]) # S- SA
    line_connect(axes_sub[7], [1,0.5], [-70, 0]) # N- SA

    list_key_regions =  ['global','europe','russia','nna','easia','sna',
                         'wasia','nsa','australia','ssa','nafrica','safrica']

    def line_regions(ll_coord, ur_coord):
        c = 'k'
        lw = 2
        coordll = ll_coord
        coordur = ur_coord
        coordlr = [ur_coord[0], ll_coord[1]]
        coordul = [ll_coord[0], ur_coord[1]]
        line = matplotlib.lines.Line2D((coordll[0],coordul[0]),(coordll[1],coordul[1])
                                       , linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordll[0],coordlr[0]),(coordll[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordul[0]),(coordur[1],coordul[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)
        line = matplotlib.lines.Line2D((coordur[0],coordlr[0]),(coordur[1],coordlr[1]),
                                       linewidth=lw, linestyle='-', c=c)
        ax_map.add_line(line)

    for key_region in dict_regions:
        line_regions(dict_regions[key_region][0:2], dict_regions[key_region][2:4])

    # Plot the central map
    # vmin = -10
    # vmax = 10
    lon = np.arange(-179.5, 180.5, 1)
    lat = np.arange(-89.5, 90.5, 1)

    # calculate the relative trend of modis per grid cell over land
    ds_trend_ba = data_ba['trend_mk']/data_ba['intercept_mk']*100

    # set color for the nan values (white)
    null_bkgd_ds_trend_ba = ds_trend_ba/ds_trend_ba*4
    null_bkgd_ds_trend_ba = null_bkgd_ds_trend_ba.fillna(0.3)
    null_bkgd_ds_trend_ba.plot(ax = ax_map, cmap = 'Greys', add_colorbar=False, vmin=0, vmax=4)

    # plot the map in the central axe
    cmap = plt.cm.RdBu.reversed()
    cmap.set_bad('white',0.5)
    
    mappable_cbar = ds_trend_ba.plot(ax = ax_map, cmap=cmap, add_colorbar=False, 
                                       levels = np.arange(-10, 11, 1))

    ax_map.set_title('       Burned area          Trend for 2000-2015', fontsize=18)

    # add dots where the p-value is significant
    trend_ba_pval = data_ba['p_val_mk'].where(data_ba['p_val_mk']<0.05,np.nan).values
    trend_ba_pval[~np.isnan(trend_ba_pval)] = 1

    # put one dot every 2 dots
    mask = np.ones(trend_ba_pval.shape)*np.nan
    for i in range(0, len(data_ba.lon), 3):
        for j in range(0, len(data_ba.lat), 3):
            mask[j, i] = 7
    trend_ba_pval = trend_ba_pval * mask
    x, y = np.meshgrid(data_ba.lon, data_ba.lat)
    ax_map.scatter(x, y, s=trend_ba_pval, 
                   transform=ccrs.PlateCarree(), c='k', alpha=0.5)

    for ikey, key in enumerate(dict_regions.keys()):
        cb_sub = subplot_map_corr(axes_sub[ikey], 
                                  ds_modis_y_corr_ba, dict_regions, key)
    
    # Parameters of the colorbar
    axins = inset_axes(ax_map,
                        width="3.8%",  
                        height="100%",
                        loc='center right',
                        borderpad=-39
                       )
    cb = fig.colorbar(mappable_cbar, cax=axins, orientation="vertical")
    cb.set_label(r'Burned area trend %/yr', labelpad=14, size=16, rotation=90)
    cb.ax.tick_params(labelsize=14)

    # Save the figure
    fig.savefig('figure5_correlation_aod_regionalBA.jpeg', bbox_inches='tight', dpi=100)
    
def figure6bis(data_ba, list_region_burned_area, ds_modis_y_corr_ba, dict_regions):
    
    ## Defining the figure and axes frame
    fig = plt.figure(figsize=(25,10))
    gs = gridspec.GridSpec(ncols=4, nrows=3, hspace = 0.1, wspace=0.1)

    # axes_sub to store the axes for the trends
    axes_sub = []

    # style of the axes
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        plt.rc('axes', linewidth=1.3)
        fontsize = 13
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        ax.tick_params(direction='out', length=4, width=1.3, pad=12, 
                       bottom=True, top=False, left=True, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    list_regions = ['Global','Europe', 'Russia', 'North NA', 'East Asia', 'South NA', 
               'West Asia', 'North SA', 'Australia', 'South SA', 'North Africa', 'South Africa']

    count=0
    for sub_x in range(0,3):
        for sub_y in range(0,4):
            # create and append the trend subplots
            axes_sub.append(fig.add_subplot(gs[sub_x,sub_y], projection=ccrs.PlateCarree()))
            axes_sub[count].coastlines()
            # apply a function to make the subplots look nice
            thickax(axes_sub[count])
            if list_regions[count]=='Global':
                axes_sub[count].set_title(list_regions[count], fontweight='bold', c='k', fontsize=16, pad=8)
            else:
                axes_sub[count].set_title(list_regions[count], c='k', fontsize=16, pad=8)
            count+=1

    list_key_regions =  ['global','europe','russia','nna','easia','sna',
                         'wasia','nsa','australia','ssa','nafrica','safrica']

    for ikey, key in enumerate(dict_regions.keys()):
        cb_sub = subplot_map_r_val(axes_sub[ikey], 
                                  ds_modis_y_corr_ba, dict_regions, key)
    
#     # Parameters of the colorbar
    axins = inset_axes(axes_sub[7],
                        width="5%",  
                        height="300%",
                        loc='center right',
                        borderpad=-5
                       )
    cb = fig.colorbar(cb_sub, cax=axins, orientation="vertical")
    cb.set_label(r'R val - linear correlation between regional burned area and AOD', labelpad=14, size=15, rotation=90)
    cb.ax.tick_params(labelsize=14)

    # Save the figure
    fig.savefig('figure6bis_correlation_aod_regionalBA.jpeg', bbox_inches='tight', dpi=100)
    
    
def regional_yearly_accumulated_BA(ds_ba, region_coord):
    ds_ba['land_mask'] = mask_ocean(ds_ba, ds_ba.lon, ds_ba.lat)
    longitude = ds_ba.lon.values
    latitude = ds_ba.lat.values
    slice_ds = ds_ba.sel(lat=slice(nearest_coord(region_coord[1], latitude), 
                                   nearest_coord(region_coord[3], latitude)), 
                                   lon=slice(nearest_coord(region_coord[0], longitude), 
                                   nearest_coord(region_coord[2], longitude)))
    return (slice_ds.sum(dim=['lat', 'lon'])/((~np.isnan(slice_ds['land_mask']))[0,:,:].sum().values
                                             )*100).resample({'time': 'Y'}).sum()

#     return (slice_ds*100).mean(dim='time')
#     return (slice_ds*100).resample({'time': 'Y'}).sum(dim='time').mean(dim='time')

# jas_ba = ds_ba.sel(time=is_jas(ds_ba['time.month']))
# djf_ba = ds_ba.sel(time=is_djf(ds_ba['time.month']))

def create_list_regionalts_ba(ds_ba, dict_regions):
    list_regionalts_ba = []
    for ikey, key in enumerate(dict_regions.keys()):
        list_regionalts_ba.append(
            regional_yearly_accumulated_BA(ds_ba, dict_regions[key]))
    return list_regionalts_ba

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], 
                        colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def create_df_ba_ts(list_regional_ts, dict_regions):
    df = pd.DataFrame(columns = ['global','europe','russia','nna','easia','sna',
                                 'wasia','nsa','australia','ssa','nafrica','safrica'])
    for ikey, key in enumerate(dict_regions):
        df.loc[:,key] = list_regional_ts[ikey]
    return df

def create_df_aod_ts(list_regional_ts, dict_regions):
    df = pd.DataFrame(columns = ['global','europe','russia','nna','easia','sna',
                                 'wasia','nsa','australia','ssa','nafrica','safrica'])
    for ikey, key in enumerate(dict_regions):
        df.loc[:,key] = list_regional_ts[ikey]['od550aer'].values
    return df

def corr_matrix(df, title, ax):
    seismic_discrete = cmap_discretize('seismic', 8)
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=seismic_discrete,
                square=True, ax=ax, vmin=-1, vmax=1, linewidths=.5, linecolor='lightgray',
               cbar_kws={'label': title})
    ax.figure.axes[-1].yaxis.label.set_size(18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(rotation=90)
    
def figure_corr_matrix(ds_ba, list_ts_modis_region):
    list_regionalts_ba = create_list_regionalts_ba(ds_ba, dict_regions)
    df_burned_area = create_df_ba_ts(list_regionalts_ba, dict_regions)
    df_aod = create_df_aod_ts(list_ts_modis_region, dict_regions)
    fig = plt.figure(num=None, figsize=(26, 9.5))
    ax = plt.subplot(1,2,1)
    corr_matrix(df_aod, 'Correlation cross regional AOD', ax)
    ax = plt.subplot(1,2,2)
    corr_matrix(df_burned_area, 'Correlation cross regional burned area', ax)
    
def multivariate_regression_regionalba_aod(df_burned_area, ds_aod, dict_regions):
    X = df_burned_area.loc[:, 'europe':'safrica']
    r_square_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude)))*np.nan
    coef_array = np.zeros((len(ds_aod.longitude), len(ds_aod.latitude), X.shape[1]))*np.nan
    for ilat, lat in enumerate(ds_aod.latitude):
        for ilon, lon in enumerate(ds_aod.longitude):
            y = ds_aod['od550aer'].isel(latitude=ilat, longitude=ilon)[0:16].values
            try:
                regr = linear_model.LinearRegression()
                regr.fit(X, y)
                r_square_array[ilon, ilat] = regr.score(X, y)
                coef_array[ilon, ilat, :] = regr.coef_
            except:
                pass
    
    ds_aod['multivar_regr_R_square'] = (['latitude', 'longitude'],  r_square_array.T)
    for ikey, key in enumerate(dict_regions.keys()):
        if key != 'global':
            ds_aod['multivar_regr_coef_'+key] = (['latitude', 'longitude'],  coef_array[:,:,ikey-1].T)
    
    return ds_aod

def mk_ba_region_slope_pval_intercept(df_burned_area, dict_regions):
    list_mk_ba =[]
    for ikey, key in enumerate(dict_regions.keys()):
        if key != 'global':
            list_mk_ba.append([key, mk.original_test(df_burned_area[key])[7], 
                               mk.original_test(df_burned_area[key])[2], 
                               mk.original_test(df_burned_area[key])[8]])
    return list_mk_ba

def increase_aod_dueto_ba_2000_2015(list_mk_ba, ds_aod, dict_regions):
    for ikey, key in enumerate(dict_regions.keys()):
        if key != 'global':
            # if p val < 0.1 use the regional coef
            if (list_mk_ba[ikey-1][2] <= 0.1):
                ds_aod['contrib_'+key] = (['latitude', 'longitude'],  
                                          ds_aod['multivar_regr_coef_'+key]*list_mk_ba[ikey-1][1]*15)
    return ds_aod

def create_df_regional_contribution(ds_aod, dict_regions):
    
    ds_contrib = (ds_aod['contrib_europe']+ds_aod['contrib_nna']+
                  ds_aod['contrib_easia']+ds_aod['contrib_ssa']+
                  ds_aod['contrib_nafrica'])
    
    # Apply mask
    lon_grid, lat_grid = np.meshgrid(ds_contrib.longitude,ds_contrib.latitude)
    mask = globe.is_land(lat_grid, lon_grid)
    ds_contrib = ds_contrib.where(mask, np.nan)
    
    count=0
    list_gridcell_region = []
    for ikey, key in enumerate(dict_regions.keys()):
        list_gridcell_region.append(ds_contrib.sel(
            latitude=slice(nearest_coord(dict_regions[key][3], latitude), 
                           nearest_coord(dict_regions[key][1], latitude)), 
            longitude=slice(nearest_coord(dict_regions[key][0], longitude), 
                            nearest_coord(dict_regions[key][2], longitude))).values.reshape([-1,1])[:, 0])
        if count < list_gridcell_region[ikey].shape[0]:
            count = list_gridcell_region[ikey].shape[0]
            
    df_regional_contribution = pd.DataFrame(
        data=np.zeros((count, 12))*np.nan,
        columns=['global','europe','russia','nna','easia','sna','wasia',
               'nsa','australia','ssa','nafrica','safrica'])
    
    for ikey, key in enumerate(dict_regions.keys()):
        df_regional_contribution.loc[
            0:list_gridcell_region[ikey].shape[0]-1, key] = list_gridcell_region[ikey]
    
    return df_regional_contribution

def figure_contribution_biomassburning_to_2000_2015_aod(ds_aod, ds_modis_y, list_mk_ba, df_regional_contribution):
    # sum of contributions from regions that exhibit significative trends in burned area
    ds_contrib = (ds_aod['contrib_europe']+ds_aod['contrib_nna']+
                  ds_aod['contrib_easia']+ds_aod['contrib_ssa']+
                  ds_aod['contrib_nafrica'])
    
    fig = plt.figure(num=None, figsize=(25, 9.5))
    ax = plt.subplot(2,2,2, projection=ccrs.PlateCarree())
    # sum of contributions from regions that exhibit significative trends in burned area
    mappable_cbar_aod = ds_contrib.plot(
        vmin=-0.5, vmax=0.5, cmap='seismic', ax=ax, add_colorbar=False,
        levels = np.arange(-0.5, 0.55, 0.05))#, 
        #bar_kwargs={'label': 'AOD evolution from 2000 to 2015'})
    ax.set_title(
        'Absolute AOD contribution from biomass \n burning during 2000-2015', fontsize=16)
    ax.coastlines()
    
    axins = inset_axes(ax,
                width="3.6%",  
                height="100%",
                loc='center right',
                borderpad=-10
               )
    
    ax = plt.subplot(2,2,1, projection=ccrs.PlateCarree())
    # sum of contributions from regions that exhibit significative trends in burned area
    (ds_modis_y['trend_mk']*15).plot(
        vmin=-0.5, vmax=0.5, cmap='seismic', ax=ax, add_colorbar=False,
        levels = np.arange(-0.5, 0.55, 0.05))#, 
        #cbar_kwargs={'label': 'AOD evolution from 2000 to 2015'})
    ax.set_title(
        'Absolute AOD trend \n during 2000-2015', fontsize=14)
    ax.coastlines()
    
    ax = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
    # sum of contributions from regions that exhibit significative trends in burned area
    mappable_cbar_rsquare = ds_aod['multivar_regr_R_square'].plot(
        vmin=0.4, vmax=1, cmap='viridis', ax=ax, add_colorbar=False)#, 
        #cbar_kwargs={'label': r'$R^2$'})
    ax.coastlines()
    ax.set_title(
        r'Multivariate regression score ($R^2$)', fontsize=16)
    
    axins_n = inset_axes(ax,
                width="3.8%",  
                height="100%",
                loc='center right',
                borderpad=-10
               )
    
    def thickax(ax):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.7)
            ax.spines[axis].set_color('grey')
        plt.rc('axes', linewidth=0.9)
        fontsize = 10
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
    #         tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
    #         tick.label1.set_fontweight('bold')
        ax.tick_params(direction='in', length=4, width=0.9, pad=10, 
                       bottom=True, top=True, left=True, right=True, color='grey')
    
    ax = plt.subplot(2,2,3)
    color_data = ['k', 'r', 'y', 'b', 'g', 'm', 'c', 'grey', 
                  'orange', 'pink', 'darkred', 'springgreen']
    regions = ['global','europe','russia','nna','easia','sna','wasia',
               'nsa','australia','ssa','nafrica','safrica']
    boxprops = dict(linestyle='-', linewidth=2, color='k')
    color={'medians': 'red', 'boxes': 'grey', 'whiskers': 'grey', 'caps': 'grey'}
    medianprops = dict(linestyle='-', linewidth=2, color='r')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
    whiskerprops = dict(linewidth=1.5)
    capprops = dict(linewidth=2)
    df_regional_contribution.boxplot(column=regions, 
                                     showfliers=False, whiskerprops=whiskerprops, capprops=capprops,
                                     ax=ax, grid=False, meanprops=meanpointprops, meanline=False,
                                     showmeans=True, boxprops=boxprops, medianprops=medianprops, color=color)
    ax.set_ylabel('Regional AOD contribution \n from biomass burning', fontsize=14)
    ax.set_xticklabels(regions, color=color_data)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), color_data):
        ticklabel.set_color(tickcolor)
    ax.axhline(y=0, linestyle=':', color='k')
    thickax(ax)
    
    plt.subplots_adjust(wspace=0.07, hspace=0.2)
    
    cb = fig.colorbar(mappable_cbar_aod, cax=axins, orientation="vertical")
    cb.set_label('Absolute AOD change \n from 2000 to 2015', labelpad=14, size=16, rotation=90)

    cb = fig.colorbar(mappable_cbar_rsquare, cax=axins_n, orientation="vertical")
    cb.set_label(r'$R^2$', labelpad=14, size=16, rotation=90)