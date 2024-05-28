# connect google drive where datasets are stored
import os, sys
from google.colab import drive
drivedir='/content/drive'
drive.mount(drivedir)
os.chdir(drivedir)

%%capture
# install required modules I
!apt-get install libproj-dev proj-data proj-bin
!apt-get install libgeos-dev
!pip install cython
!pip install cartopy

%%capture
# install required modules II
!apt-get -qq install python-cartopy python3-cartopy
!pip uninstall -y shapely    # cartopy and shapely aren't friends (early 2020)
!pip install shapely --no-binary shapely

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from scipy import signal
import copy

## Define functions that will be used for the analysis
def anom(xarr):
    #monthly mean anomaly (No-detrened)
    xarr_clm = xarr.groupby('time.month').mean('time')
    xarr_ano = xarr.groupby('time.month') - xarr_clm
    return xarr_ano, xarr_clm

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    # used in the next function
    p = da.polyfit(dim=dim, deg=deg) #Returns a vector of coefficients p that minimises the squared error
    fit = xr.polyval(da[dim], p.polyfit_coefficients)#use these coefficients to get the trend
    return da - fit

def anomd(xarr):
    #detrened monthly mean anomaly
    xarr_det=detrend_dim(xarr,'time', deg=1)
    xarr_clm = xarr_det.groupby('time.month').mean('time')
    xarr_ano = xarr_det.groupby('time.month') - xarr_clm
    return xarr_ano, xarr_clm

def npcovariance(FLDtyx, IDXt):
    # Regression map between a field (T,X,Y) and a timeseries (T)
    FLD = FLDtyx - np.mean(FLDtyx, 0)
    IDX = IDXt - np.mean(IDXt)
    FLDdotIDX = np.einsum('kij,k->ij', FLD, IDX) #einsum does the matrix multiplication
    return FLDdotIDX / IDXt.__len__() #__len__ returns the length of the object

def npcorrelation(FLDtyx, IDXt):
    # Correlation map between a field (T,X,Y) and a timeseries (T) (you just normalise the rgression)
    return npcovariance(FLDtyx, IDXt) / (np.std(FLDtyx, 0) * np.std(IDXt, 0))

def npregress(FLDtyx, IDXt, std_units='yes'):
    # regression coeff. per unit of STD
    # x:fld ,y:index
    if std_units == 'yes':
        # print('Reg. coeff per units of index')
        return npcovariance(FLDtyx, IDXt) / np.std(IDXt, 0)
    else:
        # print('Reg. coeff per units of STD')
        return npcovariance(FLDtyx, IDXt) / np.var(IDXt, 0)

def nplagreg(FLDtyx, IDXt, lags):
    # Lag-regression maps
    #Works only if the absolute value of any lags is smaller than 24
    T, Y, X = FLDtyx.shape
    lagreg = np.zeros([len(lags), Y, X])
    lagmin = min(lags)
    lagmax = max(lags)
    for i in range(len(lags)):
        IDXt[-(lagmin) + lags[i]:-lagmax + lags[i]].shape
        FLDtyx[-(lagmin):-lagmax].shape
        lagreg[i, :, :] = npregress(FLDtyx[24:-24, :], IDXt[24 + lags[i]: -24 + lags[i]])
    return lagreg

datadir=drivedir+'/MyDrive/UNI_UNIBO/Final_project'
dse=xr.open_dataset(datadir+'/ERA5_1958-2022_newg.nc')
dse['t2m']=dse['t2m']-273.15
dse['t2m'] = dse.t2m.assign_attrs(units='°C')
dse['sst']=dse['sst']-273.15
dse['sst'] = dse.sst.assign_attrs(units='°C')
dse['tp']=dse['tp']*1000*30.5
dse['tp'] = dse.tp.assign_attrs(units='mm') # mm accumulated in a month

#The current data go from lat -280 to 80, they are interrupted on my index area
lons=dse['lon'].values.copy()
lons[lons < -80] = lons[lons < -80] + 360
dse['lon'] = xr.DataArray(lons, coords=dse['lon'].coords, dims=dse['lon'].dims)
dse = dse.sortby('lon') #now they go from -80 to 280
dse['lon'] = dse['lon'] - 100 #now they go from -180 to 180 with 0th longitude being close to India
#so my index (68,89) will be (-32,-11)


dso=xr.open_dataset(datadir+'/ORAS5_1958-2022_newg.nc')
dso=dso.rename({'time_counter': 'time'}) #rename dimaesion time_counter so that it matches ERA5 name for time dimension
lons=dso['lon'].values.copy()
lons[lons < -80] = lons[lons < -80] + 360
dso['lon'] = xr.DataArray(lons, coords=dso['lon'].coords, dims=dso['lon'].dims)
dso = dso.sortby('lon') #now they go from -80 to 280
dso['lon'] = dso['lon'] - 100 #now they go from -180 to 180


# Investigate the Indian monsoons using the dataset
## Total precipitation
# Plotting temporal mean to quickly check the dataset
fig = plt.figure(figsize=(22,7))
data=dse['tp'].mean('time')
# Add the cyclic point
data,lons=add_cyclic_point(data,coord= dse['lon'])
ax1 = fig.add_subplot( projection=ccrs.PlateCarree(central_longitude=100))
cs =dse['tp'].mean('time').plot.pcolormesh(vmax=550, vmin=0)
ax1.coastlines(resolution='auto', color='k')
ax1.set(title='total precipitation mean [mm/month] - long-term average')
plt.show()


#total precipitation accumulated in each year (average)
year= dse['tp'].resample(time='1y').sum()

fig = plt.figure(figsize=(22,7))
data=year.mean('time') #mean
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=year['lon'])
ax1 = fig.add_subplot( projection=ccrs.PlateCarree(central_longitude=100))
cs = year.mean('time').plot.contourf(ax=ax1, cmap='hsv',vmax=5000, vmin=0,alpha=0.6)
ax1.coastlines(resolution='auto', color='k')
ax1.set(title=' total precipitation [mm/yr] - long-term average')

plt.show()


#plot the raw timeseries to investigate the temporal trend and seasonal cycle in my index region (68,89)
tp=dse['tp'].sel(lat=slice(18, 30), lon=slice(-32,-11)).mean(dim=('lat', 'lon')).compute()
time=dse['tp'].time

plt.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22,7))
ax.plot(time, tp,'b')

ax.set(xlabel="Date", ylabel="[mm]", title="Total precipitation over India [mm/month]")
ax.grid()


#zoom in on the seasonal cycle
tp_2years=tp.sel(time=slice('1970-01-01','1971-12-31'))

plt.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22,7))
ax.plot(tp_2years.time, tp_2years,'b')

ax.set(xlabel="Date", ylabel="[mm]", title="Total precipitation over India 1970-1971 [mm/month])")
ax.grid()


#plotting the monthly mean climatology
TPa, TP_clim = anom(tp)

plt.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,12))
ax.plot(TP_clim,'r')

ax.set(xlabel="Month", ylabel="[mm]",
       title="Monthly mean climatology [mm/month]", xticks=(0,1,2,3,4,5,6,7,8,9,10,11),
       xticklabels=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                    'sept', 'oct', 'nov', 'dec'])
ax.grid()


# Plotting july climatology to highlight the monsoon
fig = plt.figure(figsize=(22,7))
TP_anomaly, TP_climatology = anom(dse['tp'])
data= TP_climatology[6]
# Add the cyclic point
data,lons=add_cyclic_point(data,coord= dse['lon'])
ax1 = fig.add_subplot( projection=ccrs.PlateCarree(central_longitude=100))
cs =TP_climatology[6].plot.pcolormesh(vmax=600, vmin=0)
ax1.coastlines(resolution='auto', color='k')
ax1.set(title='july climatology [mm/month]')

plt.show()


# focusing over India: how does the total precipitation and winds change from jan to july?
dse_focus=dse.sel(lat=slice(5,30), lon=slice(-40,0))
U10a, U10_clm = anom(dse_focus['u10'])
V10a, V10_clm = anom(dse_focus['v10'])
TPanm ,TP_clm=anom(dse_focus['tp'])

fig, axs = plt.subplots(nrows=1,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(28,20))

y2d, x2d = np.meshgrid(dse_focus['lat'], dse_focus['lon'])

data= TP_clm[0] #tp in jan
#Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse_focus['lon'])
cs = axs[0].contourf(lons,dse_focus['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='Blues',extend='both',levels=np.arange(0,500,100))

axs[0].quiver(x2d[:,:], y2d[:,:], U10_clm[0,:,:],
                V10_clm[0,:,:], scale=150,
                transform=ccrs.PlateCarree(central_longitude=100))

axs[0].coastlines(resolution='auto', color='k')
axs[0].set_title('january climatology [mm/month]')
gls = axs[0].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
gls.top_labels=False
gls.right_labels=False
axs[0].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))

data= TP_clm[6] #tp in jul
#Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse_focus['lon'])
cs = axs[1].contourf(lons,dse_focus['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='Blues',extend='both',levels=np.arange(0,500,100))

axs[1].quiver(x2d[:,:], y2d[:,:], U10_clm[6,:,:],
                V10_clm[6,:,:], scale=150,
                transform=ccrs.PlateCarree(central_longitude=100))

axs[1].coastlines(resolution='auto', color='k')
axs[1].set_title('july climatology [mm/month]')
gls = axs[1].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
gls.top_labels=False
gls.right_labels=False
axs[1].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
fig.colorbar(cs,orientation='horizontal')

plt.show()


#using the climatology to understand what percentage of the annual rain over india is caused by monsoons.
TP_monsoon=0
TP_annual=0
for i in range(0,12):
  if i in range(5,9): #between july and september
    TP_monsoon += TP_clim[i].values
    TP_annual += TP_clim[i].values
  else:
    TP_annual += TP_clim[i].values

percentage = round((TP_monsoon/TP_annual)*100)
print('Inian monsoons constitue '+str(percentage)+'% of the annual rain over India.')


## PDF
#plotting the PDF and the PDF in two distinct periods to understand if it changes over time (maybe with climate change?)
bins = np.arange(-2, 350, 2)

hist1, bin_edges = np.histogram(tp, bins, density=True)

y2 = tp.sel(time=slice('1958-01-01', '1999-12-31'))
hist2, bin_edges = np.histogram(y2, bins, density=True)

y3 = tp.sel(time=slice('2000-01-01', '2022-12-31'))
hist3, bin_edges = np.histogram(y3, bins, density=True)

bin_center = (bin_edges[0:-1]+bin_edges[1:])/2

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,12),
                       constrained_layout=True)
ax[0].plot(bin_center, hist1,'r')
ax[1].plot(bin_center, hist2,'g', label='1958-2000')
ax[1].plot(bin_center, hist3,'m', label='2000-2022')

ax[0].grid()
ax[0].set_title("PDF monthly mean precipitation [mm/month]")
ax[1].grid()
ax[1].legend()
ax[1].set_title("PDF monthly mean precipitation [mm/month]")

plt.show()


## Extract an index that track the variability of the Indian monsoons
#plotting the anomaly (with and without trend to see if it's affrected by climate change)
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20,12), constrained_layout = True)

TPindex, TP_clim_d = anomd(tp) #detrended

axs.plot(TPa.time,TPa,color='red',label='non-detrended index')
axs.plot(TPa.time,TPindex,color='grey',label='detrended index')
axs.set(xlabel="time",
       ylabel="[mm/month]",
       title="Indian Monsoon index")
plt.legend()
plt.grid()


## Power spectrum
L = len(TPindex)  #length of the signal in months
Lyr = L/12; #length of the signal in years

# Compute Power Spectra
LPS = int((L/2))
prds_yrs = Lyr/np.arange(1,LPS) #it's the x-axis
sfft_all = np.fft.fft(TPindex)
sfft = sfft_all[1:LPS]

# Normalize to get the actual signal amplitude
sfft_amp = 2*sfft/L

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10), constrained_layout = True)
ax.plot(prds_yrs,np.abs(sfft_amp), '.-')
ax.title.set_text('s_fft norm. to get amplitudes in '+dse['tp'].units)
ax.set_ylabel(dse['tp'].units)
ax.set_xlabel('time in years')
ax.set_xlim(0,23)
ax.grid()

#the moving average is used to smooth out short-term fluctuations and highlight longer-term trends
df_psp = pd.DataFrame(abs(sfft_amp))
psp_mavg=df_psp.rolling(7, center=True).mean()
ax.plot(prds_yrs,psp_mavg,'r')


## EOF
#RAW DATA EOF AND PC

#Prepare empty arrays
TP_raw = dse['tp'].values
TP_raw = np.moveaxis(TP_raw, [0, 2], [2, 0]) #Permute dimensions 0 and 2.
I,J,T = TP_raw.shape #Get dataset dimensions
eofs_raw = np.empty((J,I,T))
pcs_raw = np.empty((T,T))

norm_cos_lat = 1
if norm_cos_lat: #points at the equator have different weights than at the poles
  [LON,LAT] = np.meshgrid(dse.lon,dse.lat)
  Wyx = np.cos(np.deg2rad(LAT)) #the weight depends on the latitude
  Wxy = np.moveaxis(Wyx, [0, 1], [1, 0])
  Wxyt = np.tile(Wxy[:,:,np.newaxis], [1,1,T]) #np.tile constructs an array by repeating Wxy the number of times given by reps
                                               #np.newaxis adds an axis
  dnew = (Wxyt*TP_raw).reshape((I*J,T))
else:
  dnew = TP_raw.reshape(I*J,T)

dd = np.matmul(dnew.T,dnew)
# Extract eigenvalues and eigenvectors
[lambdas,A] = np.linalg.eig(dd)

E = np.matmul(dnew,A)#Project data onto eigenvectors
varexp_raw = 100*lambdas/lambdas.sum() #Compute explained variance

# renormalize to obtain PCs in STD units and EOFs in regression units [mm/std]
nmodes = T
for i in range(T):
 m1 = np.empty((I*J))*np.nan
 m1 = E[:,i]
 mstd = np.std(A[:,i])
 eofs_raw[:,:,i] = m1.reshape(I,J).T*mstd
 pcs_raw[:,i] = A[:,i]/mstd


fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(3, 2)

for i in range(3):
  data = eofs_raw[:,:,i]
  # Add the cyclic point
  data,lons = add_cyclic_point(data,coord=dse['lon'])
  ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree(central_longitude=100))
  cs = ax1.pcolor(lons,dse['lat'],data, vmin=-70, vmax=70,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='coolwarm')

  ax1.coastlines(resolution='auto', color='k')
  ax1.set_title('EOF'+str(i+1)+' [mm/std]')

  ax2 = fig.add_subplot(gs[i, 1])
  ax2.plot(dse['time'],pcs_raw[:,i])
  ax2.set_ylabel('PC'+str(i+1)+' [Norm.]')
  ax2.set_xlabel('time')

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.125, right=0.9,
                    wspace=0.02, hspace=0.25)
# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.1, 0.25, 0.02])
# Draw the colorbar
cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')

for i in range(3):
  print('Percentage of variance explained by mode ' + str(i+1)+ ': ' +
        str(varexp_raw[i]))


#zoom in on India for the first mode to show the climate change trend
fig = plt.figure(figsize=(5,5))
gs = fig.add_gridspec(1)
extent = [65,100,5,30] #zoom in on India

for i in range(1):
  data = eofs_raw[:,:,i]
  # Add the cyclic point
  data,lons = add_cyclic_point(data,coord=dse['lon'])
  ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree(central_longitude=100))
  cs = ax1.pcolor(lons,dse['lat'],data, vmin=-70, vmax=70,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='coolwarm')

  ax1.coastlines(resolution='auto', color='k')
  ax1.set_title('EOF'+str(i+1)+' [mm/std]')
  ax1.set_extent(extent)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.12, 0.1, 0.75, 0.02])
# Draw the colorbar
cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')


#MONTHLY MEAN ANOMALY EOF AND PC DETRENDED

#Prepare empty arrays
TP_anomd, TP_climd = anom(dse['tp'])
TP_anomd = TP_anomd.values
TP_anomd = np.moveaxis(TP_anomd, [0, 2], [2, 0]) #Permute dimensions 0 and 2.
I,J,T = TP_anomd.shape #Get dataset dimensions
eofs_anomd = np.empty((J,I,T))
pcs_anomd = np.empty((T,T))

norm_cos_lat = 1
if norm_cos_lat: #points at the equator have different weights than at the poles
  [LON,LAT] = np.meshgrid(dse.lon,dse.lat)
  Wyx = np.cos(np.deg2rad(LAT))  #the weight depends on the latitude
  Wxy = np.moveaxis(Wyx, [0, 1], [1, 0])
  Wxyt = np.tile(Wxy[:,:,np.newaxis], [1,1,T])
  dnew = (Wxyt*TP_anomd).reshape((I*J,T)) #np.tile constructs an array by repeating Wxy the number of times given by reps
                                          #np.newaxis adds an axis
else:
  dnew = TP_anomd.reshape(I*J,T)

dnew=signal.detrend(dnew,1) #detrend

dd = np.matmul(dnew.T,dnew)
# Extract eigenvalues and eigenvectors
[lambdas,A] = np.linalg.eig(dd)

E = np.matmul(dnew,A)#Project data onto eigenvectors
varexp_anomd = 100*lambdas/lambdas.sum() #Compute explained variance

# renormalize to obtain PCs in STD units and EOFs in regression units [mm/std]
nmodes = T
for i in range(T):
 m1 = np.empty((I*J))*np.nan
 m1 = E[:,i]
 mstd = np.std(A[:,i])
 eofs_anomd[:,:,i] = m1.reshape(I,J).T*mstd
 pcs_anomd[:,i] = A[:,i]/mstd


fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(3, 2)

for i in range(3):
  data = eofs_anomd[:,:,i]
  # Add the cyclic point
  data,lons = add_cyclic_point(data,coord=dse['lon'])
  ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree(central_longitude=100))
  cs = ax1.pcolor(lons,dse['lat'],data, vmin=-40,vmax=40,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='coolwarm')

  ax1.coastlines(resolution='auto', color='k')
  ax1.set_title('EOF'+str(i+1)+' [mm/std]')

  ax2 = fig.add_subplot(gs[i, 1])
  ax2.plot(dse['time'],pcs_anomd[:,i])
  ax2.set_ylabel('PC'+str(i+1)+' [Norm.]')
  ax2.set_xlabel('time')

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.125, right=0.9,
                    wspace=0.02, hspace=0.25)
# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.1, 0.25, 0.02])
# Draw the colorbar
cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')

for i in range(3):
  print('Percentage of variance explained by mode ' + str(i+1)+ ': ' +
        str(varexp_anomd[i]))


# Correlation and regression
## Compare monsoon tp index with the west African monsoon
#Compare monsoon tp index with the tp anomaly over west Africa
TP_a, TP_clm = anomd(dse['tp'])
TPidx=TP_a.sel(lon=slice(-120,60),lat=slice(-10,30)).mean(dim=('lat', 'lon')).compute()

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(16,6), constrained_layout = True)
axs.plot(time,TPindex,color='red')
axs.set_ylabel("tp anomaly Indian monsoon [mm/month]",color="red",fontsize=14)
axs.set_xlabel("Year",fontsize=14)

ax2=axs.twinx()
ax2.plot(time,TPidx,color='blue')
ax2.set_ylabel("tp anomaly West African monsoon [mm/month]",color="blue",fontsize=14)
ax2.set_title("indian monsoon index VS tp West African monsoon  R="
              +str(np.round(np.corrcoef(TPidx,TPindex)[0,1],2)),fontsize=14)

plt.show()


## Compare monsoon tp index with south African mosnoon
#Compare monsoon tp index with the tp anomaly over south Africa
TPidx=TP_a.sel(lon=slice(-100,-40),lat=slice(-40,-20)).mean(dim=('lat', 'lon')).compute()

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(16,6), constrained_layout = True)
axs.plot(time,TPindex,color='red')
axs.set_ylabel("tp anomaly Indian monsoon [mm/month]",color="red",fontsize=14)
axs.set_xlabel("Year",fontsize=14)

ax2=axs.twinx()
ax2.plot(time,TPidx,color='blue')
ax2.set_ylabel("tp anomaly South African monsoon [mm/month]",color="blue",fontsize=14)
ax2.set_title("indian monsoon index VS tp South African monsoon  R="
              +str(np.round(np.corrcoef(TPidx,TPindex)[0,1],2)),fontsize=14)

plt.show()


#Compare monsoon tp index with the tp anomaly: correlation and regression map
fig, axs = plt.subplots(nrows=1,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(25,30))
FLDa, FLDa_clm = anomd(dse['tp'])
FLDcTPad = npcorrelation(FLDa.values, TPindex.values)
FLDrTPad = npregress(FLDa.values, TPindex.values)
# data to plot
data=FLDcTPad
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse['lon'])
cs = axs[0].contourf(lons,dse['lat'],data,
                      transform = ccrs.PlateCarree(central_longitude=100),
                      cmap='bwr',extend='both',levels=np.arange(-1,1.1,0.1))
axs[0].coastlines(resolution='auto', color='k')
axs[0].set_title('Correlation tp indian monsoon index')
gls = axs[0].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
gls.top_labels=False
gls.right_labels=False
axs[0].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
fig.colorbar(cs,orientation='horizontal',pad=0.04)

# data to plot
data=FLDrTPad
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse['lon'])
cs = axs[1].contourf(lons,dse['lat'],data,
                      transform = ccrs.PlateCarree(central_longitude=100),
                      cmap='bwr',extend='both',levels=np.arange(-50,55,5))
axs[1].coastlines(resolution='auto', color='k')
axs[1].set_title('Regression tp indian monsoon index ['+dse['tp'].units+'/STDindex]')
gls = axs[1].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
gls.top_labels=False
gls.right_labels=False
axs[1].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
fig.colorbar(cs,orientation='horizontal',pad=0.04)

fig.tight_layout()


## Explore Indian monsoons teleconnection using correlation maps
# Plot all maps at once for ERA5 fields
var_levs={'u10':np.arange(-3,3.2,0.2),'v10':np.arange(-3,3.2,0.2),
           't2m':np.arange(-1,1.1,0.1),'msl':np.arange(-100,110,10),
            'sst':np.arange(-1,1.1,0.1),
          'sossheig':np.arange(-0.2,.22,0.02),'so20chgt':np.arange(-20,21,1),
           'sozotaux':np.arange(-.05,.06,.01)}

fig, axs = plt.subplots(nrows=5,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(23,23))
k=-1
for var in dse.keys():
  if var!='tp':
    k=k+1
    FLDa, FLDa_clm = anomd(dse[var])
    FLDcTPad = npcorrelation(FLDa.values, TPindex.values)
    FLDrTPad = npregress(FLDa.values, TPindex.values,)
    # data to plot
    data=FLDcTPad
    # Add the cyclic point
    data,lons=add_cyclic_point(data,coord=dse['lon'])
    cs = axs[k,0].contourf(lons,dse['lat'],data,
                      transform = ccrs.PlateCarree(central_longitude=100),
                      cmap='bwr',extend='both',levels=np.arange(-1,1.1,0.1))
    axs[k,0].coastlines(resolution='auto', color='k')
    axs[k,0].set_title('Correlation '+var+' indian monsoon index',fontsize=20)
    gls = axs[k,0].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    gls.top_labels=False
    gls.right_labels=False
    axs[k,0].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
    fig.colorbar(cs)

    # data to plot
    data=FLDrTPad
    # Add the cyclic point
    data,lons=add_cyclic_point(data,coord=dse['lon'])
    cs = axs[k,1].contourf(lons,dse['lat'],data,
                      transform = ccrs.PlateCarree(central_longitude=100),
                      cmap='bwr',extend='both',levels=var_levs[var])
    axs[k,1].coastlines(resolution='auto', color='k')
    axs[k,1].set_title('Regression '+var+' indian monsoon index ['+dse[var].units+'/STDindex]',fontsize=20)
    gls = axs[k,1].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    gls.top_labels=False
    gls.right_labels=False
    axs[k,1].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
    fig.colorbar(cs)

print('maps for ERA5 dataset')


# Plot all correlation maps at once for ORAS5 fields
fig, axs = plt.subplots(nrows=3,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(30,30))
k=-1
for var in dso.keys():
  k=k+1
  FLDa, FLDa_clm = anomd(dso[var])
  FLDcTPad = npcorrelation(FLDa.values, TPindex.values)
  FLDrTPad = npregress(FLDa.values, TPindex.values)

  # data to plot
  data=FLDcTPad
  # Add the cyclic point
  data,lons=add_cyclic_point(data,coord=dso['lon'])
  cs = axs[k,0].contourf(lons,dso['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='bwr',extend='both',levels=np.arange(-1,1.1,0.1))
  axs[k,0].coastlines(resolution='auto', color='k')
  axs[k,0].set_title('Correlation '+var+' indian monsoon index')
  gls = axs[k,0].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
  gls.top_labels=False
  gls.right_labels=False
  axs[k,0].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
  fig.colorbar(cs)

  # data to plot
  data=FLDrTPad
  # Add the cyclic point
  data,lons=add_cyclic_point(data,coord=dso['lon'])
  cs = axs[k,1].contourf(lons,dso['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='bwr',extend='both',levels=var_levs[var])
  axs[k,1].coastlines(resolution='auto', color='k')
  axs[k,1].set_title('Regression '+var+' indian monsoon index ['+dso[var].units+'/STDindex]')
  gls = axs[k,1].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
  gls.top_labels=False
  gls.right_labels=False
  axs[k,1].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
  fig.colorbar(cs)


fig.tight_layout()


## Explore indian monsoon teleconnection using lagged correlation maps
### Lag-regression TP: evolution of total precipitation
var_levs={'u10':np.arange(-3,3.2,0.2),'v10':np.arange(-3,3.2,0.2),
           't2m':np.arange(-1,1.1,0.1),'msl':np.arange(-100,110,10),
            'sst':np.arange(-1,1.1,0.1),'tp':np.arange(-50,55,5),
          'sossheig':np.arange(-0.2,.22,0.02),'so20chgt':np.arange(-20,21,1),
           'sozotaux':np.arange(-.05,.06,.01)}
var='tp'
FLDa, FLDa_clm = anomd(dse[var])
lagmin = -6
lagmax = 8
lags = range(lagmin, lagmax + 2,2)

lagreg = nplagreg(FLDa.values, TPindex.values, lags)

fig, axs = plt.subplots(nrows=4,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(23,23))
axs=axs.flatten(order='F')
k=-1
for n in range(len(lags)):
  k=k+1
  # data to plot
  data=lagreg[n,:,:]
  # Add the cyclic point
  data,lons=add_cyclic_point(data,coord=dse['lon'])
  cs = axs[k].contourf(lons,dse['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='bwr',extend='both',levels=var_levs[var])
  axs[k].coastlines(resolution='auto', color='k')
  axs[k].set_title('Lag regression '+var+'-indian monsoon index'+' lag='+str(lags[n])+' months',fontsize=20)
  gls = axs[k].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
  gls.top_labels=False
  gls.right_labels=False
  axs[k].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
  fig.colorbar(cs)

print('lag-regression tp')


### Hovmoller Diagram
FLDa, FLDa_clm = anomd(dse['tp'])
data = FLDa.sel(time=slice('1983-05-01','1984-02-01'),lat=slice(17, 30))

# Compute weights and take weighted average over latitude dimension
weights = np.cos(np.deg2rad(data.lat.values))
avg_data = (data * weights[None, :, None]).sum(dim='lat') / np.sum(weights)

# Get times and make array of datetime objects
vtimes = data.time.values.astype('datetime64[M]').astype('O')

# Specify longitude values for chosen domain
lons = data.lon.values

# Start figure
fig = plt.figure(figsize=(10,10))

# Use gridspec to help size elements of plot; small top plot and big bottom plot
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 2], hspace=0.06)

# Tick labels
x_tick_labels = [ u'0\N{DEGREE SIGN}E',
                 u'90\N{DEGREE SIGN}E', u'180\N{DEGREE SIGN}W',u'80\N{DEGREE SIGN}W']

# Top plot for geographic reference (makes small map)
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=100))
ax1.set_extent([0, 357.5, 0, 50], ccrs.PlateCarree(central_longitude=100))
ax1.set_yticks([17,30])
ax1.set_yticklabels([u'17\N{DEGREE SIGN}N', u'30\N{DEGREE SIGN}N'])
ax1.set_xticks([-80, 0, 90, 180])
ax1.set_xticklabels(x_tick_labels)
ax1.grid(linestyle='dotted', linewidth=2)

# Add geopolitical boundaries for map reference
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

# Set some titles
plt.title('Hovmoller Diagram', loc='left')

# Bottom plot for Hovmoller diagram
ax2 = fig.add_subplot(gs[1, 0])
ax2.invert_yaxis()  # Reverse the time order to do oldest first

# Plot of chosen variable averaged over latitude and slightly smoothed
clevs = np.arange(-150,150,5)
cf = ax2.contourf(lons, vtimes,avg_data, levels=clevs, cmap=plt.cm.bwr, extend='both')
cs = ax2.contour(lons, vtimes, avg_data,levels=clevs, colors='k', linewidths=1)
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.08, aspect=50, extendrect=True)
cbar.set_label('mm/month')

# Make some ticks and tick labels
ax2.set_xticks([-80, 0, 90, 180])
ax2.set_xticklabels(x_tick_labels)
ax2.set_yticks(vtimes)
ax2.set_yticklabels(vtimes)

print('Hovmoller Diagram')


### Lag-regression msl: impact on sea level pressure
var='msl'
FLDa, FLDa_clm = anomd(dse[var])
lagmin = -6
lagmax = 8
lags = range(lagmin, lagmax + 2,2)

lagreg = nplagreg(FLDa.values, TPindex.values, lags)

fig, axs = plt.subplots(nrows=4,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(23,23))
axs=axs.flatten(order='F')
k=-1
for n in range(len(lags)):
  k=k+1
  # data to plot
  data=lagreg[n,:,:]
  # Add the cyclic point
  data,lons=add_cyclic_point(data,coord=dse['lon'])
  cs = axs[k].contourf(lons,dse['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='bwr',extend='both',levels=var_levs[var])
  axs[k].coastlines(resolution='auto', color='k')
  axs[k].set_title('Lag regression '+var+' indian monsoon index'+' lag='+str(lags[n])+' months',fontsize=20)
  gls = axs[k].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
  gls.top_labels=False
  gls.right_labels=False
  axs[k].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
  fig.colorbar(cs)
print('lag regression for msl')


# Calculate a map of maximum regression coeffient for a range of lags
# Calculate a map showing in each point the lag for which the regression is maximized
##------- Calculations
lagreg_abs=np.absolute(lagreg)
lagreg_absmax=lagreg_abs.max(axis=0)
lagreg_lag_absmax=lagreg_abs.argmax(axis=0)-lags.index(0) # Calculate a map showing in each point the lag for which the regression is maximized
N,I,J=lagreg.shape
lagreg_maxmin=np.empty_like(lagreg_absmax)
for i in range(I):
  for j in range(J):
    lagreg_maxmin[i,j]=lagreg[lagreg_lag_absmax[i,j],i,j] # Calculate a map of maximum regression coeffient for a range of lags


mask=copy.copy(lagreg_maxmin)
mask[np.isfinite(mask)]=1

extent = [-80,180,-80, 80]

##------- Plotting
fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100)},
                        figsize=(10,15))
# data to plot
data=lagreg_lag_absmax
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse['lon'])
mask,lons=add_cyclic_point(mask,coord=dse['lon'])

cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)

cs = axs.pcolor(lons,dse['lat'],data*mask,
                  transform = ccrs.PlateCarree(central_longitude=100),
                  cmap=cmap,vmin=np.min(data) - 0.5,
                      vmax=np.max(data) + 0.5)

axs.set_extent(extent)
axs.coastlines(resolution='auto', color='k')
axs.set_title('Lag of max regression in months for '+str(var))
axs.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
axs.add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
fig.colorbar(cs,ticks=np.arange(np.min(data), np.max(data) + 1),orientation='horizontal')

fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100)},
                        figsize=(10,15))
# data to plot
data=lagreg_maxmin
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dse['lon'])
cs = axs.contourf(lons,dse['lat'],data,
                  transform = ccrs.PlateCarree(central_longitude=100),
                  cmap='bwr',extend='both',levels=var_levs[var])


axs.coastlines(resolution='auto', color='k')
axs.set_title('Max lag-regression ' +var+'-indian mosnoon index ['+dse[var].units+'/STDindex]')
axs.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
axs.add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
fig.colorbar(cs,orientation='horizontal')


### Lag-regression t2m: impact on surface temperature
var='t2m'
FLDa, FLDa_clm = anomd(dse[var])
lagmin = -6
lagmax = 3
lags = range(lagmin, lagmax + 1,1)

lagreg = nplagreg(FLDa.values, TPindex.values, lags)

fig, axs = plt.subplots(nrows=5,ncols=2,sharex=True,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100),'sharex': True},
                        figsize=(23,23))
axs=axs.flatten(order='F')
k=-1
for n in range(len(lags)):
  k=k+1
  # data to plot
  data=lagreg[n,:,:]
  # Add the cyclic point
  data,lons=add_cyclic_point(data,coord=dse['lon'])
  cs = axs[k].contourf(lons,dso['lat'],data,
                    transform = ccrs.PlateCarree(central_longitude=100),
                    cmap='bwr',extend='both',levels=var_levs[var])
  axs[k].coastlines(resolution='auto', color='k')
  axs[k].set_title('Lag regression '+var+'-indian monsoon index'+' lag='+str(lags[n])+' months',fontsize=20)
  gls = axs[k].gridlines(color='lightgrey', linestyle='-', draw_labels=True)
  gls.top_labels=False
  gls.right_labels=False
  axs[k].add_feature(cfeature.LAND, facecolor=(0.8,0.8,0.8))
  fig.colorbar(cs)
print('lag regression t2m')


## Explore indian monsoon teleconnection using composite analysis
TPadstd=TPindex.std().values
ihigh = TPindex.values > TPadstd*1.5
ilow = TPindex.values < -TPadstd*1.5


fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10,6), constrained_layout = True)
axs.plot(time,TPindex,color='k',label='indian monsoon')
axs.plot(time[ilow],TPindex[ilow],'bo',label='weak monsoon')
axs.plot(time[ihigh],TPindex[ihigh],'ro',label='intense monsoon')

axs.set(xlabel="time",
       ylabel="tp [mm/month]",
       title="Indian monsoon index")
plt.legend(fontsize='xx-small')
plt.grid()


#investigating the relation between strong/weak monsoons and la Nina/El Nino
SSTa, SST_clm = anomd(dse['sst'])
N34=SSTa.sel(lat=slice(-6, 6), lon=slice(80,140)).mean(dim=('lat', 'lon')).compute()
N34std=N34.std().values
iNino = N34.values > N34std*1.5
iNina = N34.values < -N34std*1.5

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10,6), constrained_layout = True)
axs.plot(time,N34,color='k',label='N34')
axs.plot(time[iNino],N34[iNino],'ro',label='Niño')
axs.plot(time[iNina],N34[iNina],'bo',label='Niña')

axs.set(xlabel="time",
       ylabel="Temperature [°C]",
       title="Nino3.4 index")
plt.legend(fontsize='xx-small')
plt.grid()



for i in range(len(time[iNino])):
  if time[iNino][i].values in time[ilow]:
    year=str(time[iNino][i].values)
    print(year[0:4]+' was a strong El Nino year and a weak indian monsoon year' )

print('')
for i in range(len(time[iNina])):
  if time[iNina][i].values in time[ihigh]:
    year=str(time[iNina][i].values)
    print(year[0:4]+' was a strong La Nina year and a strong indian monsoon year' )



#investigating the relation between stron/weak monsoons and
box1a=SSTa.sel(lat=slice(0,10), lon=slice(10,20)).mean(dim=('lat', 'lon')).compute() #anomaly of the eastern indian ocean
box2a=SSTa.sel(lat=slice(-10,10), lon=slice(-50,-30)).mean(dim=('lat', 'lon')).compute() #anomaly of the western indian ocean
IOD=box2a-box1a
IODstd=IOD.std().values
highIOD = IOD.values > IODstd*1.5
lowIOD = IOD.values < -IODstd*1.5

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10,6), constrained_layout = True)
axs.plot(IOD.time,IOD,color='k',label='IOD')
axs.plot(IOD.time[highIOD],IOD[highIOD],'ro',label='strong positive phase')
axs.plot(IOD.time[lowIOD],IOD[lowIOD],'bo',label='strong negative phase')

axs.set(xlabel="time",
       ylabel="Temperature [°C]",
       title="IOD index")
plt.legend(fontsize='xx-small')
plt.grid()



for i in range(len(time[highIOD])):
  if time[highIOD][i].values in time[ihigh]:
    year=str(time[highIOD][i].values)
    print(year[0:4]+' was a strongly positive IOD year and a strong indian monsoon year' )

print('')
for i in range(len(time[lowIOD])):
  if time[lowIOD][i].values in time[ilow]:
    year=str(time[lowIOD][i].values)
    print(year[0:4]+' was a strongly negative IOD year and a weak indian monsoon year' )


var='sst'
FLDa, FLDa_clm = anomd(dse[var])
comp_low=FLDa[ilow].mean(axis=0)
comp_high=FLDa[ihigh].mean(axis=0)


levs=np.arange(-1,1.1,0.1)

fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100)},
                        figsize=(9,6))
# data to plot
data=comp_low
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dso['lon'])
cs = axs.contourf(lons,dso['lat'],data,
                  transform = ccrs.PlateCarree(central_longitude=100),
                  cmap='bwr',extend='both',levels=levs)

axs.set_extent(extent)
axs.coastlines(resolution='auto', color='k')
axs.set_title('weak indian monsoon: ' +var+' ['+dse[var].units+']')
axs.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
fig.colorbar(cs,orientation='horizontal')

fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=100)},
                        figsize=(9,6))
# data to plot
data=comp_high
# Add the cyclic point
data,lons=add_cyclic_point(data,coord=dso['lon'])
cs = axs.contourf(lons,dso['lat'],data,
                  transform = ccrs.PlateCarree(central_longitude=100),
                  cmap='bwr',extend='both',levels=levs)

axs.set_extent(extent)
axs.coastlines(resolution='auto', color='k')
axs.set_title('strong indian monsoon: ' +var+' ['+dse[var].units+']')
axs.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
fig.colorbar(cs,orientation='horizontal')



















