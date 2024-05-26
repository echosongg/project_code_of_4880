import xarray as xr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from datetime import datetime

def get_data(file_name, var, date_time):
    # 确定文件路径
    files = os.listdir(file_name)
    da_interp = None

    # 循环遍历所有文件
    for file in files:
        # 提取文件中的年份并进行比较
        pattern = r"agcd_v1_.*_daily_(\d{4})\.nc"
        match = re.search(pattern, file)
        if match:
            year = match.group(1)  # 提取年份
        if int(year) != date_time.year:
            continue

        # 打开数据集
        ds_raw = xr.open_dataset(os.path.join(file_name, file))
        lon = ds_raw["lon"].values
        lat = ds_raw["lat"].values
        a = np.logical_and(lon >= 111.975, lon <= 156.275)
        b = np.logical_and(lat >= -44.525, lat <= -9.975)
        da_selected = ds_raw.isel(time=0)[var]
        da_selected_au = da_selected[b, :][:, a].copy()
        # resize lat & lon
        n = 0.1
        size = (int(da_selected_au.lon.size * n),
                int(da_selected_au.lat.size * n))
        new_lon = np.linspace(
            da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
        new_lon = np.float32(new_lon)
        new_lat = np.linspace(da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
        new_lat = np.float32(new_lat)
        lons, lats = np.meshgrid(new_lon, new_lat)
        # 处理每个时间点的数据
        for i in ds_raw['time'].values:
            current_date = pd.to_datetime(str(i))  # 转换为日期格式
            if current_date.strftime('%Y-%m-%d') == date_time.strftime('%Y-%m-%d'):
                ds_selected_domained = ds_raw.sel(time=i)[var]
                da_selected_au = ds_selected_domained[b, :][:, a].copy()
                temp = cv2.resize(da_selected_au.values, size, interpolation=cv2.INTER_CUBIC)
                temp = np.clip(temp, 0, None)
                da_interp = xr.DataArray(temp, dims=("lat", "lon"), coords={"lat": new_lat, "lon": new_lon, "time": i}, name=var)
                break  # 如果找到匹配日期则停止循环

        ds_raw.close()

        # 如果找到了所需日期的数据，则停止搜索其它文件
        if da_interp is not None:
            break

    return da_interp

variables = {
    'precip': 'precip/total/r005',
    'tmax': 'tmax/mean/r005',
    'tmin': 'tmin/mean/r005'
}
AWAP_dir = "/g/data/zv2/agcd/v1/"
date_time = datetime.strptime("2005-01-12", '%Y-%m-%d')
time_var="01day"
data = {}
for var, path in variables.items():
    filename = f"{AWAP_dir}{path}/{time_var}/"
    data[var] = np.array(get_data(filename, var, date_time))

# 保存数据到当前目录
for var, values in data.items():
    np.save(f"{var}_data.npy", values)
