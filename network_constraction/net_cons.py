import os
import random
from datetime import date, timedelta
from torch.utils.data import random_split, DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import maskoceans
import matplotlib
# from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import re
from torch.utils.data.dataloader import default_collate
import pickle
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
            #print(f"文件 '{file}' 中的年份是：{year} and date_time: {date_time.year}")
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
            #print("i",i)
            current_date = pd.to_datetime(str(i))  # 转换为日期格式
            if current_date.strftime('%Y-%m-%d') == date_time.strftime('%Y-%m-%d'):
                ds_selected_domained = ds_raw.sel(time=i)[var]
                da_selected_au = ds_selected_domained[b, :][:, a].copy()
                temp = cv2.resize(da_selected_au.values, size, interpolation=cv2.INTER_CUBIC)
                #mask ocean
                temp = maskoceans(lons, lats, temp)  # 确保maskoceans函数可用
                # 将被掩码的部分设置为 NaN
                #temp[masked_temp.mask] = np.nan
                da_interp = xr.DataArray(temp, dims=("lat", "lon"), coords={"lat": new_lat, "lon": new_lon, "time": i}, name=var)
                break  # 如果找到匹配日期则停止循环

        ds_raw.close()

        # 如果找到了所需日期的数据，则停止搜索其它文件
        if da_interp is not None:
            break

    return da_interp

# 定义数据集
class AWAP(Dataset):
    def __init__(self, dates, start_date, end_date, region="AUS", shuffle=True, transform=None):
        self.file_AWAP_dir = "/g/data/zv2/agcd/v1/"
        self.region = region
        self.start_date = start_date
        self.end_date = end_date
        self.filename_list = dates
        self.transform = transform
        
        # 初始化属性
        self.precip = []
        self.tmax = []
        self.tmin = []
        
        # 读取所有数据并存储在属性中
        self.load_all_data()

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        # 返回指定索引的数据
        return {
            'precip': self.precip[idx],
            'tmax': self.tmax[idx],
            'tmin': self.tmin[idx]
        }

    def read_awap_data(self, date_time, time_var="01day"):
        # 提取多个变量并处理地理约束
        variables = {
            'precip': 'precip/total/r005',
            'tmax': 'tmax/mean/r005',
            'tmin': 'tmin/mean/r005'
        }
        data = {}
        for var, path in variables.items():
            filename = f"{self.file_AWAP_dir}{path}/{time_var}/"
            data[var] = np.array(get_data(filename, var, date_time))
        return data
    
    def load_all_data(self):
        for date_time in self.filename_list:
            data = self.read_awap_data(date_time)
            self.precip.append(data['precip'])
            self.tmax.append(data['tmax'])
            self.tmin.append(data['tmin'])

# 时间范围函数
def date_range(start_date, end_date):
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
# 主程序
START_TIME = date(2005, 1, 1)
END_TIME = date(2008, 12, 31)
transforms = transforms.Compose([
    transforms.ToTensor()
])

dates = date_range(START_TIME, END_TIME)
generator = torch.Generator().manual_seed(42)
train_dates, val_dates = random_split(dates, [int(len(dates) * 1), len(dates) - int(len(dates) * 1)], generator=generator)
train_data = AWAP(train_dates, START_TIME, END_TIME, transform=transforms)
#print("train_data",train_data.__getitem__(0))
#每一批加载18个数据
#train_dataloders = DataLoader(train_data, batch_size=NB_BATCH, shuffle=False, num_workers=NB_THREADS,collate_fn=custom_collate)
#之前的叫train_data_small_Aus.pkl， 没有掩盖海洋
#海洋部分被设置成了Nan读取的时候 Nan就不要了
save_data(train_data, "/scratch/iu60/xs5813/Australian_inland.pkl")