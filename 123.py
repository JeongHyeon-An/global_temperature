# ## Global Temperature Change
#
# # import libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scipy.interpolate as interp
# import gzip
# import pickle as pkl
#
# # Load Data
# file = gzip.GzipFile('GlobalTemperatureData.pkl.gz','rb') # 파일가져오기 + binary read
# df = pkl.load(file) # 데이터프레임 = 파일로 설정하기
# file.close()
#
# print(df.keys()) # 연도별로 세계 온도가 들어가있음을 알 수 있다.
#
# YR = list(df.keys()) # 연도를 리스트로 만들기
# print(YR)
#
# iyr = 0 # 원하는 연도를 불러오기 위함
# df_yr = df[YR[iyr]] # 그 연도에 해당하는 데이터만 불러옴 ---> iyr==0 ==> 1881년
# print(df_yr.keys())
# print(df_yr) # 경도, 위도, 거기에 해당하는 데이터 값을 한눈에 볼 수 있음
#
# data = df_yr[['lon','lat','Temperature(i,j)']].to_numpy() # 내가 원하는 정보만 array로 만듦
# print(data)
#
# data[np.where(data>999)] = np.nan
# print(data) # 9999인 값들은 전부 다 없는 값이라고 지정함
#
#
#
# ## grid
# x = np.linspace(-180, 180, 100)
# y = np.linspace(-90, 90, 100)
# grid_x, grid_y = np.meshgrid( x, y) # 격자 위에 3차원 그래프 그리기
#
# ## Data Interpolate(데이터 손실 최소화) -- 격자 데이터로 맵핑 다시해주기
# # data의 경도와 위도, data의 온도를 가져와서 grid_x, grid_y에 interpolate할 것
# # data[:,[0,1]] == 모든 가로줄 선택, [경도, 위도] 선택
# # interpolate를 어디에다가 할지 --> grid_x, grid_y
# data_interp = interp.griddata(data[:,[0,1]], data[:,2], (grid_x, grid_y), method='linear')
# print(data_interp)
# # ---> nan이 많지만, plot을 통해서 숫자가 들어간 곳을 확인해야함
#
# ## plot
# h_fig, h_ax = plt.subplots()
# im = plt.imread('world_map.png')
# # 이미지를 불러올때, origin='lower'(좌표 측의 중심을 왼쪽 밑으로 내림)하면 이미지가 거꾸로 나옴
# # --> np.flipud를 사용해 다시 뒤집기
# # extent --> x축 == 0~800 // y축 == 0~400 ----> 이미지 사이즈 조정
# plt.imshow(np.flipud(im), origin='lower', extent=(-180,180,-90,90))
# # pcolormesh = 격자 위에 색 입히기 // alpha = 투명도 조절
# plt.pcolormesh(grid_x, grid_y, data_interp, cmap='coolwarm', alpha=0.7)
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.title('Global Temperature Change From YR1880 To ' + str(YR[iyr]))
# plt.ylabel('Longtitude')
# plt.xlabel('Latitude')
# # fraction = 컬러바의 사이즈 조절
# plt.colorbar(fraction=0.022, pad=0.05)
# # colorbar의 최댓값, 최솟값 표시
# plt.clim(-4,4)
# plt.show()




## Global Temperature Change

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interp
import gzip
import pickle as pkl

# Load Data
file = gzip.GzipFile('GlobalTemperatureData.pkl.gz','rb')
df = pkl.load(file)
file.close()

# print(df.keys()) # 연도별로 세계 온도가 들어가있음을 알 수 있다.

YR = list(df.keys()) # 연도를 리스트로 만들기
# print(YR)

iyr = 10 # 원하는 연도를 불러오기 위함
df_yr = df[YR[iyr]] # 그 연도에 해당하는 데이터만 불러옴
# print(df_yr.keys())
# print(df_yr) # 경도, 위도, 거기에 해당하는 데이터 값을 한눈에 볼 수 있음

data = df_yr[['lon','lat','Temperature(i,j)']].to_numpy() # 내가 원하는 정보만 array로 만듦

data[np.where(data>999)] = np.nan
print(data) # 9999인 값들은 전부 다 없는 값이라고 지정함


## grid
x = np.linspace(-180, 180, 100)
y = np.linspace(-90, 90, 100)
grid_x, grid_y = np.meshgrid( x, y) # 격자 위에 3차원 그래프 그리기

## Data Interpolate
# data의 경도와 위도, data의 온도를 가져와서 grid_x, grid_y에 interpolate할 것
data_interp = interp.griddata(data[:,[0,1]], data[:,2], (grid_x, grid_y), method='linear')
# print(data_interp)

# ## plot
# h_fig, h_ax = plt.subplots()
# im = plt.imread('world_map.png')
# # 이미지를 불러올때, origin='lower'하면 이미지가 거꾸로 나옴 --> np.flipud를 사용해 다시 뒤집기
# # extent --> x축 == 0~800 // y축 == 0~400
# plt.imshow(np.flipud(im), origin='lower', extent=(-180,180,-90,90))
# plt.pcolormesh(grid_x, grid_y, data_interp,cmap='coolwarm', alpha=0.8)
# plt.xlim(-180,180)
# plt.ylim(-90,90)
# plt.title('Global Temperature Change From YR1880 To ' + str(YR[iyr]))
# plt.ylabel('Longtitude')
# plt.xlabel('Latitude')
# plt.colorbar(fraction=0.022, pad=0.05)
# plt.clim(-4,4)
# plt.show()


## Animate Graph
import matplotlib.animation

#아래와 같은 세팅이 필요힘 (굳이 알 필요 X)
plt.rcParams["animation.html"] = 'jshtml'
plt.rcParams["figure.dpi"] = 150
plt.ioff()

h_fig, h_ax = plt.subplots()



im = plt.imread('world_map.png')


def animate(iyr=10):
    df_yr = df[YR[iyr]] # 현재 필요한 데이터 뽑아오기
    data = df_yr[['lon','lat','Temperature(i,j)']].to_numpy() # 경도, 위도, 온도만 numpy로 가져옴
    data[np.where(data>999)] = np.nan # 온도 데이터 없는 곳은 nan으로 바꿔줌
    data_interp = interp.griddata(data[:,[0,1]], data[:,2], (grid_x, grid_y), method='linear')

    plt.clf() # clear figure
    plt.cla() # clear axis ---> 둘다 애니메이트를 할 때마다 데이터 새로 써야함

    plt.imshow(np.flipud(im), origin='lower', extent=(-180, 180, -90, 90))
    plt.pcolor(grid_x, grid_y, data_interp, cmap='coolwarm', alpha=0.8, edgecolors=None)

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.title('Global Temperature Change From YR1880 To ' + str(YR[iyr]))
    plt.ylabel('Longtitude')
    plt.xlabel('Latitude')
    plt.colorbar(fraction=0.022, pad=0.05)
    plt.clim(-4, 4)
    # plt.show()

    return data_interp



# animate()
ani = matplotlib.animation.FuncAnimation(h_fig, animate, frames=30)
ani.save("test.gif", fps=30)