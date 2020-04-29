# Data EDA

## 1.**기본라이브러리 / 데이타**   

``` python
# 자주쓰는 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns

# csv 파일 불러오기
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
```   
## 2.**시각화 쉽게하기**   
### 2-1) 먼저 X컬럼을 여러분을 위해 나누어 드립니다. (솔직히 명적줘야 된다.)   
``` python
temperature_name = ["X00","X07","X28","X31","X32"] #기온
localpress_name  = ["X01","X06","X22","X27","X29"] #현지기압
speed_name       = ["X02","X03","X18","X24","X26"] #풍속
water_name       = ["X04","X10","X21","X36","X39"] #일일 누적강수량
press_name       = ["X05","X08","X09","X23","X33"] #해면기압
sun_name         = ["X11","X14","X16","X19","X34"] #일일 누적일사량
humidity_name    = ["X12","X20","X30","X37","X38"] #습도
direction_name   = ["X13","X15","X17","X25","X35"] #풍향
```   
### 2-2) 기상청 데이터 종류별로 그래프룰 확인 하는 코드는 다음과 같습니다.   
```python
train.plot(x='id', y=temperature_name, figsize=(8,3), title="title")
#x = x축 , y=보고싶은 기상청데이터 종류, figsize=(가로,세로), 타이틀은 옵션
```
- y만 water_name으로 바꾸었을 뿐인데 누적강수량을 한눈에 볼 수 있습니다.    


### 2-3) 요소별로 함수를 적용시킬 수도 있습니다.
    - 평균, 표준편차, 중앙값, 최댓값 등등을 행,열별 계산 하실 수 있습니다.    

```python
#컬럼에 대한 기초통계량 확인 (행별 계산)
train[temperature_name].mean()
#요소를 인덱스별로 합쳐서 하나의 컬럼으로 생성!  (열별 계산)
pd.Series(train[temperature_name].mean(axis = 1))
```    
## 3.**표준화 하기**   
- 표준화는 train에서 사용한 평균과 표준편차를 기억하여 나중에 사용하는 test에 적용해야 합니다.
  일단 표준화는 N(0,1) 즉, 평균을 0 표준편차를 1로 만드는 작업입니다.
  학습시 빠르게 학습되는 장점이 있으므로 강력 추천합니다.
  먼저 표준화 함수입니다.

```python
#표준화 함수 생성
def standardization(df):
    mean = np.mean(df)
    std = np.std(df)
    norm = (df - mean) / (std - 1e-07)
    return norm, mean, std
```   
## 4.**상관계수 히트맵 보기**   
- 상관계수를 보기 위해 주로 히트맵을 보시는데 보는 방법은 다음과 같습니다.

### 3-1) 기본 히트맵보는 방법
```python

#Y에 대한 상관계수 데이터프레임 생성
train_corr = train.loc[:,"Y00":"Y18"].corr()
#출력 크기조절
plt.figure(figsize=(10,10))
#히트맵 정의
ax = sns.heatmap(train_corr, cmap = "RdBu", annot = True,vmin=0, vmax=1)
#y축 잘림 방지
ax.set_ylim(len(train_corr.columns),0)
#출력
plt.show()
```  

### 3-2) 특정 컬럼과 상관계수가 높은 컬럼을 찾는 함수는 다음과 같습니다.
```python
#인풋: 기준 컬럼이 속한 데이터프레임, 기준 컬럼, 기준 상관계수
#아웃풋: 기준 컬럼과 상관계수가 기준 상관계수보다 높은 컬럼들의 이름
def high_corr(df, col, ratio):
    #Y에 대한 상관계수 데이터프레임 생성
    Y_corr = df.corr()
    Y_high = Y_corr.loc[:,Y_corr[col]> ratio].columns
    return Y_high.drop(col)

#인풋 생성: Y컬럼들
df = train.loc[:,"Y00":"Y17"]
#아웃풋 생성: Y17과 상관계수가 0.8 이상인 Y컬럼들
Y_high = high_corr(df, "Y17", 0.8)
print("Y17와의 상관계수가 높은 Y컬럼들 ", Y_high.tolist())

#응용
#Y_high = high_corr(df, "Y18", 0.8)

Y17와 상관계수가 낮은 Y컬럼들  ['Y01', 'Y02', 'Y05', 'Y06', 'Y07', 'Y08', 'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16']
```   
## 5.**고장난 센서 버리기**
- 사실 X14, X16, X19 관측값은 고장나서 측정을 하지 못합니다. 그 친구들을 손으로 없애면 손코딩이 되겠죠? 그런 친구를 자동으로 없애는 간단한 코드를 공유합니다.   
```python
#고장난 센서를 그래프로 봅니다.
train.plot(x = "id", y = train.columns[train.max() == train.min()])
plt.show()

#고장나서 0만 출력하는 관측소 데이터를 컬럼에서 삭제해주는 함수
def same_min_max(df):
    return df.drop(df.columns[df.max() == df.min()], axis=1)

train_new = same_min_max(train)
test_new  = same_min_max(test)
#사실 더 쉬운 방법이 있지만, 이런 방법으로 손코딩을 줄일 수도 있다는 것을 보여드리고 싶었습니다.
```   
