# Data 초기 적용법

1. **기본라이브러리 / 데이타**


<pre>
# 자주쓰는 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns

# csv 파일 불러오기
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
</pre>
