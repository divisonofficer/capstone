import pandas as pd
import glob

# 파일 목록을 가져옵니다.
file_paths = glob.glob('*.csv')  # 현재 디렉터리의 모든 CSV 파일을 가져옵니다.

# 각 파일을 DataFrame으로 읽고 리스트에 저장합니다.
dataframes = [pd.read_csv(file) for file in file_paths]

# 모든 DataFrame을 하나로 합칩니다.
merged_dataframe = pd.concat(dataframes, ignore_index=True)

# 결과를 새 CSV 파일로 저장합니다.
merged_dataframe.to_csv('merged.csv', index=False)
