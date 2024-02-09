import pandas as pd


def FeaturesEncoding(data: pd.DataFrame) -> pd.DataFrame: 
    columns = 'Gender'
    data[columns] = data[columns].replace(
        {'Male': 0, 'Female': 1}
        )
    columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    data[columns] = data[columns].replace(
        {'no': 0, 'yes': 1}
        ) 
    columns = ['CAEC', 'CALC']
    data[columns] = data[columns].replace(
        {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        )
    data = pd.get_dummies(data, columns=['MTRANS'], dtype='int8')
    return data
