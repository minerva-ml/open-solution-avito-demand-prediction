import pandas as pd
from joblib import Parallel, delayed
from googletrans import Translator

translator = Translator()


def translate(filepath, column_to_translate):
    df = pd.read_csv(filepath)

    for column in column_to_translate:
        translated_col = Parallel(n_jobs=8)(delayed(_translate)(i) for i in df[column].unique())
        translated_col = pd.DataFrame(translated_col, columns=[column, '{}_en'.format(column)])
        df = df.merge(translated_col, on=[column], how='left')
        df = df.drop(column, axis=1)
        df = df.rename(index=str, columns={'{}_en'.format(column): column})

    return df


def _translate(x):
    translated = x
    try:
        translated = translator.translate(x, src='ru', dest='en').text
    except Exception as e:
        # print(e)
        pass
    return x, translated
