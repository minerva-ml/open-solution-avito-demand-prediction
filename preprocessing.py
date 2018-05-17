import os

import pandas as pd
from joblib import Parallel, delayed
from googletrans import Translator

translator = Translator()

features_to_translate = ['category_name', 'city', 'description', 'param_1', 'param_2', 'param_3',
                         'parent_category_name', 'region', 'title']


def translate(filepath, filepath_en):
    if not os.path.isfile(filepath_en):
        df = pd.read_csv(filepath)

        for feature in features_to_translate:
            translated_col = Parallel(n_jobs=8)(delayed(_translate)(i) for i in df[feature].unique())
            translated_col = pd.DataFrame(translated_col, columns=[feature, '{}_en'.format(feature)])
            df = df.merge(translated_col, on=[feature], how='left')
            df = df.drop(feature, axis=1)
            df = df.rename(index=str, columns={'{}_en'.format(feature): feature})

        df.to_csv(os.path.join(filepath_en))


def _translate(x):
    t = x
    try:
        t = translator.translate(x, src='ru', dest='en').text
    except Exception as e:
        # print(e)
        pass
    return x, t
