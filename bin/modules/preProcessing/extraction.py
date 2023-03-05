from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis

def dimentionalReduction(dimRed, n, df):
  ctx='<PRE-PROCESSING> Dimentional Reduction'
  dimRedDict = {
    'pca': PCA(n_components=n),
    'svd': TruncatedSVD(n_components=n),
    'factorAnalysis' : FactorAnalysis(n_components=n),
    'FastICA': FastICA(n_components=n)
  }

  numerical_features=[feature for feature in df.columns if df[feature].dtypes!='O']

  dimRed = dimRedDict[dimRed]
  df['DimRedFeature'] = dimRed.fit_transform(df[numerical_features])

  return df
