
"""
# https://sebastianraschka.com/Articles/2014_python_lda.html
"""

"""
PCA: Principal Component Analysis（主成分分析）
Purpose: Dimensional Reduction
Core Idea: Decomposr features into N dimension by Maximum Varience
Type: Unsupervised Learning and Drop your taeget data, doesn't consider class labels
Data Hundling: X
日本語: クラス分類を「無視」して、その目的がデータの分散を最大にする「軸の検索」を目的とする
Key parameter: n_components
Library: from sklearn.decomposition import PCA
Note: we can use PCA for Supervised problem but we should drop target value
"""


"""
LDA: Linear Discriminant Analysis（線形判別分析）
Purpose: Dimensional Reduction
Core Idea: Select n dimension with Hiest Varience
Type: Supervised Learning for Classification problem which has multiple class, pattern-classification
Data Hundling: X + y
日本語: クラス分類を「無視」して、その目的がデータの分散を最大にする「軸の検索」を目的とする
Key parameter: n_components
Library: from sklearn.discriminant_analysis import Linear Discriminant Analysis, Quadratic Discriminant Analysis
Note1: With small sample data, sometimes PCA is better than LDA in Supervised Classification
Note2: Not find lines to split, Just put every class in n lines to express difference between classes
Note3: For simple data distribution
"""

class DimensionalReducsion:

    def PCAdecomposer(X, n_components=3):
        """
        X
        """
        from sklearn.decomposition import PCA
        decomposer = PCA(n_components=n_components)
        pcaX = decomposer.fit_transform(X)
        # after this, we will split X_train, X_test
        return pcaX


    def LDAClassifier(X, y, n_components=3):
        """
        Step 1: Computing the d-dimensional mean vectors
        Step 2: Computing the Scatter Matrices
        Step 3: Solving the generalized eigenvalue problem for the matrix S−1WS
        Step 4: Selecting linear discriminants for the new feature subspace
        Step 5: Transforming the samples onto the new subspace
        X + y
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(n_components=n_components)
        estimator = clf.fit(X, y).transform(X)
        return estimator

    def QDAClassifier(X, y, n_components=3):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis(n_components=n_components)
        estimator = clf.fit(X, y).transform(X)
        return estimator
