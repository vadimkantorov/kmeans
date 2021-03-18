# WIP

Will host PyTorch implementations of K-Means initialization schemes:
- k-means++ - based on https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py#L144
- k-means|| - based on  https://github.com/zxytim/kmeansII/blob/master/src/kmeansII.cc, https://github.com/jtappler/ScalableKMeansPlusPlus and https://nbviewer.jupyter.org/github/jtappler/ScalableKMeansPlusPlus/blob/master/ScalableKMeansPlusPlus.ipynb
- kmc2 - based on https://github.com/obachem/kmc2/blob/master/kmc2.pyx

# References
- https://stats.stackexchange.com/questions/135656/k-means-a-k-a-scalable-k-means
- k-means++: The Advantages of Careful Seeding, https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- Scalable K-Means++, Bahmani et al, http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf
- Fast and Provably Good Seedings for k-Means, Bachem et al, https://papers.nips.cc/paper/2016/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf
