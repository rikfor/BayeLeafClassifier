# BayeLeafClassifier (BLC)

This is a repository for BayeLeafClassifier (BLC), as featured in our publication "Annotation of single cells using Earth’s mover distance-based classification" *insert link*. 
BLC is a cell-by-cell classifier, using Earth's Mover Distance (EMD) to find suitable marker genes followed by a naive Bayesian classifier incorporated with a Random Forest classifier. 
As shown in our publication, BLC offers improved accuracy and interpretability in cell type classification, is faster compared to benchmarked methods and provides reliable certainty-levels (how certain BLC is of its annotation) for each cell.

## Found in this repository

- **Tutorials for Training and Testing BLC**: Scripts to train BLC from scratch and evaluate its performance using your data is available in the root directory. Tutorials for classify data in *BLC_classify_data.ipynb* also includes how you can look up marker genes found by BLC.
- **Figure Generation**: Code to reproduce all figures in our publication can be found in *./publication_figures*
- **Pre-trained Classifier**: The trained classifier used in our publication is available at https://drive.google.com/drive/u/0/folders/1--eGmL9U-LCvs_G_EdhPEQDKIjZbvWuI

