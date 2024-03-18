library("scPred")
library("Seurat")
library("magrittr")


expression_str <- '' #file location for data
meta_str <- '' #file loacation for metadata

expression_data <- read.csv(expression_str, row.names = 1)
metadata <- read.csv(meta_str, row.names = 1)
expression_data <- t(expression_data)

#start_time <- Sys.time()

reference <- CreateSeuratObject(counts = expression_data)
rm(expression_data)
reference@meta.data <- metadata
nCount = colSums(x = reference, slot = "counts")  
nFeature = colSums(x = GetAssayData(object = reference, slot = "counts") > 0)  
reference@meta.data$nCount_RNA <- nCount
reference@meta.data$nFeature <- nFeature

reference <- reference %>% 
  NormalizeData() %>% 
  FindVariableFeatures() %>% 
  ScaleData() %>% 
  RunPCA() %>% 
  RunUMAP(dims = 1:30)

reference <- getFeatureSpace(reference, "CellType")

reference <- trainModel(reference, allowParallel = TRUE)

saveRDS(reference, 'insert savepath')

#time_taken_seconds = difftime(Sys.time(), start_time, u = 'secs')

#print(time_taken_seconds)
