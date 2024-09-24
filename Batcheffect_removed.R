library(ggplot2)
library(Rtsne)
library(data.table)
library(ggthemes)
library(MMUPHin)
library(magrittr)
library(dplyr)
library(vegan)

#===================1.处理数据集，这里以处理IBD1和IBD2为例（1-3）============
process_data <- function(data) {
  original_features <- colnames(data)[-1]  # 不包括第一列
  new_features <- sapply(original_features, function(x) {
    split_name <- unlist(strsplit(x, "\\|"))
    s_part <- split_name[grepl("s__", split_name)]
    g_part <- split_name[grepl("g__", split_name)]
    if (length(s_part) > 0) {
      sub("s__", "", s_part)
    } else if (length(g_part) > 0) {
      sub("g__", "", g_part)
    } else {
      x
    }
  })
  
  colnames(data) <- c("Sample", new_features)
  unique_features <- unique(new_features)
  data_combined <- data[, .SD, .SDcols = c("Sample", unique_features)]
  
  for (feature in unique_features) {
    if (sum(new_features == feature) > 1) {
      data_combined[, (feature) := rowSums(.SD, na.rm = TRUE), .SDcols = which(new_features == feature)]
    }
  }
  
  return(data_combined)
}

# =====================2.读取数据=========================
dataset1 <- fread("data/IBD1.csv")
dataset2 <- fread("data/IBD2.csv")

processed_data1 <- process_data(dataset1)
processed_data2 <- process_data(dataset2)

fwrite(processed_data1, "data/IBD1_rename.csv")
fwrite(processed_data2, "data/IBD2_rename.csv")

#===================3.两个数据集的特征求并集然后构建组合数据矩阵================
data1 <- fread("data/IBD1_rename.csv")
data2 <- fread("data/IBD2_rename.csv")

X1 <- as.matrix(data1[, -1, with=FALSE])
X2 <- as.matrix(data2[, -1, with=FALSE])

print(paste("dataset1 样本:", nrow(data1), "特征:", ncol(data1) - 1))
print(paste("dataset2 样本:", nrow(data2), "特征:", ncol(data2) - 1))

all_features <- union(colnames(X1), colnames(X2))
diff_features <- intersect(colnames(X1), colnames(X2))
print(paste("union_features 数量:", length(all_features)))
print(paste("diff_features 数量:", length(diff_features)))

X1_full <- matrix(0, nrow=nrow(X1), ncol=length(all_features))
colnames(X1_full) <- all_features
X1_full[, colnames(X1)] <- X1

X2_full <- matrix(0, nrow=nrow(X2), ncol=length(all_features))
colnames(X2_full) <- all_features
X2_full[, colnames(X2)] <- X2

# 保留样本名
X1_full <- cbind(data1[, 1, with=FALSE], X1_full)
X2_full <- cbind(data2[, 1, with=FALSE], X2_full)

# 确保列名一致
colnames(X1_full) <- c("SampleID", all_features)
colnames(X2_full) <- c("SampleID", all_features)

# 合并数据
combined_data <- rbind(
  data.frame(X1_full),
  data.frame(X2_full)
)

print(paste("combined_data 行数:", nrow(combined_data)))
print(paste("combined_data 列数:", ncol(combined_data)))

rownames(combined_data) <- combined_data$SampleID
combined_data <- combined_data[, -1]
write.csv(combined_data, "data/IBD1_IBD2_combined.csv", row.names = TRUE)

#====================4.再次合并两个已经合并好的数据集，进行批次效应去除====================
data1 <- fread("data/IBD1_IBD2_combined.csv")
data2 <- fread("data/IBD3_IBD4_combined.csv")

X1 <- as.matrix(data1[, -1, with=FALSE])
X2 <- as.matrix(data2[, -1, with=FALSE])

print(paste("dataset1 样本:", nrow(data1), "特征:", ncol(data1) - 1))
print(paste("dataset2 样本:", nrow(data2), "特征:", ncol(data2) - 1))

# 获取特征并集
all_features <- union(colnames(X1), colnames(X2))
diff_features <- intersect(colnames(X1), colnames(X2))
print(paste("union_features 数量:", length(all_features)))
print(paste("diff_features 数量:", length(diff_features)))

# 创建包含所有特征的数据框，并将缺失的特征赋值为0
X1_full <- matrix(0, nrow=nrow(X1), ncol=length(all_features))
colnames(X1_full) <- all_features
X1_full[, colnames(X1)] <- X1

X2_full <- matrix(0, nrow=nrow(X2), ncol=length(all_features))
colnames(X2_full) <- all_features
X2_full[, colnames(X2)] <- X2

# 保留样本名
X1_full <- cbind(data1[, 1, with=FALSE], X1_full)
X2_full <- cbind(data2[, 1, with=FALSE], X2_full)

# 确保列名一致
colnames(X1_full) <- c("SampleID", all_features)
colnames(X2_full) <- c("SampleID", all_features)

# 合并数据
combined_data <- rbind(
  data.frame(X1_full),
  data.frame(X2_full)
)

print(paste("combined_data 行数:", nrow(combined_data)))
print(paste("combined_data 列数:", ncol(combined_data)))

rownames(combined_data) <- combined_data$SampleID
combined_data <- combined_data[, -1]
write.csv(combined_data, "data/IBD_combined_4data.csv", row.names = TRUE)

#============================5.批次效应去除和评估===========================
combin_abundance <- read.csv("data/IBD_combined_4data.csv",header = TRUE,sep = ",",row.names = 1)
combin_metadata <- read.csv("data/IBD_metadata_4data.csv",header = TRUE,sep = ",",row.names = 1)

# 对每行进行归一化
normalized_data <- t(apply(combin_abundance, 1, function(x) x/sum(x)))

# 将归一化后的矩阵转换回数据框
normalized_df <- as.data.frame(t(normalized_data))

# 更新数据框中的列名
colnames(normalized_df) <- rownames(combin_abundance)

fit_adjust_batch <- adjust_batch(feature_abd = normalized_df,
                                 batch = "dataset",
                                 covariates = "label",
                                 data = combin_metadata,
                                 control = list(verbose = FALSE))

abd_adj <- fit_adjust_batch$feature_abd_adj

write.csv(normalized_data,"data/resulttest/IBD_combined_scaled.csv")
write.csv(t(abd_adj),"data/resulttest/IBD_removebe.csv")

# 评估批次效应处理前后
D_before <- vegdist(normalized_data)
D_after <- vegdist(t(abd_adj))
set.seed(1)
fit_adonis_before <- adonis2(D_before ~ dataset, data = combin_metadata)
print(fit_adonis_before)
fit_adonis_after <- adonis2(D_after ~ dataset, data = combin_metadata)
print(fit_adonis_after)
