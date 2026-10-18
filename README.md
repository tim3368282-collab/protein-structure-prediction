# protein-structure-prediction
protein-structure-prediction
Protein Structure Prediction with Deep Learning
使用深度學習進行蛋白質結構預測

Project Type: AI / Deep Learning / Bioinformatics
Environment: Kaggle Notebook
Framework: Python, TensorFlow / Keras
Base Project: MiniFold
Status: Work in Progress（持續開發中）

1. Project Overview｜專案簡介
中文介紹

本專案以開源 MiniFold 專案為基礎，嘗試在 Kaggle Notebook 環境中建立蛋白質殘基間距離預測模型的資料處理與訓練流程。

蛋白質的三維結構與其生物功能密切相關。透過深度學習分析蛋白質序列、演化資訊及已知結構資料，可以訓練模型預測不同胺基酸殘基之間的距離關係，作為後續蛋白質結構分析與重建的參考。

本專案著重於理解既有深度學習模型的架構、資料處理方式、訓練流程，以及在實際執行環境中進行程式相容性調整與實驗驗證。

目前專案仍處於實作與驗證階段，尚未宣稱完成模型效能驗證或蛋白質三維結構重建。

English Description

This project explores protein residue distance prediction using deep learning, based on the open-source MiniFold project.

The implementation is conducted in a Kaggle Notebook environment. The work focuses on understanding the model architecture, inspecting protein data, adapting legacy code for compatibility with the execution environment, and establishing a model training workflow.

The current implementation is a work in progress. Model performance evaluation and three-dimensional structure reconstruction have not yet been established as completed results.

2. Project Objectives｜專案目標
Understand the fundamentals of deep learning for protein structure prediction.
Inspect and preprocess protein sequence and structural data.
Adapt an existing Keras-based model to a modern execution environment.
Validate input and target tensor shapes before training.
Establish a reproducible training workflow.
Evaluate model performance and document limitations in future iterations.

中文目標：

理解深度學習在蛋白質結構預測中的應用。
檢查與整理蛋白質序列及結構資料。
調整既有 Keras 模型程式，以改善環境相容性。
驗證模型輸入與標籤的張量形狀。
建立可重現的訓練流程。
在後續實驗中評估模型效能並記錄限制。
3. Methodology｜研究方法
Data Processing｜資料處理

The project uses protein records containing sequence information, evolutionary features, and structural information.

目前已進行的資料處理工作包括：

讀取 ProteinNet 格式的蛋白質紀錄。
依據胺基酸序列長度篩選小於 200 個 residue 的蛋白質。
檢查資料中的 [ID]、[PRIMARY]、[EVOLUTIONARY]、[TERTIARY] 與 [DIST] 等欄位。
使用 MiniFold 的資料生成器建立模型訓練所需的輸入與目標張量。

Data note: The relationship between the initially filtered dataset and the final
