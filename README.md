# 那邊有一隻超可愛的狗勾！狗狗分類大挑戰！
## Fine-Grained Visual Classification: Dog Breed Identification

**Kaggle : 2025 Iyatomi Lab. Competition —— 2025 Iyatomi 實驗室課堂影像分類競賽**
🔗 [競賽頁面總覽 (Overview)](https://www.kaggle.com/competitions/2025-iyatomi-lab-competition/overview)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Rank](https://img.shields.io/badge/Kaggle_Rank-2nd_Place-gold)
![Accuracy](https://img.shields.io/badge/Best_Valid_Acc-92.48%25-green)

---

## 專案概述

本專案旨在解決 **細粒度視覺分類 (Fine-Grained Visual Classification, FGVC)** 的挑戰，目標是準確區分 120 種外觀高度相似的犬種（例如：西伯利亞哈士奇與阿拉斯加雪橇犬），不同於一般的貓狗分類，此任務要求模型具備極強的局部特徵提取能力，以辨識眼睛形狀、毛色紋理與口鼻長度等細微差異。

這項專案是為了 **2025 Iyatomi Lab Competition** 所開發，最終在 Kaggle Private Leaderboard 取得 **0.86177**，榮獲第二名的成績。

---

## 核心成效與實驗結果

我們比較了從輕量級 CNN 到 Transformer 的多種架構，實驗證明 **Vision Transformer (ViT)** 在此任務上表現最佳。

| 模型架構 (Model) | 優化策略 (Strategy) | 最佳驗證準確率 (Best Val Acc) | Kaggle Private LB |
| :--- | :--- | :--- | :--- |
| **ViT-b-16** | **預訓練 + 標籤平滑 (Label Smoothing)** | **92.48%** | **0.86177** |
| ResNet-50 | CBAM (注意力機制) + 解凍訓練 | 88.72% | - |
| EfficientNet-b4 | 標籤平滑 (Label Smoothing) | 87.21% | - |
| EfficientNet-b3 | CBAM (注意力機制) | 87.00% | - |

---

## 方法論與策略

### 1. 模型架構創新 (Architecture Optimization)
* **CNN 改良 (CNN + CBAM):** 針對 ResNet 與 EfficientNet 系列，我們手動嵌入了 **CBAM (Convolutional Block Attention Module)** 注意力模組。這強制模型去關注「哪個特徵通道重要 (Channel Attention)」以及「哪個圖片位置重要 (Spatial Attention)」，成功將 ResNet-50 的準確度從 81.95% 大幅提升至 88.72%。
* **Transformer 應用:** 採用 `vit_b_16` 進行遷移學習，將圖像分割為 $16 \times 16$ 的 Patch，利用 Transformer Encoder 處理長距離的特徵依賴。

### 2. 訓練策略 (Training Strategy)
為了克服資料集較小 (約 1200 張) 帶來的過擬合風險，我們實施了嚴格的訓練流程：
* **漸進式解凍 (Progressive Unfreezing):** 初期凍結骨幹網路 (Backbone)，僅訓練分類頭 (Classifier)。
* **分組學習率 (Differential Learning Rates):**
    * Backbone (預訓練層): $1 \times 10^{-5}$ (保留通用特徵)
    * Classifier (新層): $1 \times 10^{-3}$ (快速適應新類別)
* **優化器設定:** 使用 `RAdam` 搭配 `CosineAnnealingLR` 排程器，實現穩定的收斂。

### 3. 資料增強分析 (Data Augmentation Analysis)
* **有效策略:** 隨機裁切 (Random Resized Crop)、水平翻轉 (Horizontal Flip)、正規化 (Normalization)。
* **無效/負面策略:** 實驗發現 **AutoAugment** 與 **TTA (Test Time Augmentation)** 反而導致準確度下降（例如 EfficientNet-b1 下降至 0.83）。推測是因為在小規模數據集上，過度激進的幾何變換破壞了區分品種的關鍵細微特徵。

---

## 可解釋性分析 (Explainability)

利用 **Grad-CAM** 技術，我們視覺化了模型的關注區域，以驗證模型是否學到了正確的特徵。

| 成功案例 (Success) | 失敗/多目標案例 (Failure) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/da2c6af8-0e62-4ec4-a968-76fac1b5a148" width="300"> | <img src="https://github.com/user-attachments/assets/437d525e-2c5b-4b72-a430-b89d7a7243f4" width="300"> |
| **分析:** 模型準確地將注意力集中在犬隻的**頭部與五官**，這是區分品種的最關鍵區域 。 | **分析:** 當畫面中有多隻狗或背景複雜時，模型的注意力有時會分散，甚至錯誤關注到背景草地，導致預測信心下降。 |

---

## 如何執行 (How to Run)

本專案的所有實作邏輯均整合於 Jupyter Notebook 中。

1.  **環境需求 (Requirements):**
    ```bash
    pip install torch torchvision grad-cam pandas opencv-python matplotlib
    ```

2.  **執行訓練:**
    開啟 `mis583_group22.ipynb`，該 Notebook 包含完整的 End-to-End 流程：
    * **資料預處理:** 定義 Transform 與 DataLoader。
    * **模型建構:** 包含 `get_base_model()` 與自定義的 `CNNWithCBAM` 類別。
    * **訓練迴圈:** 包含 RAdam 優化器與驗證機制。
    * **視覺化:** 執行 Grad-CAM 程式碼區塊以生成熱力圖。

---

## 團隊成員 (Contributors)

**國立中山大學 資訊管理學系碩士班**
**Department of Information Management, National Sun Yat-sen University**

* **M144020035 陳彥宇** 
* **M144020054 戴振宇** 
* **M144020056 張祐豪** 

---

## 參考文獻 (References)
1.  Kaggle Competition: [2025 Iyatomi Lab. Competition](https://www.kaggle.com/competitions/2025-iyatomi-lab-competition/overview)
2.  Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
3.  Woo, S., et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
4.  Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
