---
language: 
- zh
license: "apache-2.0"
---

## Chinese MRC roberta_wwm_ext_large

* 使用大量中文MRC数据训练的roberta_wwm_ext_large模型，详情可查看：https://github.com/basketballandlearn/MRC_Competition_Dureader
* 此库发布的再训练模型，在 阅读理解/分类 等任务上均有大幅提高<br/>
（已有多位小伙伴在Dureader-2021等多个比赛中取得**top5**的成绩😁）

|                模型/数据集                 |  Dureader-2021  |  tencentmedical |
| ------------------------------------------|--------------- | --------------- |
|                                           |    F1-score    |    Accuracy     |
|                                           |  dev / A榜     |     test-1      |
| macbert-large (哈工大预训练语言模型)         | 65.49 / 64.27  |     82.5        |
| roberta-wwm-ext-large (哈工大预训练语言模型) | 65.49 / 64.27  |     82.5        |
| macbert-large (ours)                      | 70.45 / **68.13**|   **83.4**    |
| roberta-wwm-ext-large (ours)              | 68.91 / 66.91   |    83.1        |


