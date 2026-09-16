# 中青年人群肥胖风险预测系统 (A Prediction System for Chinese Adults' Obesity)

基于中国中青年人群饮食行为与肥胖关系研究训练的随机森林模型，供研究与科普展示使用的静态网页原型。

## 说明

- 纯前端实现（HTML + CSS + 原生 JS），所有推理在浏览器本地完成，不上传任何用户数据。
- 模型：RandomForest（300 棵树，max_depth=10，min_samples_leaf=4），通过嵌套交叉验证网格搜索确定超参数，
  在独立测试集上 AUC ≈ 0.938（95% CI 0.907–0.964）。
- 纳入变量：EFA 单因子结构膳食模式得分、运动习惯、肥胖家族史、EBQ-18 进食行为总分、吸烟状态。
- `model.json` 由训练脚本导出（300 棵树的分裂结构 + 标准化参数），`script.js` 中的推理逻辑已与
  scikit-learn 原始 `predict_proba` 输出逐样本核对一致。

## 本地预览

用任意静态文件服务器打开本目录即可，例如：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000`。

## 免责声明

本工具仅为学术研究衍生的展示原型，预测结果不能替代专业医疗诊断或建议。
