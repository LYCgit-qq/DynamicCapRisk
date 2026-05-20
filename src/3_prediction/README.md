# 3_prediction 模块说明

本目录负责 MT-RP 及基线模型的预测训练、评估与结果汇总。

## 脚本执行顺序

1. `dataset.py`
   - 作用：构建预测模型数据集（历史 15s → 预测未来 3s）。
   - 工作流程：
     - 加载原始多模态信号数据 (`act`, `eye`, `phy`)。
     - 加载能力波动结果 `cap_pkl`。
     - 加载风险评估结果 `risk_csv`。
     - 提取逐窗口特征并对齐三路信号。
     - 构建序列样本 `X` / `y_ability` / `y_risk_reg` / `y_risk_cls`。
     - 按驾驶员能力分层划分 train/val/test。
     - 选项：数据增强、Z-score 标准化。
     - 输出：`config` 中 `paths.output_pkl` 指定的 dataset pickle 文件。
   - 运行方式：
     - `python dataset.py`
     - `python dataset.py -c config/dataset.yaml`

2. `trainer_dl.py`
   - 作用：训练深度学习风险预测模型，包括 `MT-RP` 及深度基线模型。
   - 支持模型类型：`mtrp`, `lstm`, `gru`, `cnn_lstm`。
   - 工作流程：
     - 读取 dataset pickle。
     - 构建模型并训练。
     - 早停验证并保存 `best_model.pt` / `final_model.pt`。
     - 自动调用 `evaluator.py` 对 `test` 集进行评估。
   - 输出：`output/3_prediction/runs/{timestamp}_{model_type}/`
   - 运行方式：
     - `python trainer_dl.py`
     - `python trainer_dl.py -c config/trainer_dl.yaml`

3. `trainer_svr_cart.py`
   - 作用：训练传统机器学习基线模型 `SVR` 和 `CART`。
   - 工作流程：
     - 读取 dataset pickle，将 `(N, T, D)` 展平为 `(N, T*D)`。
     - 标准化特征。
     - 分别训练 `ability`、`risk_reg`、`risk_cls` 三个任务。
     - 保存模型与评估指标。
   - 输出：`output/3_prediction/runs/{timestamp}_{model_name}/`
   - 运行方式：
     - `python trainer_svr_cart.py --model svr`
     - `python trainer_svr_cart.py --model cart`
     - `python trainer_svr_cart.py -c config/trainer_svr_cart.yaml`

4. `run_grid_search.py`
   - 作用：执行网格搜索并并行训练多个实验。
   - 工作流程：
     - 读取 `config/run_grid_search.yaml`。
     - 生成临时配置文件并调用 `TRAIN_SCRIPT` 指定的训练脚本。
     - 记录任务进度和日志。
   - 适用场景：需要批量调参、自动化训练时使用。
   - 运行方式：
     - `python run_grid_search.py`

5. `evaluator.py`
   - 作用：对单个已训练模型检查点执行风险预测结果评估。
   - 工作流程：
     - 加载模型 checkpoint 和 dataset pickle。
     - 在指定 split 上生成预测结果。
     - 保存评估报告、指标 CSV、预测结果 CSV。
     - 可选生成可视化图表。
   - 运行方式：
     - `python evaluator.py`
     - `python evaluator.py -c config/evaluator.yaml --ckpt <path> --split test`

6. `evaluator_compare.py`
   - 作用：横向对比不同模型的评估表现。
   - 支持：`SVR`, `CART`, `LSTM`, `GRU`, `CNN-LSTM`, `MT-RP`。
   - 工作流程：
     - 读取多个模型 checkpoint。
     - 在统一 dataset 上计算对比指标。
     - 输出对比表、报告和各模型预测结果。 
   - 运行方式：
     - `python evaluator_compare.py`
     - `python evaluator_compare.py -c config/evaluator_compare.yaml`
     - `python evaluator_compare.py --ckpt_dir output/3_prediction/runs/`

7. `summarize_results.py`
   - 作用：汇总 `output/3_prediction/runs/` 下所有实验结果。
   - 工作流程：
     - 读取每个 run 的 `run_config.yaml` 和评估报告/指标。
     - 生成全量实验汇总 CSV、模型对比结果、消融结果。
     - 复制最优模型报告并附加原始配置。
   - 运行方式：
     - `python summarize_results.py`

## 目录关系说明

- `dataset.py` 负责数据集准备，是后续训练和评估的前提。
- `trainer_dl.py` / `trainer_svr_cart.py` 负责模型训练。
- `run_grid_search.py` 用于批量训练和调参。
- `evaluator.py` 和 `evaluator_compare.py` 负责单模型评估与模型间对比。
- `summarize_results.py` 负责整体结果汇总。

## 参考配置文件

- `config/dataset.yaml`
- `config/trainer_dl.yaml`
- `config/trainer_svr_cart.yaml`
- `config/evaluator.yaml`
- `config/evaluator_compare.yaml`
- `config/run_grid_search.yaml`
