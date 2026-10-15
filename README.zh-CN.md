# DWTS 投票机制分析

**当观众票数不可见时，投票规则会怎样影响比赛结果？**

本仓库是 Team 2627761 在 2026 MCM Problem C 中开展的 DWTS 研究的软件实现与后续扩展。Yike chen 参与了该团队的原始研究。项目基于公开评分与淘汰记录，推断潜在粉丝支持、比较投票规则，并分析选手及舞伴特征与评分的关联。

[English README](README.md) · [方法](docs/METHODS.md) · [结果](docs/RESULTS.md) · [运行指南](docs/RUNNING.md)

## 研究问题与方法

- 哪些粉丝支持模式与公开评分和淘汰结果相容？
- 名次法、百分比法，以及评委或粉丝救援机制，会怎样改变每周结果？
- 选手特征和舞伴历史表现与评分、推断粉丝支持有什么关联？

项目包括贝叶斯逆向推断、PyMC 参考模型、Numba Metropolis–Hastings 采样引擎、规则模拟、按赛季分组的嵌套随机森林、OLS 和 SHAP。配置和输入校验值记录实验假设；精确并列排序与舞伴身份归一化修正了两项已确认的实现问题。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
python -m dwts.demo --out outputs/my-demo
python -m pytest -q
```

已验证环境为 Python 3.12。Windows PowerShell 的激活命令是 `.venv\Scripts\Activate.ps1`。

演示使用四位虚构选手和虚构票数。评委与粉丝各占一半时，名次法淘汰 A，百分比法淘汰 C。输出包含 CSV、JSON 和 SVG；已有输出目录不会被覆盖。

默认测试无需外部数据；需要官方数据或历史实验的测试会明确排除。GitHub Actions 运行自包含测试和演示。

## 已保存的实验结果

当前代码版本保存的 v5 实验覆盖 34 季、2,777 个选手周，采用五折外层赛季分组验证，并在内层选择历史权重。

| 回归目标 | 折外 R² | 折外 RMSE |
|---|---:|---:|
| 评委均分 | 0.434488 | 1.100571 |
| 推断粉丝份额 | 0.555199 | 0.063604 |

评委 RMSE 使用均分单位，粉丝 RMSE 使用份额单位。这些是已保存的研究结果，快速演示不会拟合这些模型。[结果说明](docs/RESULTS.md)记录六项任务、与参考报告的数值差异，以及解释范围。

## 运行分析

按[运行指南](docs/RUNNING.md)取得官方数据，运行完整研究：

```bash
python -m dwts run --config configs/study.json --out outputs/my-study
```

所有拟合赛季通过采样诊断后，可继续分析结果：

```bash
python -m dwts.analysis --source outputs/my-study --out outputs/my-analysis
```

该入口比较声明的救援机制，使用此前周评委份额拟合 OLS，并分别计算争议周比例和事件率。未定义的多人救援周留空；各项样本范围和分母记录在输出中。它复用已拟合的后验均值。

正常完成的独立分析返回退出码 **0**。与原报告的数值对照通过 `--compare-reference` 单独启用，并记录自己的比较状态；详见[运行指南](docs/RUNNING.md)。

## 结果的解释范围

- 粉丝份额是潜变量估计，真实观众票数不可见。
- 全季后验用于回顾性分析，尚未评估前瞻预测表现。
- SHAP 和特征重要性反映关联，不能证明舞伴的因果作用。
- 规则比较保留每周实际参赛名单，未模拟完整的替代赛季。
- 下游 RF/OLS 使用后验均值，未传播全部票数估计不确定性。

## 研究团队与来源

原始研究由 Team 2627761 共同完成，Yike chen 是团队成员。当前代码版本及后续扩展与比赛期间工作的区别，见[来源与贡献说明](docs/PROVENANCE.md)。原报告和原始研究成果保留完整团队归属。

原报告、原始数据和后验采样链不随仓库发布。公开可见不代表已授予开源许可证，见[许可状态](LICENSE_STATUS.md)。
