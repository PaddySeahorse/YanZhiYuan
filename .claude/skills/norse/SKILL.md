# Norse Library Skills

Norse 是一个用于脉冲神经网络 (Spiking Neural Networks) 的深度学习库，基于 PyTorch 构建。

## 📚 文档文件

### 入门指南
- [root_README.md](root_README.md) - 项目主 README，安装和使用说明
- [root_contributing.md](root_contributing.md) - 贡献指南

### 开发文档
- [development.md](development.md) - 开发文档，包含架构说明和示例
- [hardware.md](hardware.md) - GPU 硬件加速指南
- [tasks.md](tasks.md) - 内置任务列表和使用方法

### API 参考
- [norse.torch.md](norse.torch.md) - norse.torch 模块文档（神经网络层）
- [norse.torch.functional.md](norse.torch.functional.md) - norse.torch.functional 文档（函数实现）
- [api.md](api.md) - 完整 API 概览
- [index.md](index.md) - 文档索引

### 关于
- [about.md](about.md) - 关于 Norse 项目
- [benchmark_README.md](benchmark_README.md) - 性能基准测试

## 💻 源代码

- [functional_coba_lif.py](functional_coba_lif.py) - 电导-based LIF 神经元实现

## 🚀 快速开始

```python
import torch
import norse.torch as snn

# 创建 LIF 神经元
layer = snn.LIFCell(input_features=10, hidden_features=20)

# 运行
data = torch.randn(8, 10)  # batch_size=8, input_features=10
output, state = layer(data)
```

## 📖 阅读顺序建议

1. 先看 [root_README.md](root_README.md) 了解项目
2. 查看 [tasks.md](tasks.md) 了解可用示例
3. 阅读 [development.md](development.md) 了解架构
4. 参考 [norse.torch.md](norse.torch.md) 和 [norse.torch.functional.md](norse.torch.functional.md) 了解 API
