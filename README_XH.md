# XH2a的GPTQ

## 安装依赖

1. 环境配置

```bash
#setuptools 80之后的版本采用 /opt/extdata/.conda/envs/torch2_7/bin/python -m pip install -e . --use-pep517， 多了--use-pep517参数会导致找不到torch
pip install setuptools==79.0.1
pip uninstall SoundFile
pip install SoundFile
```
