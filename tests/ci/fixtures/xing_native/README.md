# Xing4.0 native CI source

These three unmodified configuration and model files come from
[XingChen-AGI/Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/tree/a32a2e99b5877c6b7ad16e9954ec08488bcc5fa2),
revision `a32a2e99b5877c6b7ad16e9954ec08488bcc5fa2`:

- `config.json`
- `configuration_xing4_0.py`
- `modeling_xing4_0.py`

Copyright: China Telecom Artificial Intelligence Technology Co., Ltd.
License: Apache-2.0. The accompanying [LICENSE](LICENSE) is copied from the
[upstream repository](https://github.com/XingChen-AGI/Xing4.0-29B-A4B/tree/88b5308955ec9b072282ce0a7fd5d00e0029eb9c),
revision `88b5308955ec9b072282ce0a7fd5d00e0029eb9c`.

The fixture changes dimensions before constructing a two-layer model with
random weights and creates a small WordLevel tokenizer locally. No pretrained
weights or upstream tokenizer assets are included. CI runs offline using this
source; `XING4_0_MODEL_DIR` may explicitly override it for compatibility checks.
