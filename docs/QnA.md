
## 🐛 常见问题


### Q: 可以使用 COCO 格式的标注吗？

A: 当前版本主要支持 YOLO 格式。如需使用 COCO 格式，需要先转换为 YOLO 格式。

### Q: data.yaml 中的类别与我的数据集不符怎么办？⚠️

A: **重要**：修改 `data/data.yaml` 中的 `nc` 和 `names` 与你的数据集对应。

`data.yaml` 默认包含43个GTSRB交通标志类别。如果你的数据集不同，**必须修改**！

**检查步骤**：
```bash
# 1. 统计你的标签中有多少个不同类别
grep -oE "^[0-9]+" data/train/labels/*.txt | cut -d: -f2 | sort -u | wc -l

# 2. 检查data.yaml中的nc值
grep "^nc:" data/data.yaml

# 如果不一致，需要修改！
```

**修改方法**：
```bash
# 编辑 data/data.yaml
nano data/data.yaml
```

改为你的实际类别数和名称：
```yaml
nc: 5  # 改为你实际的类别数
names:
  0: Your Class A
  1: Your Class B
  2: Your Class C
  3: Your Class D
  4: Your Class E
```

**保护机制**：执行 `prepare-data` 命令时：
- ✅ 如果 `/configs/data.yaml` 已存在，**不会覆盖**（保护你的修改）
- ✅ 仅在首次运行或使用 `--force` 标志时生成

**强制重新生成**（谨慎使用）：
```bash
python main.py prepare-data --force --split --images-dir raw/images --labels-dir raw/labels
# ⚠️  这会覆盖你现有的 data.yaml，请确保已备份重要配置！
```

### Q: 训练过程中显存不足怎么办？

A: 降低批大小或图像大小：
```bash
python main.py train --batch-size 8 --config configs/default.yaml
```

### Q: 如何加快训练速度？

A: 使用快速配置或简化模型：
```bash
python main.py train --config configs/quick.yaml --model yolov8n
```

### Q: 推理结果格式是什么？

A: JSON格式，包含每个检测框的类别、置信度和坐标，详见 [inference.py](../src/inference.py)

