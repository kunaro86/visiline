import xml.etree.ElementTree as ET
from pathlib import Path


def voc_to_yolo(
    voc_dir: str,
    output_dir: str,
    class_names: list[str],
    image_ext: str = ".jpg",
    encoding: str = "utf-8",
) -> dict[str, int]:
    """
    将 Pascal VOC 格式的 XML 标注文件批量转换为 YOLO 格式的 TXT 文件

    参数:
        voc_dir: VOC 标注文件所在目录（包含 .xml 文件）
        output_dir: 转换后 YOLO 标注文件的输出目录
        class_names: 类别名称列表，索引即为 YOLO 中的 class_id
                    例如: ["person", "bicycle", "car", "motorbike"]
        image_ext: 对应图像文件的扩展名，用于验证文件存在性（默认 ".jpg"）
        encoding: XML 文件编码（默认 "utf-8"）

    返回:
        统计字典，包含转换成功的文件数、跳过文件数、错误文件数

    功能说明:
        1. 遍历 voc_dir 下所有 .xml 文件
        2. 解析 XML 获取图像尺寸和物体标注
        3. 将边界框坐标从 (xmin, ymin, xmax, ymax) 转换为归一化的
           (x_center, y_center, width, height)
        4. 将类别名称映射为整数 ID
        5. 保存为 YOLO 格式的 .txt 文件到 output_dir
    """

    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 构建类别名称到 ID 的映射
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # 统计信息
    stats = {"success": 0, "skipped": 0, "error": 0}

    # 获取所有 VOC XML 文件
    xml_files = list(Path(voc_dir).glob("*.xml"))

    for xml_path in xml_files:
        try:
            # 解析 XML
            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            # 提取图像尺寸
            size = root.find("size")
            if size is None:
                raise ValueError(f"未找到 <size> 标签: {xml_path.name}")
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            # 构建输出文件路径（保持原文件名，仅改扩展名）
            output_txt = Path(output_dir) / xml_path.with_suffix(".txt").name

            # 收集当前图像的所有标注
            yolo_lines = []

            # 遍历所有物体标注
            for obj in root.findall("object"):
                # 提取类别和边界框
                class_name = obj.find("name").text
                bndbox = obj.find("bndbox")

                if class_name not in class_to_id:
                    print(
                        f"⚠️  未知类别 '{class_name}'，已跳过（文件: {xml_path.name}）"
                    )
                    continue

                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # 坐标转换：绝对像素 → 归一化中心坐标
                x_center = ((xmin + xmax) / 2.0) / img_width
                y_center = ((ymin + ymax) / 2.0) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 边界检查（防止因标注错误导致坐标越界）
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                class_id = class_to_id[class_name]
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

            # 写入 YOLO 格式文件（即使为空也创建文件，保持与图像一一对应）
            with open(output_txt, "w", encoding=encoding) as f:
                f.writelines(yolo_lines)

            stats["success"] += 1

        except ET.ParseError as e:
            print(f"❌ XML 解析失败: {xml_path.name} - {e}")
            stats["error"] += 1
        except Exception as e:
            print(f"❌ 处理失败: {xml_path.name} - {e}")
            stats["error"] += 1

    # 输出统计摘要
    total = len(xml_files)
    print(
        f"\n✅ 转换完成: {stats['success']}/{total} 成功, "
        f"{stats['skipped']} 跳过, {stats['error']} 错误"
    )

    return stats


if __name__ == "__main__":
    # 定义你的类别列表（顺序很重要！索引 = class_id）
    my_classes = ["edge"]

    # 调用转换函数
    result = voc_to_yolo(
        voc_dir="./raw/temp",  # VOC XML 标注目录
        output_dir="./raw/labels",  # 输出目录
        class_names=my_classes,
        image_ext=".jpg",
    )

    # 查看转换结果
    print(f"成功: {result['success']}, 错误: {result['error']}")
