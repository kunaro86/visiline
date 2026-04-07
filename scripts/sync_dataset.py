from pathlib import Path


def sync_jpg_to_xml(
    jpg_dir: str, xml_dir: str, dry_run: bool = False, ignore_case: bool = True
) -> dict[str, int]:
    """
    以 XML 标注目录为准，清理图片目录中无对应标注的 JPG 文件

    参数:
        jpg_dir: 存放 *.jpg 文件的目录路径
        xml_dir: 存放 *.xml 文件的目录路径（作为基准）
        dry_run: 是否开启安全预览模式。True 时仅打印操作不删除文件，False 时执行真实删除
        ignore_case: 是否忽略文件名大小写进行匹配（推荐 True）

    返回:
        包含 'kept'（保留）、'deleted'（删除）、'error'（失败）数量的统计字典
    """
    jpg_path = Path(jpg_dir)
    xml_path = Path(xml_dir)

    # 路径合法性校验
    if not jpg_path.is_dir():
        raise NotADirectoryError(f"图片目录不存在或无法访问: {jpg_dir}")
    if not xml_path.is_dir():
        raise NotADirectoryError(f"标注目录不存在或无法访问: {xml_dir}")

    # 1. 提取 XML 基准文件名集合
    xml_basenames: set[str] = set()
    for xml_file in xml_path.glob("*.xml"):
        name = xml_file.stem.lower() if ignore_case else xml_file.stem
        xml_basenames.add(name)

    if not xml_basenames:
        print("⚠️  标注目录中未找到任何 .xml 文件，已中止清理操作。")
        return {"kept": 0, "deleted": 0, "error": 0}

    stats = {"kept": 0, "deleted": 0, "error": 0}
    mode_label = "【预览】" if dry_run else "【执行】"

    # 2. 遍历 JPG 文件并比对
    jpg_files = list(jpg_path.glob("*.jpg"))
    for jpg_file in jpg_files:
        base_name = jpg_file.stem.lower() if ignore_case else jpg_file.stem

        if base_name not in xml_basenames:
            # 无对应标注，执行删除或预览
            if dry_run:
                print(f"{mode_label} 将移除无标注图片: {jpg_file.name}")
                stats["deleted"] += 1
            else:
                try:
                    jpg_file.unlink()
                    print(f"{mode_label} 已移除: {jpg_file.name}")
                    stats["deleted"] += 1
                except PermissionError:
                    print(f"{mode_label} 权限不足，跳过: {jpg_file.name}")
                    stats["error"] += 1
                except OSError as e:
                    print(f"{mode_label} 系统错误，跳过: {jpg_file.name} ({e})")
                    stats["error"] += 1
        else:
            stats["kept"] += 1

    # 3. 输出统计摘要
    print(
        f"\n✅ 同步完成 | 保留: {stats['kept']} | 移除: {stats['deleted']} | 异常: {stats['error']}"
    )
    return stats


if __name__ == "__main__":
    # 第一步：强烈建议先使用 dry_run=True 预览
    print("=== 预览模式 ===")
    stats_preview = sync_jpg_to_xml(
        jpg_dir="./raw/images/JPEGImages_backup", xml_dir="./raw/temp", dry_run=True
    )

    # 第二步：确认无误后，关闭 dry_run 执行真实清理
    print("\n=== 执行模式 ===")
    stats_final = sync_jpg_to_xml(
        jpg_dir="./raw/images/JPEGImages_backup", xml_dir="./raw/temp", dry_run=False
    )
