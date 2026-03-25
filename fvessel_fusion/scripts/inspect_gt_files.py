from pathlib import Path
import argparse
import pandas as pd


def try_read_txt(txt_path: Path):
    """
    尝试读取 gt txt，自动猜分隔符
    """
    seps = [",", "\t", " "]
    last_err = None

    for sep in seps:
        try:
            if sep == " ":
                df = pd.read_csv(txt_path, sep=r"\s+", header=None, engine="python")
            else:
                df = pd.read_csv(txt_path, sep=sep, header=None, engine="python")
            return df, sep
        except Exception as e:
            last_err = e

    raise RuntimeError(f"读取失败: {txt_path}, last_err={last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    gt_dir = clip_dir / "gt"

    if not gt_dir.exists():
        raise FileNotFoundError(f"找不到 gt 目录: {gt_dir}")

    txt_files = sorted(gt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"gt 目录下没有 txt 文件: {gt_dir}")

    print("=" * 60)
    print("GT 文件列表")
    print("=" * 60)
    for f in txt_files:
        print(f.name)

    print("\n" + "=" * 60)
    print("逐个读取预览")
    print("=" * 60)

    for txt_path in txt_files:
        print("\n" + "-" * 60)
        print(f"文件: {txt_path.name}")

        df, sep = try_read_txt(txt_path)
        print(f"分隔符推测: {repr(sep)}")
        print(f"shape: {df.shape}")
        print("前5行:")
        print(df.head())

        # 保存一个预览 csv，方便你用 Excel 看
        out_csv = gt_dir / f"{txt_path.stem}_preview.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"已导出预览: {out_csv}")

    print("\n完成。下一步就能根据列数判断各文件含义。")


if __name__ == "__main__":
    main()
