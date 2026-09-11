import os
import sys
import glob

# OSS 上传已从入库流程中移除（MinerU 改为签名 URL 直传文件），
# oss2 不再随镜像安装，缺失时仅发出警告，保证模块导入不报错
try:
    import oss2
    OSS2_AVAILABLE = True
except ImportError:
    OSS2_AVAILABLE = False
    oss2 = None
    print("警告: 未安装 oss2，upload_to_oss 的上传功能不可用（入库流程已不再依赖 OSS，可忽略）")


# ============ 配置区 ============
# 推荐从环境变量读取，避免把密钥写进代码
ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
# 桶名（必须和 pdf_mineru.py 里的硬编码域名一致）
BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "vl-image")
# 地域对应的 Endpoint
ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
# 在 OSS 上存放 PDF 的目录前缀
OBJECT_PREFIX = "pdf/"


def get_bucket():
    """创建并返回 OSS Bucket 对象。"""
    if not OSS2_AVAILABLE:
        raise RuntimeError(
            "未安装 oss2，无法使用 OSS 上传功能。"
            "MinerU 解析已改为签名 URL 直传文件，入库流程无需 OSS；"
            "如确需单独使用本模块上传，请先安装 oss2（pip install oss2）。"
        )
    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        raise RuntimeError(
            "缺少 OSS 访问凭证。请先在系统环境变量中设置 "
            "OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET，"
            "或在脚本顶部直接填入。"
        )
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
    return bucket


def upload_pdf(local_path: str, object_name: str = None) -> str:
    """
    上传单个 PDF 到 OSS，并返回公网可访问的 URL。

    :param local_path: 本地 PDF 文件路径
    :param object_name: OSS 上的对象名（文件名）。不传则使用本地文件名。
    :return: 公网 URL，格式为 https://<bucket>.<endpoint_host>/<prefix><object_name>
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"本地文件不存在: {local_path}")

    if object_name is None:
        object_name = os.path.basename(local_path)

    # 拼出 OSS 上的完整 key
    key = OBJECT_PREFIX + object_name

    bucket = get_bucket()

    # 设置请求头，让浏览器/下载工具把它当作 PDF 处理
    headers = {"Content-Type": "application/pdf"}

    print(f"[上传] {local_path}  ->  oss://{BUCKET_NAME}/{key}")
    result = bucket.put_object_from_file(key, local_path, headers=headers)

    if result.status != 200:
        raise RuntimeError(f"上传失败，HTTP 状态码: {result.status}")

    # 构造公网 URL（和 pdf_mineru.py 中的格式保持一致）
    # ENDPOINT 形如 https://oss-cn-shanghai.aliyuncs.com，去掉 https:// 后拼到 bucket 名后面
    host = ENDPOINT.replace("https://", "").replace("http://", "")
    public_url = f"https://{BUCKET_NAME}.{host}/{key}"
    print(f"[完成] 公网 URL: {public_url}")
    return public_url


def upload_dir(local_dir: str, pattern: str = "*.pdf") -> list:
    """
    批量上传目录下所有匹配 pattern 的文件。

    :param local_dir: 本地目录路径
    :param pattern: 文件名匹配模式，默认 *.pdf
    :return: 上传成功的 URL 列表
    """
    if not os.path.isdir(local_dir):
        raise NotADirectoryError(f"目录不存在: {local_dir}")

    files = glob.glob(os.path.join(local_dir, pattern))
    if not files:
        print(f"目录 {local_dir} 下没有匹配 {pattern} 的文件")
        return []

    print(f"找到 {len(files)} 个待上传文件")
    urls = []
    for fp in files:
        try:
            url = upload_pdf(fp)
            urls.append(url)
        except Exception as e:
            print(f"[失败] {fp}: {e}")
    return urls


if __name__ == "__main__":
    # 用法示例：
    # 1. 上传单个文件
    #    python upload_to_oss.py "D:\reports\中芯国际2024年报.pdf"
    # 2. 批量上传某个目录下所有 PDF
    #    python upload_to_oss.py --dir "D:\reports"
    # 3. 同时上传多个文件
    #    python upload_to_oss.py file1.pdf file2.pdf

    if len(sys.argv) < 2:
        print("用法:")
        print("  单文件: python upload_to_oss.py <本地PDF路径>")
        print("  批量:   python upload_to_oss.py --dir <本地目录> [glob模式，默认 *.pdf]")
        print("  多文件: python upload_to_oss.py <file1.pdf> <file2.pdf> ...")
        sys.exit(0)

    if sys.argv[1] == "--dir":
        # 批量模式
        local_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.pdf"
        upload_dir(local_dir, pattern)
    else:
        # 单文件或多文件模式
        for path in sys.argv[1:]:
            try:
                upload_pdf(path)
            except Exception as e:
                print(f"[失败] {path}: {e}")
