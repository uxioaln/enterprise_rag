import requests
import time
import zipfile
import os
import shutil

api_key = 'sk-FcYqWCqSy2wEgA58N10VLOCRLhIeowyDbMrBqO1KmzWmeBQs'

def apply_upload_url(file_name, page_range=None):
    """
    方式二：调用 MinerU 批量上传链接申请接口，获取签名上传 URL（不需要自己的 OSS）。
    :param file_name: 本地文件名（建议带正确后缀，如 'xxx.pdf'）
    :param page_range: 可选，页码范围字符串（如 '1-200'），传入时仅解析指定页码段。
                      MinerU v4 batch 接口原生支持 file.page_ranges 字段。
                      为 None 时解析整份文件（向后兼容）。
    :return: (batch_id, upload_url) 元组；签名 URL 有效期 24 小时
    """
    url = 'https://mineru.net/api/v4/file-urls/batch'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }
    # 构建 file 对象：基础字段 is_ocr=True，可选 page_ranges 字段
    file_obj = {
        'name': file_name,
        'is_ocr': True,
    }
    if page_range is not None:
        # MinerU v4 文档：file.page_ranges 形如 "1-200" 或 "2,4-6"
        file_obj['page_ranges'] = page_range
    data = {
        # file 级参数：is_ocr 与原 URL 方式保持一致，开启 OCR
        'files': [file_obj],
        # 请求级参数：与原 URL 方式保持一致，关闭公式识别
        'enable_formula': False,
    }

    res = requests.post(url,headers=header,json=data)
    print(res.status_code)
    print(res.json())
    result = res.json()
    # 接口状态码非 0 时视为申请失败
    if result.get('code') != 0:
        raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")
    batch_id = result['data']['batch_id']
    # files 只传了一个文件，file_urls 顺序与 files 一致，取第一个即可
    upload_url = result['data']['file_urls'][0]
    return batch_id, upload_url

def upload_file(upload_url, file_path):
    """
    将本地文件通过 PUT 上传到 MinerU 返回的签名上传 URL。
    注意：上传文件时无须设置 Content-Type 请求头；上传完成后系统会自动提交解析任务。
    :param upload_url: apply_upload_url 返回的签名上传链接
    :param file_path: 本地文件路径
    """
    with open(file_path, 'rb') as f:
        res = requests.put(upload_url, data=f)
    print(res.status_code)
    if res.status_code != 200:
        raise RuntimeError(f"上传文件失败，HTTP 状态码: {res.status_code}, 响应内容: {res.text}")
    print(f"文件上传成功: {file_path}")

def get_batch_id(file_path, page_range=None):
    """
    一步完成"申请上传链接 + 上传本地文件"，返回 batch_id（替代原 URL 方式的 get_task_id）。
    上传完成后 MinerU 会自动提交解析任务，后续用 get_result(batch_id) 轮询结果。
    :param file_path: 本地 PDF 文件路径
    :param page_range: 可选，页码范围字符串（如 '1-200'），用于分段解析；为 None 时解析整份文件
    :return: batch_id
    """
    file_name = os.path.basename(file_path)
    batch_id, upload_url = apply_upload_url(file_name, page_range=page_range)
    print('batch_id:', batch_id)
    upload_file(upload_url, file_path)
    return batch_id

def get_result(task_id):
    """
    轮询批量解析结果并下载解压（task_id 即申请上传链接时返回的 batch_id）。
    任务状态：waiting-file（等待上传）/ pending（排队中）/ running（解析中）/
    converting（格式转换中）/ done（完成）/ failed（失败）。
    """
    url = f'https://mineru.net/api/v4/extract-results/batch/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        # 批量接口返回 extract_result 列表，这里只提交了一个文件，取第一条
        extract_result = result.get('extract_result', [])
        if not extract_result:
            print("暂无解析结果，等待5秒后重试...")
            time.sleep(5)
            continue
        item = extract_result[0]
        print(item)
        state = item.get('state')
        err_msg = item.get('err_msg', '')
        # 如果任务还在进行中，等待后重试
        if state in ['waiting-file', 'pending', 'running', 'converting']:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        # 任务明确失败（MinerU 返回 failed 状态或携带 err_msg）
        if state == 'failed' or err_msg:
            # 抛出异常而非返回 None，让 Celery 任务能捕获并传递真实错误信息给前端
            raise RuntimeError(f"MinerU 解析失败: {err_msg or state}")
        # 如果任务完成，下载文件
        if state == 'done':
            full_zip_url = item.get('full_zip_url')
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                # 解压到与 zip 同名的目录（保留 full.md / content_list.json / images/ 等所有文件）
                extract_dir = unzip_file(local_filename)
                return extract_dir
            else:
                raise RuntimeError("MinerU 解析完成但未返回 full_zip_url，无法下载结果")
        # 其他未知状态
        raise RuntimeError(f"MinerU 返回未知状态: {state}")

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹，返回解压目录的绝对路径。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录（去 .zip 后缀）
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    abs_dir = os.path.abspath(extract_dir)
    print(f"已解压到: {abs_dir}")
    return abs_dir

# 将 MinerU 解压目录中的 content_list.json 和 images/ 抽取到指定目标目录
def export_content_list(extract_dir, target_dir, file_name):
    """
    从 MinerU 的解压目录中复制 content_list.json 和 images/ 到目标目录。
    目标目录结构：
        target_dir/
            {file_name}_content_list.json
            {file_name}_content_list_v2.json  (如果存在)
            images/
    :param extract_dir: unzip_file 返回的解压目录
    :param target_dir: 目标目录（会自动创建）
    :param file_name: 用于命名 content_list 文件（一般是 pdf 去扩展名的 base_name）
    :return: 目标目录的绝对路径
    """
    extract_dir = os.path.abspath(extract_dir)
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 拷贝 content_list.json
    # 统一重命名为 {file_name}_content_list.json，让下游能通过 file_name 关联到 subset.csv
    src_cl_v1 = os.path.join(extract_dir, f"{file_name}_content_list.json")
    dst_cl_v1 = os.path.join(target_dir, f"{file_name}_content_list.json")
    if os.path.exists(src_cl_v1):
        shutil.copy2(src_cl_v1, dst_cl_v1)
        print(f"已复制: {src_cl_v1} -> {dst_cl_v1}")
    else:
        # 兜底：解压目录里的实际文件可能带有 task_id 前缀，glob 匹配后统一重命名
        import glob
        candidates = glob.glob(os.path.join(extract_dir, "*_content_list.json"))
        for c in candidates:
            shutil.copy2(c, dst_cl_v1)
            print(f"已复制并重命名: {c} -> {dst_cl_v1}")
        candidates_v2 = glob.glob(os.path.join(extract_dir, "*_content_list_v2.json"))
        dst_cl_v2 = os.path.join(target_dir, f"{file_name}_content_list_v2.json")
        for c in candidates_v2:
            shutil.copy2(c, dst_cl_v2)
            print(f"已复制并重命名: {c} -> {dst_cl_v2}")

    # 拷贝 images/ 目录
    src_images = os.path.join(extract_dir, "images")
    if os.path.isdir(src_images):
        dst_images = os.path.join(target_dir, "images")
        if os.path.exists(dst_images):
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)
        print(f"已复制目录: {src_images} -> {dst_images}")

    return target_dir


# --------------------------------------------------------------------------
# 大文件分段解析：超过 chunk_size 页的 PDF 分多段提交，最后合并 content_list
# --------------------------------------------------------------------------

# 默认分段页数阈值：MinerU 单次解析建议 ≤200 页，超过则分段循环
DEFAULT_CHUNK_SIZE = 200


def get_total_pages(file_path):
    """
    读取 PDF 总页数。优先使用 pypdfium2（docling 已安装该依赖），
    缺失时返回 None，调用方可据此回退到不分段的单次解析。
    :param file_path: 本地 PDF 文件路径
    :return: int 总页数；None 表示无法读取（依赖缺失）
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("警告：未安装 pypdfium2，无法读取 PDF 页数，将走单段解析。"
              "如需启用分段，请安装：pip install pypdfium2 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return None
    try:
        pdf = pdfium.PdfDocument(file_path)
        total = len(pdf)
        pdf.close()
        return total
    except Exception as err:
        print(f"警告：读取 PDF 页数失败：{err}，将走单段解析。")
        return None


def build_page_range_segments(total_pages, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    根据总页数构建分段区间列表（1-based 闭区间）。
    例如 total_pages=500, chunk_size=200 -> [(1,200),(201,400),(401,500)]
    :param total_pages: 总页数
    :param chunk_size: 每段最大页数
    :return: [(start, end), ...] 列表
    """
    if total_pages <= 0:
        return []
    segments = []
    start = 1
    while start <= total_pages:
        end = min(start + chunk_size - 1, total_pages)
        segments.append((start, end))
        start = end + 1
    return segments


def _load_segment_content_list(extract_dir, file_name_no_ext):
    """
    从单段解析的解压目录中读取 content_list.json 与 full.md，返回 (items, images_dir, md_path)。
    兼容两种命名：{file_name_no_ext}_content_list.json 与 task_id 前缀的 *_content_list.json。
    :param extract_dir: get_result 返回的解压目录
    :param file_name_no_ext: PDF 去扩展名后的 stem，用于优先匹配
    :return: (items 列表, images 目录绝对路径或 None, full.md 绝对路径或 None)
    """
    import json as _json
    import glob as _glob

    extract_dir = os.path.abspath(extract_dir)
    # 优先精确匹配 {stem}_content_list.json
    candidates = [os.path.join(extract_dir, f"{file_name_no_ext}_content_list.json")]
    # 兜底：glob 匹配 *_content_list.json（避开 _v2 版本）
    candidates += _glob.glob(os.path.join(extract_dir, "**", "*_content_list.json"), recursive=True)
    candidates += _glob.glob(os.path.join(extract_dir, "*_content_list.json"))

    cl_path = None
    for c in candidates:
        if os.path.exists(c) and not c.endswith("_content_list_v2.json"):
            cl_path = c
            break

    items = []
    if cl_path and os.path.exists(cl_path):
        with open(cl_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        # content_list.json 顶层即 items 数组（MinerU 标准格式）
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get('content_list'), list):
            items = data['content_list']

    images_dir = os.path.join(extract_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = None

    # 定位该段 full.md：解压目录根下，或嵌套子目录
    md_path = os.path.join(extract_dir, "full.md")
    if not os.path.exists(md_path):
        md_candidates = _glob.glob(os.path.join(extract_dir, "**", "full.md"), recursive=True)
        if md_candidates:
            md_path = md_candidates[0]
        else:
            md_path = None

    return items, images_dir, md_path


def _merge_segment_images(seg_images_dir, seg_idx, merged_images_dir, items):
    """
    将单段 images 目录合并到统一目录，文件名加段前缀避免冲突，
    并同步更新 items 中引用该段图片的 img_path 字段。
    :param seg_images_dir: 该段 images 目录
    :param seg_idx: 段号（从 1 开始）
    :param merged_images_dir: 合并后的统一 images 目录
    :param items: 该段的 content_list items（已做 page_idx 偏移）
    """
    if not seg_images_dir or not os.path.isdir(seg_images_dir):
        return
    os.makedirs(merged_images_dir, exist_ok=True)
    prefix = f"seg{seg_idx}_"
    # 段内原文件名 -> 合并后新文件名映射
    name_map = {}
    for fname in os.listdir(seg_images_dir):
        src = os.path.join(seg_images_dir, fname)
        if not os.path.isfile(src):
            continue
        new_name = prefix + fname
        dst = os.path.join(merged_images_dir, new_name)
        shutil.copy2(src, dst)
        name_map[fname] = new_name

    # 更新 items 中的 img_path 字段：把原文件名替换为加段前缀的新名
    for item in items:
        img_path = item.get('img_path') or item.get('img_name')
        if not img_path:
            continue
        # img_path 可能是 "images/xxx.jpg" 或纯文件名
        base = os.path.basename(img_path)
        if base in name_map:
            new_base = name_map[base]
            new_path = os.path.join('images', new_base)
            if 'img_path' in item:
                item['img_path'] = new_path
            if 'img_name' in item:
                item['img_name'] = new_base


def _write_merged_extract_dir(merged_dir, file_name_no_ext, merged_items, seg_images_dir, seg_md_paths):
    """
    将合并后的 content_list / images / full.md 写到一个合成解压目录，
    结构与单段 MinerU 解压目录一致，便于复用 export_content_list 等下游函数。
    :param merged_dir: 合并后的解压目录
    :param file_name_no_ext: PDF 去扩展名 stem，用于命名 content_list 文件
    :param merged_items: 合并后的 content_list items
    :param seg_images_dir: 已合并好的 images 目录路径
    :param seg_md_paths: 各段 full.md 路径列表（按段顺序），用于拼接为合并后的 full.md
    """
    import json as _json
    os.makedirs(merged_dir, exist_ok=True)
    # 写 content_list.json：统一命名为 {stem}_content_list.json（与 export_content_list 期望一致）
    cl_path = os.path.join(merged_dir, f"{file_name_no_ext}_content_list.json")
    with open(cl_path, 'w', encoding='utf-8') as f:
        _json.dump(merged_items, f, ensure_ascii=False, indent=2)
    print(f"已写合并 content_list: {cl_path}（共 {len(merged_items)} 个 item）")
    # 复制合并好的 images 目录到合成解压目录
    if seg_images_dir and os.path.isdir(seg_images_dir):
        dst_images = os.path.join(merged_dir, "images")
        if os.path.exists(dst_images):
            shutil.rmtree(dst_images)
        shutil.copytree(seg_images_dir, dst_images)
        print(f"已合并 images 目录: {seg_images_dir} -> {dst_images}")
    # 拼接各段 full.md 为合并后的 full.md（与 page_idx 偏移后的页码顺序一致）
    md_out_path = os.path.join(merged_dir, "full.md")
    with open(md_out_path, 'w', encoding='utf-8') as out_f:
        for i, md_path in enumerate(seg_md_paths):
            if md_path and os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read()
                if i > 0:
                    out_f.write("\n\n---\n\n")
                out_f.write(content)
    print(f"已写合并 full.md: {md_out_path}（共拼接 {len(seg_md_paths)} 段）")


def parse_pdf(file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    大文件分段解析主入口：
    - 总页数 ≤ chunk_size：走原单段逻辑（get_batch_id + get_result）
    - 总页数 > chunk_size：按 chunk_size 分段循环申请签名 URL（带 page_ranges），
      上传同一份 PDF，每段独立轮询解析；最后合并 content_list 的 page_idx 偏移与 images 目录，
      产出统一解压目录，结构与单段一致，下游 export_content_list 可直接复用。
    :param file_path: 本地 PDF 文件路径
    :param chunk_size: 每段最大页数，默认 200
    :return: 合并后的解压目录绝对路径；单段时即 get_result 返回值
    """
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    total_pages = get_total_pages(file_path)
    # 无法读取页数或页数未超阈值：走原单段逻辑
    if total_pages is None or total_pages <= chunk_size:
        if total_pages is not None:
            print(f"[parse_pdf] 总页数 {total_pages} ≤ {chunk_size}，走单段解析")
        else:
            print(f"[parse_pdf] 无法读取页数，走单段解析（按整份文件提交）")
        batch_id = get_batch_id(file_path)
        return get_result(batch_id)

    # 分段循环
    segments = build_page_range_segments(total_pages, chunk_size)
    print(f"[parse_pdf] 总页数 {total_pages} > {chunk_size}，分 {len(segments)} 段解析：{segments}")

    merged_items = []
    # 统一 images 目录：所有段图片合并到此处，文件名加段前缀防冲突
    merged_images_root = os.path.abspath(f"{file_name_no_ext}_merged_images_tmp")
    has_any_images = False
    seg_md_paths = []  # 各段 full.md 路径，按段顺序拼接为合并后的 full.md
    page_offset = 0  # 累计已处理页数，用于修正 page_idx

    for idx, (start, end) in enumerate(segments):
        seg_idx = idx + 1
        page_range = f"{start}-{end}"
        print(f"\n[parse_pdf] === 段 {seg_idx}/{len(segments)} 页码 {page_range} ===")
        batch_id = get_batch_id(file_path, page_range=page_range)
        seg_extract_dir = get_result(batch_id)
        if not seg_extract_dir or not os.path.isdir(seg_extract_dir):
            raise RuntimeError(f"段 {seg_idx} (页码 {page_range}) 解析失败，未返回有效解压目录")

        # 读取该段 content_list / images / full.md
        seg_items, seg_images_dir, seg_md_path = _load_segment_content_list(seg_extract_dir, file_name_no_ext)
        print(f"[parse_pdf] 段 {seg_idx} 解析得到 {len(seg_items)} 个 item")

        # page_idx 偏移修正：MinerU 每段 page_idx 从 0 起，需加上前段累计页数
        seg_pages = end - start + 1
        for item in seg_items:
            raw_idx = item.get('page_idx')
            if raw_idx is not None:
                item['page_idx'] = raw_idx + page_offset

        # 合并 images：段前缀防冲突，同步更新 items 的 img_path
        if seg_images_dir:
            _merge_segment_images(seg_images_dir, seg_idx, merged_images_root, seg_items)
            has_any_images = True

        seg_md_paths.append(seg_md_path)
        merged_items.extend(seg_items)
        page_offset += seg_pages
        print(f"[parse_pdf] 段 {seg_idx} 完成，累计 page_offset={page_offset}")

    # 产出合成解压目录（与单段解压目录结构一致）
    # 目录名以原 PDF stem + _merged 命名，放在当前工作目录下
    merged_dir = os.path.abspath(f"{file_name_no_ext}_merged")
    _write_merged_extract_dir(
        merged_dir, file_name_no_ext, merged_items,
        merged_images_root if has_any_images else None,
        seg_md_paths
    )
    print(f"\n[parse_pdf] 全部 {len(segments)} 段合并完成 -> {merged_dir}")
    print(f"[parse_pdf] 合并后总 item 数: {len(merged_items)}，覆盖页码 1-{page_offset}")
    return merged_dir


if __name__ == "__main__":
    file_path = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    # 演示：自动按 200 页分段解析；≤200 页走单段
    extract_dir = parse_pdf(file_path)
    print('extract_dir:', extract_dir)
