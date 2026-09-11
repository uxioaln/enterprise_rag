import requests
import time
import zipfile
import os
import shutil

api_key = 'sk-FcYqWCqSy2wEgA58N10VLOCRLhIeowyDbMrBqO1KmzWmeBQs'

def apply_upload_url(file_name):
    """
    方式二：调用 MinerU 批量上传链接申请接口，获取签名上传 URL（不需要自己的 OSS）。
    :param file_name: 本地文件名（建议带正确后缀，如 'xxx.pdf'）
    :return: (batch_id, upload_url) 元组；签名 URL 有效期 24 小时
    """
    url = 'https://mineru.net/api/v4/file-urls/batch'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }
    data = {
        # file 级参数：is_ocr 与原 URL 方式保持一致，开启 OCR
        'files': [
            {
                'name': file_name,
                'is_ocr': True,
            }
        ],
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

def get_batch_id(file_path):
    """
    一步完成"申请上传链接 + 上传本地文件"，返回 batch_id（替代原 URL 方式的 get_task_id）。
    上传完成后 MinerU 会自动提交解析任务，后续用 get_result(batch_id) 轮询结果。
    :param file_path: 本地 PDF 文件路径
    :return: batch_id
    """
    file_name = os.path.basename(file_path)
    batch_id, upload_url = apply_upload_url(file_name)
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

if __name__ == "__main__":
    file_path = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    batch_id = get_batch_id(file_path)
    print('batch_id:', batch_id)
    get_result(batch_id)
