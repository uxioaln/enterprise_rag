import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5, lines_per_page: int = 50):
        """
        按行分割 markdown 文件，每个分块记录起止行号、估算页码和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param lines_per_page: 估算的每页行数（年报通常 40-60 行/页）
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            # 根据行号估算页码：行号 // lines_per_page + 1
            estimated_page = (start // lines_per_page) + 1
            chunks.append({
                'lines': [start + 1, end],  # 行号从1开始
                'page': estimated_page,     # 估算的页码
                'text': chunk_text
            })
            i += chunk_size - chunk_overlap
        return chunks

    def split_reports_from_content_list(self, content_list_dir: Path, output_dir: Path, subset_csv: Path = None, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        基于 MinerU 的 content_list.json（V1 扁平结构）按真实页码分块。
        与 split_markdown_reports（按行估算页码）的区别：
        - 页码直接来自 content_list.json 的 page_idx 字段（0-based 转 1-based），更准确。
        - 文本块：按页聚合后用 RecursiveCharacterTextSplitter 切分，每块带 page。
        - 表格：单独成块，type='image_table'，带 page、img_path、table_body、caption、footnote。
        - 图片：单独成块，type='image_table'，带 page、img_path、image_caption。
        :param content_list_dir: 存放 *_content_list.json 与 images/ 的目录
        :param output_dir: 输出分块 JSON 的目录
        :param subset_csv: subset.csv 路径（用于补充 company_name / sha1 / file_name）
        :param chunk_size: 文本分块大小（token）
        :param chunk_overlap: 文本分块重叠 token
        """
        # 建立 file_name 去扩展名 -> (company_name, sha1) 的映射
        file2meta = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2meta[file_no_ext] = {
                        'company_name': row.get('company_name', ''),
                        'sha1': row.get('sha1', ''),
                        'file_name': str(row['file_name'])
                    }
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file2meta[str(row['sha1'])] = {
                        'company_name': row.get('company_name', ''),
                        'sha1': row.get('sha1', ''),
                        'file_name': str(row.get('sha1', ''))
                    }
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')

        output_dir.mkdir(parents=True, exist_ok=True)
        cl_paths = list(content_list_dir.glob("*_content_list.json"))
        # 过滤掉 v2 版本（仅处理 v1）
        cl_paths = [p for p in cl_paths if not p.name.endswith("_content_list_v2.json")]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for cl_path in cl_paths:
            # file_name_no_ext 来自 content_list 文件名的前缀部分（去掉 _content_list.json）
            file_name_no_ext = cl_path.name.replace("_content_list.json", "")
            meta = file2meta.get(file_name_no_ext)
            # 查不到 subset.csv 中对应的条目（通常是 task_id 命名的旧 content_list），
            # 直接跳过，避免产生 sha1 为空的脏 chunked_reports，进而阻塞后续 create_vector_dbs
            if meta is None:
                print(f"警告：content_list 文件 {cl_path.name} 在 subset.csv 中找不到对应条目（{file_name_no_ext}），跳过该文件。请确认其来源是否为合法的 PDF 文件名（去扩展名），如不需要请删除该 content_list 后重跑 chunk_reports。")
                continue

            with open(cl_path, 'r', encoding='utf-8') as f:
                items = json.load(f)

            # 按 page_idx 聚合 text 类型 item
            page_to_texts: Dict[int, List[str]] = {}
            # 收集表格和图片
            table_chunks: List[Dict] = []
            image_chunks: List[Dict] = []

            for item in items:
                item_type = item.get('type', '')
                page_idx = item.get('page_idx')
                if page_idx is None:
                    continue
                page = page_idx + 1  # 0-based -> 1-based

                if item_type in ('text',):
                    # text 类型（可能含 title 层级）
                    text_content = item.get('text', '')
                    if not text_content or not text_content.strip():
                        continue
                    page_to_texts.setdefault(page, []).append(text_content)
                elif item_type == 'table':
                    # 表格单独成块
                    table_text = self._table_to_text(item)
                    table_chunks.append({
                        "page": page,
                        "type": "image_table",
                        "sub_type": "table",
                        "img_path": item.get('img_path', ''),
                        "table_caption": item.get('table_caption', []),
                        "table_footnote": item.get('table_footnote', []),
                        "table_body": item.get('table_body', ''),
                        "text": table_text,
                        "length_tokens": self.count_tokens(table_text)
                    })
                elif item_type == 'image':
                    # 图片单独成块
                    image_text = self._image_to_text(item)
                    image_chunks.append({
                        "page": page,
                        "type": "image_table",
                        "sub_type": "image",
                        "img_path": item.get('img_path', ''),
                        "image_caption": item.get('image_caption', []),
                        "image_footnote": item.get('image_footnote', []),
                        "text": image_text,
                        "length_tokens": self.count_tokens(image_text)
                    })
                # 其他类型（equation / code）暂不单独成块，已在 markdown 文本流中体现

            # 对每页文本做 token 切块
            chunks: List[Dict] = []
            chunk_id = 0
            for page in sorted(page_to_texts.keys()):
                page_text = "\n\n".join(page_to_texts[page])
                if not page_text.strip():
                    continue
                page_chunks = text_splitter.split_text(page_text)
                for ck in page_chunks:
                    chunks.append({
                        "id": chunk_id,
                        "page": page,
                        "type": "content",
                        "text": ck,
                        "length_tokens": self.count_tokens(ck)
                    })
                    chunk_id += 1

            # 表格/图片块按页追加到 chunks 末尾
            for extra in table_chunks + image_chunks:
                extra['id'] = chunk_id
                chunks.append(extra)
                chunk_id += 1

            metainfo = {
                "sha1": meta.get('sha1', ''),
                "company_name": meta.get('company_name', ''),
                "file_name": meta.get('file_name', file_name_no_ext + '.pdf')
            }
            out_path = output_dir / f"{file_name_no_ext}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理(content_list): {cl_path.name} -> {out_path.name}，共 {len(chunks)} 个分块（文本 {chunk_id - len(table_chunks) - len(image_chunks)} 个 / 表格 {len(table_chunks)} 个 / 图片 {len(image_chunks)} 个）")
        print(f"共处理 {len(cl_paths)} 个 content_list.json")

    def _table_to_text(self, table_item: Dict) -> str:
        """把 content_list 中的 table item 转为可被检索/嵌入的文本描述。"""
        parts = []
        caption = table_item.get('table_caption') or []
        if isinstance(caption, list) and caption:
            parts.append("表格说明：" + " ".join([str(c) for c in caption if c]))
        body = table_item.get('table_body') or ''
        if body:
            # 去掉 HTML 标签便于文本检索（保留换行）
            import re
            plain = re.sub(r'<[^>]+>', ' ', body)
            plain = re.sub(r'\s+', ' ', plain).strip()
            parts.append("表格内容：" + plain)
        footnote = table_item.get('table_footnote') or []
        if isinstance(footnote, list) and footnote:
            parts.append("表格脚注：" + " ".join([str(f) for f in footnote if f]))
        if not parts:
            parts.append("表格")
        return "\n".join(parts)

    def _image_to_text(self, image_item: Dict) -> str:
        """把 content_list 中的 image item 转为可被检索/嵌入的文本描述。"""
        parts = []
        caption = image_item.get('image_caption') or []
        if isinstance(caption, list) and caption:
            parts.append("图片说明：" + " ".join([str(c) for c in caption if c]))
        footnote = image_item.get('image_footnote') or []
        if isinstance(footnote, list) and footnote:
            parts.append("图片脚注：" + " ".join([str(f) for f in footnote if f]))
        if not parts:
            parts.append("图片")
        return "\n".join(parts)

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None, skip_file_names: Optional[set] = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        :param skip_file_names: 需要跳过的文件名（去扩展名，即 md 的 stem）集合，
               用于 content_list 路径已处理过的报告，避免被 markdown 估算页码重复覆盖
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_file_names = skip_file_names or set()
        processed_count = 0
        for md_path in all_md_paths:
            # content_list 路径已处理过的报告跳过，避免用估算页码覆盖真实页码
            if md_path.stem in skip_file_names:
                print(f"跳过（已有 content_list 分块）: {md_path.name}")
                continue
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # subset.csv 中查不到对应条目时跳过，避免产生 sha1 为空的脏分块文件，
            # 进而阻塞后续 create_vector_dbs（与 split_reports_from_content_list 的保护逻辑一致）
            if not sha1:
                print(f"警告：markdown 文件 {md_path.name} 在 subset.csv 中找不到对应条目，跳过该文件。如不需要请删除该 md 后重跑 chunk_reports。")
                continue
            chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
            output_json_path = output_dir / (md_path.stem + ".json")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name}")
            processed_count += 1
        print(f"共分割 {processed_count} 个 markdown 文件")
