# utils/text_processor.py
"""
文本处理工具：加载各种格式文档、分块
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """
    从文件夹递归加载所有文档，返回文档列表。
    每个文档是一个字典：{"text": 内容, "source": 文件路径}
    支持 .txt, .md, .pdf, .docx
    """
    documents = []
    folder = Path(folder_path)
    
    for file_path in folder.glob("**/*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        
        text = None
        if suffix in [".txt", ".md"]:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = _read_pdf(file_path)
        elif suffix == ".docx":
            text = _read_docx(file_path)
        else:
            continue
        
        if text and text.strip():
            documents.append({
                "text": text,
                "source": str(file_path)
            })
            print(f"   加载: {file_path.name} ({len(text)} 字符)")
    
    return documents

def _read_pdf(file_path: Path) -> str:
    """读取 PDF 文件（需要安装 pypdf）"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except ImportError:
        print(f"   ⚠️ 未安装 pypdf，跳过 PDF: {file_path.name}")
        return ""
    except Exception as e:
        print(f"   ⚠️ 读取 PDF 失败 {file_path.name}: {e}")
        return ""

def _read_docx(file_path: Path) -> str:
    """读取 Word 文档（需要安装 python-docx）"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        print(f"   ⚠️ 未安装 python-docx，跳过 DOCX: {file_path.name}")
        return ""
    except Exception as e:
        print(f"   ⚠️ 读取 DOCX 失败 {file_path.name}: {e}")
        return ""

def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 512,
    overlap: int = 50
) -> Tuple[List[str], List[Dict]]:
    """
    将文档列表分块。
    返回:
        chunks: 文本块列表
        metadatas: 每个块对应的元数据列表
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    all_chunks = []
    all_metadatas = []
    
    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        # 使用 langchain 分块
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": source})
    
    return all_chunks, all_metadatas