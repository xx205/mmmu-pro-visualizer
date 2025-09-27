#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMMU-Pro 可视化（按 id 配对 · 不改题干 · 题干下按 options 逐项列出 · 占位锚点
· 10 选项/锚点 · 解析字符串化 options · 支持 <image k> 作为选项 · 元数据筛选
· MathJax · 原图落盘（多进程）· 增量渲染）

依赖：
  pip install -U datasets pillow pandas

示例：
  python visualize_mmmu_pro_pairs_idjoin_mathjax_mp_options_listed.py --rows 240 --page-size 60
  python visualize_mmmu_pro_pairs_idjoin_mathjax_mp_options_listed.py --subject History --workers 64
"""
import argparse
import ast
import html
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import load_dataset, Image as HFImage

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore


# --------------------- Utils ---------------------
def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def pick_image_columns(colnames: Sequence[str]) -> List[str]:
    cols: List[str] = []
    if "image" in colnames:
        cols.append("image")
    pat = re.compile(r"^image_\d+$", re.IGNORECASE)
    cols.extend([c for c in colnames if pat.match(c)])
    # 去重并保持顺序
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _decide_format(im: "Image.Image") -> Tuple[str, str]:
    has_alpha = (im.mode in ("RGBA", "LA")) or (im.mode == "P" and "transparency" in im.info)
    return ("PNG", ".png") if has_alpha else ("JPEG", ".jpg")


def worker_save_original(task: Tuple[str, str, bytes, str, int, bool, bool]) -> Tuple[str, str, int, int]:
    """子进程：将 bytes/path 的原图写盘为 JPEG/PNG（保留原尺寸）。"""
    (base_key, out_dir, img_bytes, img_path, jpeg_quality, jpeg_optimize, jpeg_progressive) = task
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    im = Image.open(BytesIO(img_bytes)) if img_bytes else Image.open(img_path)
    fmt, suffix = _decide_format(im)
    im = im.convert("RGB") if fmt == "JPEG" else im.convert("RGBA")
    w, h = im.size
    dst = out_dir_p / f"{base_key}{suffix}"
    try:
        if fmt == "JPEG":
            im.save(dst, fmt, quality=jpeg_quality, optimize=jpeg_optimize, progressive=jpeg_progressive)
        else:
            im.save(dst, fmt, optimize=True)
    except Exception:
        dst = out_dir_p / f"{base_key}.png"
        im.convert("RGBA").save(dst, "PNG", optimize=True)
    return base_key, f"images/{dst.name}", w, h


def parse_placeholders(text: str) -> List[int]:
    """抽取文本中的 <image k> 序列（按出现顺序，保留重复）。"""
    if not isinstance(text, str):
        return []
    return [int(m.group(1)) for m in re.finditer(r"<\s*image\s*(\d+)\s*>", text, flags=re.IGNORECASE)]


def normalize_options(options_field: Any) -> List[str]:
    """统一 options -> list[str]；兼容字符串化列表。"""
    if isinstance(options_field, list):
        return ["" if x is None else str(x) for x in options_field]
    if isinstance(options_field, str):
        s = options_field.strip()
        # 先尝试 literal_eval
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return ["" if x is None else str(x) for x in v]
        except Exception:
            pass
        # 回退：抽取引号内片段
        pattern = re.compile(r"""
            '([^']*)'   |   "([^"]*)"
        """, re.VERBOSE)
        parts = pattern.findall(s)
        flat: List[str] = []
        for a, b in parts:
            t = a if a else b
            if t != "":
                flat.append(t)
        return flat if flat else [s]
    return []


def placeholders_from_options(opts: List[str]) -> List[int]:
    ks: List[int] = []
    for t in opts:
        m = re.fullmatch(r"\s*<\s*image\s*(\d+)\s*>\s*", t, flags=re.IGNORECASE)
        if m:
            ks.append(int(m.group(1)))
    return ks


def extract_lettered_options_from_question(text: Any) -> Dict[str, str]:
    """从题干中解析 'A./A)/A:/' 等样式的选项文本，返回 {letter: text}。

    - 仅解析 A..J 共 10 项。
    - 支持分隔符：'.' '．' '。' ':' '：' '、' ')' '）' 及其组合，如 '(A) '、'A.) ' 等。
    - 文本范围：从标记结束到下一个标记开始（或字符串末尾）。
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    s = text

    # 在极少数数据中，选项可能出现在 “Options:”/“选项：” 后，
    # 这里不强制裁剪，而是通用地在全文中查找 A..J 标签。
    # 识别 label 的正则：前缀是开头或空白/分隔符，允许可选左括号，
    # 捕获 [A-J]，允许可选右括号/右方括号，后接常见的分隔符再空白。
    pat = re.compile(
        r"(?:^|[\s\[{(,，;；])(?:\(|\[)?([A-J])(?:\)|\])?\s*(?:[.．:：、)])\s*",
        flags=re.MULTILINE,
    )
    matches = list(pat.finditer(s))
    if not matches:
        return {}

    # 组装片段
    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        L = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        seg = s[start:end].strip()
        # 去掉开头冗余的标点/破折号等
        seg = re.sub(r"^[\s\-–—:：、.．。)*）]*", "", seg)
        # 清理末尾多余的空白
        seg = seg.strip()
        if L not in out:
            out[L] = seg
    return out


# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-split", default="test")
    ap.add_argument("--std-config", default="standard (10 options)")
    ap.add_argument("--std-split", default="test")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--page-size", type=int, default=60)
    ap.add_argument("--out", default="mmmu_pro_pairs_idjoin_mathjax_mp_options_listed")
    # 多进程与 JPEG 选项
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 8))
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--jpeg-optimize", action="store_true")
    ap.add_argument("--jpeg-progressive", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) 载入两个配置
    ds_v_all = load_dataset("MMMU/MMMU_Pro", "vision")
    assert args.vision_split in ds_v_all, f"Vision split not found. Avail={list(ds_v_all.keys())}"
    ds_v = ds_v_all[args.vision_split]

    ds_s_all = load_dataset("MMMU/MMMU_Pro", args.std_config)
    assert args.std_split in ds_s_all, f"Standard split not found. Avail={list(ds_s_all.keys())}"
    ds_s = ds_s_all[args.std_split]

    # 2) 图像列 decode=False（传 bytes/path 给子进程）
    v_img_cols = pick_image_columns(ds_v.column_names)
    s_img_cols = pick_image_columns(ds_s.column_names)
    for c in v_img_cols:
        ds_v = ds_v.cast_column(c, HFImage(decode=False))
    for c in s_img_cols:
        ds_s = ds_s.cast_column(c, HFImage(decode=False))

    # 3) 过滤 + to pandas
    def to_pd(ds, subject: Optional[str], keep_cols: Sequence[str], img_cols: Sequence[str]):
        rows = []
        for i in range(len(ds)):
            row = ds[i]
            if subject and str(row.get("subject", "")) != subject:
                continue
            rows.append(row)
        df = pd.DataFrame(rows)
        for k in keep_cols:
            if k not in df.columns:
                df[k] = None
        df["_img_cols"] = [list(img_cols)] * len(df)
        return df

    keep_v = ["id", "subject", "answer"] + v_img_cols
    keep_s = ["id", "subject", "question", "options", "answer", "explanation", "img_type", "topic_difficulty"] + s_img_cols
    df_v = to_pd(ds_v, args.subject, keep_v, v_img_cols)
    df_s = to_pd(ds_s, args.subject, keep_s, s_img_cols)
    if df_v.empty or df_s.empty:
        print("[!] One side is empty after filtering; nothing to visualize.")
        return

    # 4) 按 id 内连接并裁剪
    merged = pd.merge(df_v, df_s, on="id", how="inner", suffixes=("_v", "_s"))
    if args.start > 0:
        merged = merged.iloc[args.start:]
    if args.rows > 0:
        merged = merged.iloc[: args.rows]

    # 辅助：有重名列时加后缀
    overlap = (set(df_v.columns) & set(df_s.columns)) - {"id"}

    def mcol(side: str, col: str) -> str:
        return f"{col}_{side}" if col in overlap else col

    # 5) 多进程保存原图
    tasks: List[Tuple[str, str, bytes, str, int, bool, bool]] = []

    def add_tasks_for_row(row: pd.Series):
        rid = str(row.get("id"))
        # 左：vision
        for col in (row.get("_img_cols_v") or []):
            c = mcol("v", col)
            if c not in row:
                continue
            cell = row[c]
            base_key = f"v_{rid}_{col}"
            img_bytes, img_path = b"", ""
            if isinstance(cell, dict):
                img_bytes = cell.get("bytes") or b""
                img_path = cell.get("path") or ""
            elif isinstance(cell, str):
                img_path = cell
            else:
                continue
            tasks.append(
                (base_key, str(img_dir), img_bytes, img_path, int(args.jpeg_quality), bool(args.jpeg_optimize), bool(args.jpeg_progressive))
            )
        # 右：standard
        for col in (row.get("_img_cols_s") or []):
            c = mcol("s", col)
            if c not in row:
                continue
            cell = row[c]
            m = re.match(r"^image_(\d+)$", str(col), flags=re.IGNORECASE)
            idx = int(m.group(1)) if m else 1
            base_key = f"s_{rid}_image_{idx}"
            img_bytes, img_path = b"", ""
            if isinstance(cell, dict):
                img_bytes = cell.get("bytes") or b""
                img_path = cell.get("path") or ""
            elif isinstance(cell, str):
                img_path = cell
            else:
                continue
            tasks.append(
                (base_key, str(img_dir), img_bytes, img_path, int(args.jpeg_quality), bool(args.jpeg_optimize), bool(args.jpeg_progressive))
            )

    for _, row in merged.iterrows():
        add_tasks_for_row(row)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: Dict[str, Tuple[str, int, int]] = {}
    if tasks:
        print(f"[i] Saving {len(tasks)} images with {args.workers} processes ...")
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [ex.submit(worker_save_original, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                base_key, rel, w, h = f.result()
                results[base_key] = (rel, w, h)
                if i % 200 == 0:
                    print(f"  [..] {i}/{len(tasks)}")
        print(f"[✓] Images saved: {len(results)}")
    else:
        print("[i] No images to save.")

    # 6) 组装 pairs JSON
    pairs: List[Dict[str, Any]] = []
    letters = list("ABCDEFGHIJ")

    for _, r in merged.iterrows():
        rid = str(r.get("id"))
        subj_v = r.get("subject_v", r.get("subject", ""))
        subj_s = r.get("subject_s", r.get("subject", ""))

        # 左：vision
        left_imgs: List[Dict[str, Any]] = []
        for col in (r.get("_img_cols_v") or []):
            key = f"v_{rid}_{col}"
            if key in results:
                rel, w, h = results[key]
                left_imgs.append({"w": w, "h": h, "src": rel})
        left = {"id": rid, "subject": subj_v or subj_s, "answer": r.get("answer_v", ""), "images": left_imgs}

        # 右：收集图片全集
        right_imgs_all: Dict[int, Dict[str, Any]] = {}
        for col in (r.get("_img_cols_s") or []):
            m = re.match(r"^image_(\d+)$", str(col), flags=re.IGNORECASE)
            idx = int(m.group(1)) if m else 1
            key = f"s_{rid}_image_{idx}"
            if key in results:
                rel, w, h = results[key]
                right_imgs_all[idx] = {"w": w, "h": h, "src": rel}

        # 答案规范化
        ans_raw = str(r.get("answer_s", r.get("answer", "")) or "")
        answer_letter = re.sub(r"[.)\s]", "", ans_raw.strip()).upper()
        answer_text_norm = ans_raw.strip().lower()

        # 选项规范化
        # 注意：有些 split 两侧都含有 options 字段，merge 后将被重命名为 options_s；
        # 这里用 mcol('s', 'options') 取标准侧的列名。
        opt_col = mcol("s", "options")
        opts_raw = normalize_options(r.get(opt_col, r.get("options", [])))
        cleaned = [("" if x is None else str(x)).strip() for x in opts_raw]

        # 占位顺序：题干优先，否则按选项顺序
        occ_q = parse_placeholders(r.get("question", ""))
        occ_opts = placeholders_from_options(cleaned)
        occ_seq = occ_q if len(occ_q) > 0 else occ_opts

        # 组织锚点
        occ_imgs: List[Dict[str, Any]] = []
        warnings: List[str] = []
        k_to_first_id: Dict[int, str] = {}
        for i, k in enumerate(occ_seq, 1):
            it = right_imgs_all.get(k)
            fig_id = f"img-{rid}-k{k}-{i}"
            if it is None:
                warnings.append(f"占位 <image {k}> 无对应图片列 image_{k}")
                occ_imgs.append({"missing": True, "k": k, "id": fig_id})
            else:
                occ_imgs.append({"k": k, "src": it["src"], "w": it["w"], "h": it["h"], "id": fig_id})
                k_to_first_id.setdefault(k, fig_id)

        extra_imgs: List[Dict[str, Any]] = []
        if right_imgs_all:
            ref = set(occ_seq)
            for k, it in sorted(right_imgs_all.items()):
                if k not in ref:
                    fid = f"img-{rid}-x{k}"
                    extra_imgs.append({"k": k, "src": it["src"], "w": it["w"], "h": it["h"], "id": fid})
                    k_to_first_id.setdefault(k, fid)

        # 生成 options_items（后端版）
        def _as_letter_token(s: str) -> bool:
            t = re.sub(r"[.)\s]", "", s or "")
            return bool(re.fullmatch(r"[A-J]", t, flags=re.IGNORECASE))

        is_pure_letters = len(cleaned) > 0 and all(_as_letter_token(x) for x in cleaned)
        options_items: List[Dict[str, Any]] = []

        if is_pure_letters:
            # 题干选项仅为字母时，优先保持原始顺序：位置字母 + 该位置的内容（仍是字母）。
            # 若题干文本中能解析到 A:/B:/… 的正文，则替换为解析到的正文；
            # 锚点与高亮规则保持与其他分支一致。
            parsed_from_q = extract_lettered_options_from_question(r.get("question", ""))
            for i, tok in enumerate(cleaned):
                L = letters[i] if i < len(letters) else "?"
                # 默认文本为该位置给出的字母（例如 options=['B','A','C'] -> A. B）
                txt = parsed_from_q.get(L, "") or tok
                href: Optional[str] = None
                # 若文本中含有占位 <image k>，跳转到第一个锚点
                m_href = re.search(r"<\s*image\s*(\d+)\s*>", txt or "", flags=re.IGNORECASE)
                if m_href:
                    k = int(m_href.group(1))
                    if k in k_to_first_id:
                        href = f"#{k_to_first_id[k]}"
                # 若选项文本就是图片占位，文本可不重复显示
                text_for_view = "" if re.fullmatch(r"\s*<\s*image\s*(\d+)\s*>\s*", txt or "", flags=re.IGNORECASE) else txt
                is_corr = (bool(answer_letter) and (L == answer_letter)) or (
                    (not answer_letter) and answer_text_norm and (text_for_view or "").strip().lower() == answer_text_norm
                )
                options_items.append({"label": L, "text": text_for_view, "href": href, "is_correct": is_corr})
        else:
            # 文本/占位选项：按顺序 A..J 渲染；<image k> 绑定锚点
            for i, t in enumerate(cleaned):
                L = letters[i] if i < len(letters) else "?"
                href = None
                text = t
                # 如果完全是图片占位，则仅给锚点，不重复文字
                m_full = re.fullmatch(r"\s*<\s*image\s*(\d+)\s*>\s*", t, flags=re.IGNORECASE)
                if m_full:
                    k = int(m_full.group(1))
                    href = f"#{k_to_first_id.get(k, '')}" if k in k_to_first_id else None
                    text = ""  # 图片型选项不重复占位文本
                else:
                    # 若文本中包含占位，则跳到第一个锚点
                    m_part = re.search(r"<\s*image\s*(\d+)\s*>", t or "", flags=re.IGNORECASE)
                    if m_part:
                        k = int(m_part.group(1))
                        if k in k_to_first_id:
                            href = f"#{k_to_first_id[k]}"
                is_corr = (bool(answer_letter) and (L == answer_letter)) or (
                    (not answer_letter) and answer_text_norm and t.strip().lower() == answer_text_norm
                )
                options_items.append({"label": L, "text": text, "href": href, "is_correct": is_corr})

        if (answer_letter or answer_text_norm) and not any(it["is_correct"] for it in options_items):
            warnings.append(f"answer='{answer_letter or ans_raw}' 未能对齐到选项列表")

        right = {
            "id": rid,
            "subject": subj_s or subj_v,
            # 不改题干：优先标准侧，避免潜在重名
            "question": r.get(mcol("s", "question"), r.get("question", "")),
            "options_items": options_items,      # 后端解析的选项
            "options_raw_list": cleaned,         # ★ 前端兜底重建使用
            "answer": answer_letter or ans_raw,
            "explanation": r.get(mcol("s", "explanation"), r.get("explanation", "")),
            "img_type": r.get(mcol("s", "img_type"), r.get("img_type", "")),
            "topic_difficulty": r.get(mcol("s", "topic_difficulty"), r.get("topic_difficulty", "")),
            "occ_images": occ_imgs,
            "extra_images": extra_imgs,
            "warnings": warnings,
        }

        pairs.append({"left": left, "right": right})

    # 7) 过滤器选项
    subjects = sorted(set([(p["left"].get("subject") or "") for p in pairs] + [(p["right"].get("subject") or "") for p in pairs]))
    subjects_opts = "".join(f"<option value='{esc(s)}'>{esc(s) or 'All'}</option>" for s in (["All"] + subjects))
    img_types = sorted(set([p["right"].get("img_type") or "" for p in pairs]))
    imgtype_opts = "".join(f"<option value='{esc(s)}'>{esc(s) or 'All'}</option>" for s in (["All"] + img_types))
    diffs = sorted(set([p["right"].get("topic_difficulty") or "" for p in pairs]))
    diff_opts = "".join(f"<option value='{esc(s)}'>{esc(s) or 'All'}</option>" for s in (["All"] + diffs))

    data_json = json.dumps(pairs, ensure_ascii=False)

    # 8) HTML 模板
    html_tpl = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMMU-Pro · Vision vs Standard(10) — Anchors & MathJax (Multiprocess · Options Listed)</title>
<style>
  :root{ --gap:14px; --bg:#0b0b0c; --pane:#101012; --card:#141416; --fg:#e7e7ea; --muted:#9aa0a6; }
  *{ box-sizing:border-box; }
  body{ margin:0; padding:var(--gap); background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }
  header h1{ font-size:18px; margin:0 8px 0 0; }
  header select, header input{ padding:8px 10px; border-radius:10px; border:1px solid #2a2a2f; background:#0e0e10; color:var(--fg); }
  .pairs{ display:flex; flex-direction:column; gap:var(--gap); }
  .pair{ display:grid; grid-template-columns: 1fr 1fr; gap: var(--gap); align-items: stretch; }
  .cell{ display:flex; }
  .card{ background:var(--card); border:1px solid #222; border-radius:14px; overflow:hidden; display:flex; flex-direction:column; content-visibility:auto; contain-intrinsic-size: 600px; }
  .media{ display:flex; gap:8px; overflow:auto; padding:8px; background:#0e0e10; border-bottom:1px solid #222; }
  .media figure{ margin:0; display:flex; flex-direction:column; gap:6px; min-width:200px; position:relative; }
  .badge{ position:absolute; top:6px; left:6px; padding:2px 6px; font:12px/1 monospace; border-radius:10px; border:1px solid #333; background:#0b1220; color:#c7d2fe; }
  img{ width:100%; height:auto; object-fit:contain; border-radius:12px; }
  .chips{ display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 0; }
  .chip{ display:inline-flex; align-items:center; gap:6px; padding:2px 8px; border-radius:999px; background:#0e1a2a; color:#c7d2fe; border:1px solid #223; font-size:12px; text-decoration:none; }
  .info{ padding:10px; display:flex; flex-direction:column; gap:8px; }
  .qid{ color:#9aa0a6; font-size:12px; }
  .q{ white-space:pre-wrap; }
  .opts-title{ margin-top:6px; color:#cbd5e1; font-weight:600; }
  /* 选项列表不再显示浏览器自带 1. 2. 编号 */
  .opts{ margin:4px 0 0 12px; padding-left:0; list-style:none; }
  .opts li{ margin:2px 0; }
  .opt-tag{ display:inline-block; min-width:1.2rem; text-align:center; margin-right:6px; padding:1px 6px; border-radius:6px; background:#1b2838; color:#bfdbfe; border:1px solid #274; font:12px/1 monospace; text-decoration:none; }
  .opt-correct{ background:#063; color:#d1fae5; border-color:#0b6; }
  .opt-jump{ text-decoration:none; }
  .ans mark{ background:#2563eb33; color:#dbeafe; padding:2px 6px; border-radius:8px; }
  .warnbar{ background:#2a1a0a; color:#fde68a; padding:6px 10px; border-top:1px solid #3b2a0b; }
  .hidden{ display:none !important; }
  #sentinel{ height:1px; }
  .lightbox{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; background:rgba(0,0,0,.85); z-index:9999; }
  .lightbox.open{ display:flex; }
  .lightbox img{ max-width:96vw; max-height:96vh; box-shadow:0 0 0 1px #333, 0 10px 40px rgba(0,0,0,.6); }
</style>
</head>
<body>
<header>
  <h1>MMMU-Pro · Vision vs Standard(10) — Anchors & MathJax (Multiprocess · Options Listed)</h1>
  <input id="q" placeholder="搜索（全文）">
  <label>Subject <select id="f_subject">__SUBJECT_OPTIONS__</select></label>
  <label>Image Type <select id="f_imgtype">__IMGTYPE_OPTIONS__</select></label>
  <label>Difficulty <select id="f_diff">__DIFF_OPTIONS__</select></label>
</header>

<section id="pairs" class="pairs"></section>
<div id="sentinel"></div>
<div id="lightbox" class="lightbox" aria-hidden="true"><img alt=""></div>

<script id="data" type="application/json">__DATA__</script>
<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\(","\\)"]]}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

<script>
  const PAGE_SIZE = __PAGE_SIZE__;
  const data = JSON.parse(document.getElementById('data').textContent);
  const container = document.getElementById('pairs');
  const q = document.getElementById('q');
  const fSubject = document.getElementById('f_subject');
  const fImgType = document.getElementById('f_imgtype');
  const fDiff = document.getElementById('f_diff');
  const lightbox = document.getElementById('lightbox');
  let page = 0, fullyRendered = false;

  function buildImgFigure(img, badge){
    const fig = document.createElement('figure');
    if (badge){ const b=document.createElement('span'); b.className='badge'; b.textContent=badge; fig.appendChild(b); }
    if (img && img.src){
      const im=document.createElement('img'); im.src=img.src;
      if (img.w && img.h){ im.width=img.w; im.height=img.h; }
      im.loading='lazy'; im.decoding='async';
      im.addEventListener('click', ()=>{ lightbox.classList.add('open'); lightbox.querySelector('img').src=im.src; });
      fig.appendChild(im);
    }else{
      const d=document.createElement('div'); d.className='warnbar'; d.textContent='（无图片）'; fig.appendChild(d);
    }
    return fig;
  }

  function cardLeft(L){
    const art=document.createElement('article'); art.className='card'; art.dataset.subject=L.subject||'';
    const media=document.createElement('div'); media.className='media';
    if(Array.isArray(L.images)&&L.images.length){ L.images.forEach((img,i)=> media.appendChild(buildImgFigure(img, `Vision ${i+1}`))); }
    else{ const d=document.createElement('div'); d.className='warnbar'; d.textContent='（无图片）'; media.appendChild(d); }
    const info=document.createElement('div'); info.className='info';
    const meta=document.createElement('div'); meta.className='meta'; meta.innerHTML=`Subject: <b>${L.subject||''}</b>`;
    const qid=document.createElement('div'); qid.className='qid'; qid.textContent=`ID: ${L.id||''}`;
    const qv=document.createElement('div'); qv.className='q'; qv.textContent='(Vision-only) The question is embedded in the image.';
    const ans=document.createElement('div'); ans.className='ans'; ans.innerHTML=`Answer: <mark>${L.answer||''}</mark>`;
    info.append(meta,qid,qv,ans); art.append(media,info); return art;
  }

  function buildChips(occ){
    const nav=document.createElement('div'); nav.className='chips';
    if(!occ||!occ.length){ const s=document.createElement('span'); s.className='chip'; s.textContent='No <image k> placeholder'; nav.appendChild(s); return nav; }
    occ.forEach(o=>{ const a=document.createElement('a'); a.className='chip'; a.href='#'+o.id; a.textContent=`Image ${o.k}`; nav.appendChild(a); });
    return nav;
  }

  function cardRight(R){
    const art=document.createElement('article'); art.className='card';
    art.dataset.subject=R.subject||''; art.dataset.imgtype=R.img_type||''; art.dataset.diff=R.topic_difficulty||'';

    const media=document.createElement('div'); media.className='media';
    if(R.occ_images&&R.occ_images.length){ R.occ_images.forEach(o=>{ const fig=buildImgFigure(o,`Image ${o.k}`); if(o.id) fig.id=o.id; media.appendChild(fig); }); }
    if(R.extra_images&&R.extra_images.length){ R.extra_images.forEach(o=>{ const fig=buildImgFigure(o,`Extra ${o.k}`); if(o.id) fig.id=o.id; media.appendChild(fig); }); }
    if(!media.children.length){ const d=document.createElement('div'); d.className='warnbar'; d.textContent='（无图片）'; media.appendChild(d); }

    const info=document.createElement('div'); info.className='info';
    const meta=document.createElement('div'); meta.className='meta';
    meta.innerHTML=`Subject: <b>${R.subject||''}</b> · <small>${R.img_type||''}</small> · <small>${R.topic_difficulty||''}</small>`;
    const qid=document.createElement('div'); qid.className='qid'; qid.textContent=`ID: ${R.id||''}`;
    const chips=buildChips(R.occ_images);

    const qdiv=document.createElement('div'); qdiv.className='q'; qdiv.textContent=R.question||'';

    // **题干后列出选项**
    // 优先使用后端解析的 options_items；为空则回退到 options_raw_list；
    // 若仍为空，尝试从题干中解析 A..J 选项。
    const title=document.createElement('div'); title.className='opts-title'; title.textContent='Options:';
    // 使用 ul 而不是 ol，去除 1. 2. 3. 级别编号
    const opts=document.createElement('ul'); opts.className='opts';

    let items = Array.isArray(R.options_items) ? R.options_items.slice() : [];
    if (!items.length) {
      const raw = Array.isArray(R.options_raw_list) ? R.options_raw_list : [];
      const kToId = {};
      (R.occ_images||[]).forEach(o=>{ if(o&&o.k&&o.id) kToId[o.k]=o.id; });
      (R.extra_images||[]).forEach(o=>{ if(o&&o.k&&o.id && !kToId[o.k]) kToId[o.k]=o.id; });
      const normAns = (R.answer||'').toString().trim().toUpperCase().replace(/[.)\s]/g,'');
      const ansText = (R.answer||'').toString().trim().toLowerCase();
      for (let i=0;i<raw.length && i<10;i++){
        const L = String.fromCharCode(65+i);
        const t = (raw[i]??'').toString().trim();
        let href=null, text=t;
        const m = t.match(/^\s*<\s*image\s*(\d+)\s*>\s*$/i);
        if (m){ const k=parseInt(m[1],10); if(kToId[k]) href='#'+kToId[k]; text=''; }
        // 高亮规则：若答案是字母，仅按标签字母匹配；
        // 仅当答案不是字母时，才按文本匹配。
        const is_correct = (normAns && L===normAns) || (!normAns && t.toLowerCase()===ansText);
      items.push({label:L, text, href, is_correct});
      }
    }

    // 第三级兜底：从题干中抽取 "A./A)/A: …" 的选项
    if (!items.length && (R.question||'')){
      const s = (R.question||'').toString();
      const pat = /(^|[\s\[{(,，;；])(?:\(|\[)?([A-J])(?:\)|\])?\s*(?:[.．:：、)])\s*/g;
      let m, positions=[];
      while ((m=pat.exec(s))){ positions.push({L:m[2].toUpperCase(), from:m.index, end:m.index+m[0].length}); }
      if (positions.length){
        const kToId = {}; (R.occ_images||[]).forEach(o=>{ if(o&&o.k&&o.id) kToId[o.k]=o.id; }); (R.extra_images||[]).forEach(o=>{ if(o&&o.k&&o.id && !kToId[o.k]) kToId[o.k]=o.id; });
        const normAns=(R.answer||'').toString().trim().toUpperCase().replace(/[.)\s]/g,'');
        const ansText=(R.answer||'').toString().trim().toLowerCase();
        for (let i=0;i<positions.length;i++){
          const cur=positions[i], next=positions[i+1];
          let seg = s.slice(cur.end, next?next.from:s.length).trim();
          seg = seg.replace(/^[\s\-–—:：、.．。)*）]*/,'').trim();
          let href=null, text=seg;
          const mh = seg.match(/<\s*image\s*(\d+)\s*>/i);
          if(mh){ const k=parseInt(mh[1],10); if(kToId[k]) href='#'+kToId[k]; }
          if(/^\s*<\s*image\s*(\d+)\s*>\s*$/i.test(seg)) text='';
          const L=cur.L; const is_correct=(normAns && L===normAns) || (!normAns && seg.toLowerCase()===ansText);
          items.push({label:L, text, href, is_correct});
        }
      }
    }

    // 纯字母 options 的特殊展示：
    // 若 items 仍为空（或 items 只有 label 无正文），则回退为“位置字母 + 该位置原始内容”。
    if ((!items || !items.length) && Array.isArray(R.options_raw_list)){
      const raw = R.options_raw_list;
      const normAns=(R.answer||'').toString().trim().toUpperCase().replace(/[.)\s]/g,'');
      const ansText=(R.answer||'').toString().trim().toLowerCase();
      for(let i=0;i<raw.length && i<10;i++){
        const L=String.fromCharCode(65+i);
        const t=(raw[i]??'').toString().trim();
        const is_correct=(normAns && L===normAns) || (!normAns && t.toLowerCase()===ansText);
        items.push({label:L, text:t, href:null, is_correct});
      }
    }

    for(const it of items){
      const li=document.createElement('li');
      const tag=document.createElement('span'); tag.className='opt-tag'+(it.is_correct?' opt-correct':''); tag.textContent=it.label||'?';
      if (it.href){ const a=document.createElement('a'); a.href=it.href; a.className='opt-jump'; a.appendChild(tag); li.appendChild(a); }
      else { li.appendChild(tag); }
      if (it.text){ const code=document.createElement('code'); code.textContent=it.text; li.appendChild(code); }
      opts.appendChild(li);
    }

    const ans=document.createElement('div'); ans.className='ans'; ans.innerHTML=`Answer: <mark>${R.answer||''}</mark>`;

    info.append(meta,qid,chips,qdiv,title,opts,ans);
    if(R.warnings&&R.warnings.length){ const wb=document.createElement('div'); wb.className='warnbar'; wb.textContent='注意：'+R.warnings.join('；'); info.appendChild(wb); }
    art.append(media,info); return art;
  }

  function buildPairRow(pair){
    const row=document.createElement('div'); row.className='pair';
    row.dataset.subject=(pair.right?.subject||pair.left?.subject||'');
    row.dataset.imgtype=(pair.right?.img_type||'');
    row.dataset.diff=(pair.right?.topic_difficulty||'');
    const cellL=document.createElement('div'); cellL.className='cell left';
    const cellR=document.createElement('div'); cellR.className='cell right';
    cellL.appendChild(cardLeft(pair.left||{}));
    cellR.appendChild(cardRight(pair.right||{}));
    row.append(cellL,cellR); return row;
  }

  function renderNextPage(){
    if(fullyRendered) return;
    const from=page*PAGE_SIZE, to=Math.min(data.length, from+PAGE_SIZE);
    for(let i=from;i<to;i++){ container.appendChild(buildPairRow(data[i])); }
    page++; if(page*PAGE_SIZE>=data.length) fullyRendered=true;
    if(from===0){ [...container.querySelectorAll('img')].slice(0,6).forEach(img=>{ img.setAttribute('fetchpriority','high'); img.removeAttribute('loading'); }); }
    if(window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
  }

  function applyFilters(){
    const s=(q.value||'').trim().toLowerCase();
    const subj=fSubject.value, tp=fImgType.value, df=fDiff.value;
    if(s||subj!=='All'||tp!=='All'||df!=='All'){ if(!fullyRendered){ while(!fullyRendered) renderNextPage(); } }
    const rows=[...container.querySelectorAll('.pair')];
    for(const row of rows){
      const textOK=!s || row.textContent.toLowerCase().includes(s);
      const subjOK=(subj==='All')||(row.dataset.subject===subj);
      const typeOK=(tp==='All')||(row.dataset.imgtype===tp);
      const diffOK=(df==='All')||(row.dataset.diff===df);
      row.classList.toggle('hidden', !(textOK&&subjOK&&typeOK&&diffOK));
    }
    if(window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
  }

  const sentinel=document.getElementById('sentinel');
  const io=new IntersectionObserver((entries)=>{ if(entries.some(e=>e.isIntersecting)){ if(!q.value&&fSubject.value==='All'&&fImgType.value==='All'&&fDiff.value==='All') renderNextPage(); }},{ rootMargin:'1200px 0px' });
  io.observe(sentinel);

  q.addEventListener('input', applyFilters);
  fSubject.addEventListener('change', applyFilters);
  fImgType.addEventListener('change', applyFilters);
  fDiff.addEventListener('change', applyFilters);
  lightbox.addEventListener('click', ()=>{ lightbox.classList.remove('open'); lightbox.querySelector('img').removeAttribute('src'); });
  window.addEventListener('keydown', e=>{ if(e.key==='Escape') lightbox.click(); });

  renderNextPage();
</script>
</body>
</html>
"""
    html_doc = (
        html_tpl.replace("__SUBJECT_OPTIONS__", subjects_opts)
        .replace("__IMGTYPE_OPTIONS__", imgtype_opts)
        .replace("__DIFF_OPTIONS__", diff_opts)
        .replace("__DATA__", data_json)
        .replace("__PAGE_SIZE__", str(max(1, int(args.page_size))))
    )

    (out / "index.html").write_text(html_doc, encoding="utf-8")
    print("[✓] Wrote:", (out / "index.html").resolve())


if __name__ == "__main__":
    if Image is None:
        raise SystemExit("Pillow 未安装或导入失败：请先 `pip install pillow`")
    main()
