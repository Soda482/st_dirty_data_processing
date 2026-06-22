"""
脏数据处理工具 - Dirty Data Cleaner
基于 Streamlit 的交互式数据清洗 Web 应用
"""

import io
import base64
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import chardet
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# 配置中文字体支持（Windows 下使用微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="脏数据处理工具",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 主题切换
# ============================================================
def apply_theme(mode: str):
    """自定义 light / dark 主题样式"""
    if mode == "dark":
        bg = "#0D0D0D"
        panel = "#1A1A1A"
        panel_hover = "#252525"
        text = "#FFFFFF"
        text_secondary = "#E0E0E0"
        text_muted = "#A0A0A0"
        accent = "#FF6B6B"
        border = "#333333"
        success = "#4CAF50"
        warning = "#FF9800"
        error = "#F44336"
    else:
        bg = "#FFFFFF"
        panel = "#F5F5F5"
        panel_hover = "#EEEEEE"
        text = "#1A1A1A"
        text_secondary = "#424242"
        text_muted = "#757575"
        accent = "#FF4B4B"
        border = "#E0E0E0"
        success = "#4CAF50"
        warning = "#FF9800"
        error = "#F44336"

    st.markdown(
        f"""
        <style>
        /* 全局背景和文字 */
        .stApp {{ background-color: {bg}; color: {text}; }}
        
        /* 侧边栏 */
        .css-1d391kg {{ background-color: {panel} !important; }}
        .css-1d391kg p, .css-1d391kg li, .css-1d391kg h3, .css-1d391kg span {{ color: {text_secondary} !important; }}
        
        /* 页面标题 */
        h1, h2, h3, h4, h5, h6 {{ color: {text} !important; }}
        
        /* 段落文字 */
        p, li, span, label {{ color: {text_secondary} !important; }}
        
        /* 大数字 */
        .big-number {{
            font-size: 32px; font-weight: 700; color: {accent};
            text-align: center; padding: 10px;
        }}
        
        /* 指标标签 */
        .metric-label {{ text-align: center; color: {text_muted}; font-size: 14px; }}
        
        /* 章节标题 */
        .section-title {{
            font-size: 22px; font-weight: 600;
            margin: 20px 0 10px 0; color: {accent};
            border-bottom: 2px solid {border}; padding-bottom: 6px;
        }}
        
        /* 信息卡片 */
        .info-card {{
            background-color: {panel}; border-radius: 8px;
            padding: 14px; margin: 6px 0;
        }}
        
        /* 标签样式 */
        .tag-ok {{
            display:inline-block; background:{success}; color:white;
            padding:2px 10px; border-radius:12px; font-size:12px; margin-right:6px;
        }}
        .tag-warn {{
            display:inline-block; background:{warning}; color:white;
            padding:2px 10px; border-radius:12px; font-size:12px; margin-right:6px;
        }}
        .tag-bad {{
            display:inline-block; background:{error}; color:white;
            padding:2px 10px; border-radius:12px; font-size:12px; margin-right:6px;
        }}
        
        /* 数据表格 */
        .dataframe {{ background-color: {panel} !important; }}
        .dataframe th {{ background-color: {panel_hover} !important; color: {text} !important; border-color: {border} !important; }}
        .dataframe td {{ color: {text_secondary} !important; border-color: {border} !important; }}
        
        /* Tab 标签 */
        .css-15tx938 {{ color: {text_muted} !important; }}
        .css-15tx938[aria-selected="true"] {{ color: {text} !important; background-color: {panel_hover} !important; }}
        
        /* 按钮 */
        .stButton>button {{ color: {text} !important; background-color: {panel} !important; border-color: {border} !important; }}
        .stButton>button:hover {{ background-color: {panel_hover} !important; }}
        .stButton>button[type="primary"] {{ background-color: {accent} !important; color: white !important; }}
        
        /* 选择框 */
        .stSelectbox label, .stRadio label, .stCheckbox label {{ color: {text_secondary} !important; }}
        .css-13sdm1b {{ background-color: {panel} !important; color: {text} !important; border-color: {border} !important; }}
        
        /* 输入框 */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {{ color: {text} !important; background-color: {panel} !important; }}
        .stTextInput label, .stNumberInput label {{ color: {text_secondary} !important; }}
        
        /* 滑块 */
        .css-1iyq7zh {{ color: {text_muted} !important; }}
        
        /* 分割线 */
        .css-1r4sjp {{ background-color: {border} !important; }}
        
        /* 警告/成功/错误信息 */
        .stAlert {{ background-color: {panel} !important; border-color: {border} !important; }}
        .stAlert p {{ color: {text_secondary} !important; }}
        
        /* 进度条 */
        .stProgress>div>div {{ background-color: {accent} !important; }}
        
        /* 文件上传 */
        .stFileUploader label {{ color: {text_secondary} !important; }}
        .css-1cypcdb {{ border-color: {border} !important; }}
        
        /* 时间显示 */
        .css-12w0qpk {{ color: {text_muted} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.title("🧹 脏数据处理工具")
    st.caption("Dirty Data Cleaner v1.0")
    theme_mode = st.radio(
        "🎨 显示主题",
        ["Light 🌞", "Dark 🌙"],
        index=0,
        horizontal=True,
        help="切换 Light / Dark 显示模式",
    )
    apply_theme("dark" if "Dark" in theme_mode else "light")

    st.divider()
    st.markdown("**📖 使用说明**")
    st.markdown(
        "1. 上传 CSV / Excel 文件\n"
        "2. 查看数据诊断报告\n"
        "3. 进行清洗操作\n"
        "4. 对比前后变化并导出"
    )


# ============================================================
# 工具函数
# ============================================================
def detect_encoding(raw_bytes: bytes) -> str:
    """自动检测文件编码"""
    result = chardet.detect(raw_bytes[: min(50000, len(raw_bytes))])
    enc = result.get("encoding") or "utf-8"
    # 兼容常见中文编码
    if enc and enc.lower() in ("ascii", "gb2312"):
        return "gbk"
    return enc


def load_file(file) -> pd.DataFrame:
    """根据后缀加载文件"""
    name = file.name.lower()
    raw = file.read()

    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            enc = detect_encoding(raw)
            # 先尝试自动检测，失败则回退到 utf-8 / gbk
            for encoding in (enc, "utf-8", "gbk", "utf-8-sig", "latin-1"):
                if encoding is None:
                    continue
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                    return df
                except Exception:
                    continue
            raise ValueError("无法解析 CSV 文件编码")

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        else:
            raise ValueError("暂不支持的文件格式，仅支持 CSV / TXT / XLSX / XLS")
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        return pd.DataFrame()


def build_sample_dirty_data() -> pd.DataFrame:
    """生成示例脏数据，便于用户无需上传即可体验"""
    np.random.seed(42)
    n = 200
    names = ["张三", "李四", "王五", "赵六", "钱七", "孙八"]
    cities = ["北京", "上海", "广州", "深圳", "杭州", " 成都", "南京"]

    df = pd.DataFrame(
        {
            "姓名": np.random.choice(names, n),
            "年龄": np.random.choice(
                [18, 22, 25, 30, 45, 999, -1, 200, np.nan, 28], n
            ),
            "性别": np.random.choice(["男", "女", "M", "F", "Male", " ", "男 "], n),
            "城市": np.random.choice(cities, n),
            "月薪": np.random.choice(
                [5000, 8000, 12000, 15000, 20000, np.nan, 999999, 0, 3000], n
            ),
            "入职日期": pd.to_datetime(
                np.random.choice(
                    pd.date_range("2018-01-01", "2024-12-31").astype(str).tolist()
                    + ["2023/13/40", "invalid", ""],
                    n,
                ),
                errors="coerce",
            ),
            "部门": np.random.choice(
                ["技术部", "市场部", "技术部 ", "销售部", "  人事部", "财务部", np.nan], n
            ),
            "绩效": np.random.choice(["A", "B", "C", "D", "a", "b", "优秀", " ", np.nan], n),
        }
    )
    # 故意加入重复行
    dup = df.sample(15, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    return df


# ============================================================
# 诊断模块
# ============================================================
def diagnose(df: pd.DataFrame) -> dict:
    """对数据进行全面诊断，返回结果字典"""
    diag = {}

    diag["n_rows"] = len(df)
    diag["n_cols"] = len(df.columns)
    diag["total_cells"] = diag["n_rows"] * diag["n_cols"]

    # 缺失值统计
    missing = df.isna().sum()
    diag["missing_total"] = int(missing.sum())
    diag["missing_by_col"] = missing.to_dict()
    diag["missing_pct"] = (
        (missing / diag["n_rows"] * 100).round(2).to_dict()
        if diag["n_rows"]
        else {}
    )

    # 重复行
    dup_mask = df.duplicated(keep=False)
    diag["duplicate_rows"] = int(dup_mask.sum())
    diag["duplicate_unique_groups"] = int(
        df[dup_mask].drop_duplicates().shape[0] if dup_mask.any() else 0
    )

    # 全空格字符串（空白但非 NaN）
    whitespace = {}
    for col in df.columns:
        if df[col].dtype == object:
            n = df[col].apply(
                lambda x: isinstance(x, str) and x.strip() == ""
            ).sum()
            if n > 0:
                whitespace[col] = int(n)
    diag["whitespace_cells"] = whitespace

    # 每列的数据类型与基本统计
    col_stats = {}
    numeric_cols = []
    for col in df.columns:
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "unique": int(s.nunique(dropna=True)),
            "missing": int(s.isna().sum()),
        }
        
        # 检测单列重复（出现次数 >= 2 的值，不算缺失值）
        value_counts = s.value_counts(dropna=True)
        duplicate_values = value_counts[value_counts >= 2]
        info["duplicate_count"] = int(duplicate_values.sum() - len(duplicate_values))  # 减去每个值的第一次出现
        
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            info["min"] = float(s.min()) if not s.isna().all() else None
            info["max"] = float(s.max()) if not s.isna().all() else None
            info["mean"] = float(s.mean()) if not s.isna().all() else None
            # IQR 异常值
            if not s.isna().all() and s.std() > 0:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((s < low) | (s > high)).sum())
            else:
                outliers = 0
            info["outliers_iqr"] = outliers
            # 计算该列总异常数（缺失值 + IQR异常值）
            info["total_anomalies"] = info["missing"] + outliers
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["min_date"] = s.min()
            info["max_date"] = s.max()
            # 无法解析的日期（转为 NaT）通常代表原数据中混杂了非法日期
            info["invalid_date"] = int(s.isna().sum())
            # 计算该列总异常数（缺失值/无效日期）
            info["total_anomalies"] = info["missing"]
        else:
            # 文本字段：检查前后空格 / 大小写不一致
            strip_issue = int(
                s.apply(lambda x: isinstance(x, str) and x != x.strip()).sum()
            )
            info["strip_issue"] = strip_issue
            info["mixed_case"] = int(
                s.apply(
                    lambda x: isinstance(x, str)
                    and x.lower() != x
                    and x.upper() != x
                ).sum()
            )
            # 检测低频值（出现次数 <= 2 的值，可能是拼写错误或异常值）
            rare_values = value_counts[value_counts <= 2]
            info["rare_values_count"] = int(rare_values.sum())
            if len(rare_values) > 0:
                info["rare_values"] = rare_values.to_dict()
            # 计算该列总异常数（缺失值 + 低频值）
            info["total_anomalies"] = info["missing"] + info["rare_values_count"]
        col_stats[col] = info
    diag["col_stats"] = col_stats
    diag["numeric_cols"] = numeric_cols

    # 数据类型错误检测：看起来像数字但存为字符串
    number_like_cols = {}
    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_valid = coerced.notna().sum()
        if n_valid > 0 and n_valid >= len(df[col].dropna()) * 0.5:
            number_like_cols[col] = int(n_valid)
    diag["number_like_cols"] = number_like_cols

    return diag


# ============================================================
# 主流程
# ============================================================
st.header("🧹 脏数据处理工具")
st.caption("上传 / 诊断 / 清洗 / 可视化 / 导出 一站式完成")

# ---------- 步骤 1：数据加载 ----------
st.markdown('<p class="section-title">步骤 1 · 加载数据</p>', unsafe_allow_html=True)

upload_mode = st.radio(
    "数据来源",
    ["📁 上传文件", "🎲 使用内置示例脏数据"],
    horizontal=True,
)

if upload_mode.startswith("📁"):
    uploaded_file = st.file_uploader(
        "拖拽或选择 CSV / TXT / XLSX / XLS 文件",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None:
        df_original = load_file(uploaded_file)
        if not df_original.empty:
            st.success(f"成功加载文件：{uploaded_file.name}（{df_original.shape[0]} 行 × {df_original.shape[1]} 列）")
            st.session_state["df_original"] = df_original
            st.session_state["file_name"] = uploaded_file.name
    else:
        st.info("请上传一个文件，或选择使用示例数据体验完整功能。")
        st.stop()
else:
    df_original = build_sample_dirty_data()
    st.success(f"已载入示例脏数据：{df_original.shape[0]} 行 × {df_original.shape[1]} 列")
    st.session_state["df_original"] = df_original
    st.session_state["file_name"] = "sample_dirty_data.csv"

if "df_original" not in st.session_state or st.session_state["df_original"].empty:
    st.stop()

df_original = st.session_state["df_original"]

# 初始化清洗中的数据
if "df_clean" not in st.session_state or st.session_state.get("df_original_shape") != df_original.shape:
    st.session_state["df_clean"] = df_original.copy()
    st.session_state["df_original_shape"] = df_original.shape
    st.session_state["clean_log"] = []

df_clean = st.session_state["df_clean"]
clean_log = st.session_state["clean_log"]

# ---------- 步骤 2：数据预览 & 诊断 ----------
st.markdown('<p class="section-title">步骤 2 · 数据预览与诊断</p>', unsafe_allow_html=True)

tab_prev, tab_diag = st.tabs(["📋 数据预览", "🔍 诊断报告"])

with tab_prev:
    c1, c2 = st.columns([3, 1])
    with c1:
        show_rows = st.slider("预览行数", 5, min(100, len(df_clean)), 10)
        st.dataframe(df_clean.head(show_rows), use_container_width=True)
    with c2:
        st.markdown("**📊 基本信息**")
        info_df = pd.DataFrame(
            {
                "列名": df_clean.columns,
                "数据类型": [str(df_clean[c].dtype) for c in df_clean.columns],
                "非空值": [int(df_clean[c].notna().sum()) for c in df_clean.columns],
                "缺失值": [int(df_clean[c].isna().sum()) for c in df_clean.columns],
                "唯一值": [int(df_clean[c].nunique(dropna=True)) for c in df_clean.columns],
            }
        )
        st.dataframe(info_df, use_container_width=True, hide_index=True)

with tab_diag:
    diag = diagnose(df_clean)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(
            f'<div class="big-number">{diag["n_rows"]}</div><div class="metric-label">总数据行数</div>',
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            f'<div class="big-number">{diag["missing_total"]}</div><div class="metric-label">缺失单元格</div>',
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            f'<div class="big-number">{diag["duplicate_rows"]}</div><div class="metric-label">重复行</div>',
            unsafe_allow_html=True,
        )
    with metric_cols[3]:
        total_anomalies = sum(
            s.get("total_anomalies", 0) for s in diag["col_stats"].values()
        )
        st.markdown(
            f'<div class="big-number">{total_anomalies}</div><div class="metric-label">总异常数（含缺失）</div>',
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**列级缺失值热力图**")
        missing_df = pd.DataFrame(
            {col: df_clean[col].isna().astype(int) for col in df_clean.columns}
        )
        # 采样显示避免图片过大
        sample_n = min(len(missing_df), 500)
        if len(missing_df) > sample_n:
            missing_df = missing_df.sample(sample_n, random_state=0)
        
        # 使用 plotly 绘制热力图，更好地支持中文
        fig_hm = px.imshow(
            missing_df.T,
            labels=dict(x="行（采样）", y="列", color="是否缺失"),
            color_continuous_scale=[[0, "#F0F2F6"], [1, "#FF4B4B"]],
            aspect="auto",
            height=max(300, len(df_clean.columns) * 25),
        )
        fig_hm.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
            title_text=f"缺失值分布（红色 = 缺失）| 采样 {sample_n} 行",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_b:
        st.markdown("**各列缺失比例**")
        miss_pct = pd.Series(diag["missing_pct"]).sort_values(ascending=False)
        miss_pct = miss_pct[miss_pct > 0]
        if miss_pct.empty:
            st.info("👍 当前数据无缺失值")
        else:
            fig_mp = px.bar(
                x=miss_pct.values,
                y=miss_pct.index,
                orientation="h",
                labels={"x": "缺失率 (%)", "y": "列"},
                text=miss_pct.values,
                color=miss_pct.values,
                color_continuous_scale="Reds",
                height=max(300, 40 * len(miss_pct)),
            )
            fig_mp.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_mp, use_container_width=True)

    st.markdown("**📋 详细列级诊断**")
    rows = []
    for col, info in diag["col_stats"].items():
        tags = ""
        if info.get("missing", 0) > 0:
            tags += f'<span class="tag-warn">缺失 {info["missing"]}</span>'
        if info.get("outliers_iqr", 0) > 0:
            tags += f'<span class="tag-bad">异常 {info["outliers_iqr"]}</span>'
        if info.get("rare_values_count", 0) > 0:
            tags += f'<span class="tag-warn">低频 {info["rare_values_count"]}</span>'
        if info.get("duplicate_count", 0) > 0:
            tags += f'<span class="tag-warn">重复 {info["duplicate_count"]}</span>'
        if info.get("strip_issue", 0) > 0:
            tags += f'<span class="tag-warn">空格 {info["strip_issue"]}</span>'
        if info.get("mixed_case", 0) > 0:
            tags += f'<span class="tag-warn">大小写 {info["mixed_case"]}</span>'
        if col in diag["number_like_cols"]:
            tags += f'<span class="tag-bad">疑似数字 {diag["number_like_cols"][col]}</span>'
        if tags == "":
            tags = '<span class="tag-ok">正常</span>'
        rows.append(
            {
                "列名": col,
                "类型": info["dtype"],
                "异常数": info.get("total_anomalies", 0),
                "唯一值": info["unique"],
                "问题标签": tags,
            }
        )
    st.markdown(
        pd.DataFrame(rows).to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

# ---------- 步骤 3：清洗操作 ----------
st.markdown('<p class="section-title">步骤 3 · 数据清洗</p>', unsafe_allow_html=True)

op_tab1, op_tab2, op_tab3, op_tab4 = st.tabs(
    ["🗑️ 重复 / 缺失处理", "🔤 文本规范化", "🔢 数值与异常值", "📅 日期与类型修正"]
)

# -------- 3.1 重复 / 缺失 --------
with op_tab1:
    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("**去除重复行**")
        n_dup = int(df_clean.duplicated().sum())
        st.info(f"当前检测到 {n_dup} 条完全重复行")
        if st.button("🗑️ 删除所有完全重复行", disabled=n_dup == 0, key="drop_dup"):
            before = len(df_clean)
            st.session_state["df_clean"] = df_clean.drop_duplicates().reset_index(drop=True)
            df_clean = st.session_state["df_clean"]
            after = len(df_clean)
            clean_log.append(f"[{datetime.now():%H:%M:%S}] 删除重复行：{before - after} 条")
            st.success(f"已删除 {before - after} 条重复行")
            st.rerun()

    with sub2:
        st.markdown("**缺失值处理**")
        cols_with_na = [c for c in df_clean.columns if df_clean[c].isna().any()]
        if not cols_with_na:
            st.info("✅ 当前无缺失值")
        else:
            target_col = st.selectbox("选择列", cols_with_na, key="na_col")
            na_cnt = int(df_clean[target_col].isna().sum())
            st.info(f"「{target_col}」共 {na_cnt} 个缺失值")

            if pd.api.types.is_numeric_dtype(df_clean[target_col]):
                method = st.radio(
                    "填充方式",
                    ["均值填充", "中位数填充", "众数填充", "指定常量", "删除含缺失值的行"],
                    key="na_num_method",
                )
                if method == "指定常量":
                    val = st.number_input("常量值", value=0.0, key="na_num_val")
                else:
                    val = None
            else:
                method = st.radio(
                    "填充方式",
                    ["众数填充", "指定文本", "删除含缺失值的行", "填充为空字符串"],
                    key="na_str_method",
                )
                if method == "指定文本":
                    val = st.text_input("填充文本", value="未知", key="na_str_val")
                else:
                    val = None

            if st.button("🧹 应用", key="btn_na"):
                before = int(df_clean[target_col].isna().sum())
                if method == "删除含缺失值的行":
                    st.session_state["df_clean"] = df_clean.dropna(subset=[target_col]).reset_index(drop=True)
                    clean_log.append(f"[日期] 删除「{target_col}」缺失行：{before} 条")
                elif method == "均值填充":
                    st.session_state["df_clean"][target_col] = df_clean[target_col].fillna(df_clean[target_col].mean())
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{target_col}」均值填充缺失：{before} 处")
                elif method == "中位数填充":
                    st.session_state["df_clean"][target_col] = df_clean[target_col].fillna(df_clean[target_col].median())
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{target_col}」中位数填充缺失：{before} 处")
                elif method == "众数填充":
                    mode_vals = df_clean[target_col].mode()
                    mode_val = mode_vals.iloc[0] if len(mode_vals) else "未知"
                    st.session_state["df_clean"][target_col] = df_clean[target_col].fillna(mode_val)
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{target_col}」众数({mode_val})填充：{before} 处")
                elif method == "填充为空字符串":
                    st.session_state["df_clean"][target_col] = df_clean[target_col].fillna("").astype(str)
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{target_col}」填充空字符串：{before} 处")
                else:
                    st.session_state["df_clean"][target_col] = df_clean[target_col].fillna(val)
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{target_col}」常量({val})填充：{before} 处")
                df_clean = st.session_state["df_clean"]
                st.success("已应用处理")
                st.rerun()

# -------- 3.2 文本规范化 --------
with op_tab2:
    obj_cols = list(df_clean.select_dtypes(include=["object"]).columns)
    if not obj_cols:
        st.info("当前无文本列")
    else:
        tgt = st.multiselect("选择要处理的文本列", obj_cols, default=obj_cols, key="txt_cols")
        actions = st.multiselect(
            "执行的操作",
            ["去除前后空格", "统一转为小写", "统一转为大写", "去除全角空白 / 特殊字符"],
            default=["去除前后空格"],
            key="txt_actions",
        )
        if st.button("🧹 应用文本规范化", disabled=not tgt or not actions):
            count_strip = 0
            for col in tgt:
                for act in actions:
                    if act == "去除前后空格":
                        old = df_clean[col].apply(
                            lambda x: isinstance(x, str) and x != x.strip()
                        ).sum()
                        st.session_state["df_clean"][col] = df_clean[col].apply(
                            lambda x: x.strip() if isinstance(x, str) else x
                        )
                        count_strip += int(old)
                    elif act == "统一转为小写":
                        st.session_state["df_clean"][col] = df_clean[col].apply(
                            lambda x: x.lower() if isinstance(x, str) else x
                        )
                    elif act == "统一转为大写":
                        st.session_state["df_clean"][col] = df_clean[col].apply(
                            lambda x: x.upper() if isinstance(x, str) else x
                        )
                    elif act == "去除全角空白 / 特殊字符":
                        import unicodedata

                        st.session_state["df_clean"][col] = df_clean[col].apply(
                            lambda x: (
                                unicodedata.normalize("NFKC", x).replace("\u3000", " ").strip()
                                if isinstance(x, str)
                                else x
                            )
                        )
            df_clean = st.session_state["df_clean"]
            clean_log.append(
                f"[{datetime.now():%H:%M:%S}] 文本规范化：{tgt} -> {actions}（去空格 {count_strip} 处）"
            )
            st.success("文本规范化完成")
            st.rerun()

    st.markdown("**文本值统一映射**（如把 M/男/Male 统一为「男」）")
    if obj_cols:
        map_col = st.selectbox("选择列", obj_cols, key="map_col")
        unique_vals = sorted([str(v) for v in df_clean[map_col].dropna().unique()])
        st.write("当前取值：", unique_vals[:20], "…共" if len(unique_vals) > 20 else "",)
        n_map = st.number_input("要设置的映射对数", min_value=1, max_value=10, value=1, key="n_map")
        mappings = {}
        for i in range(int(n_map)):
            c1a, c1b = st.columns(2)
            with c1a:
                old_v = st.text_input(f"原值 {i+1}", key=f"old_{i}")
            with c1b:
                new_v = st.text_input(f"替换为 {i+1}", key=f"new_{i}")
            if old_v and new_v:
                mappings[old_v] = new_v
        if st.button("🔄 应用映射", disabled=not mappings):
            changed = 0
            for old_v, new_v in mappings.items():
                mask = df_clean[map_col].astype(str) == old_v
                changed += int(mask.sum())
                st.session_state["df_clean"].loc[mask, map_col] = new_v
            df_clean = st.session_state["df_clean"]
            clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{map_col}」文本映射 {mappings}，共 {changed} 处")
            st.success(f"共替换 {changed} 处")
            st.rerun()

# -------- 3.3 数值 & 异常值 --------
with op_tab3:
    num_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    if not num_cols:
        st.info("当前无数值列")
    else:
        col3 = st.selectbox("选择数值列", num_cols, key="num_col")
        s = df_clean[col3].dropna()
        if len(s) > 0 and s.std() > 0:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers_mask = (df_clean[col3] < low) | (df_clean[col3] > high)
            outlier_cnt = int(outliers_mask.sum())
            st.info(f"IQR 范围：[{low:.2f}, {high:.2f}]，检测到 {outlier_cnt} 个异常值")
            st.dataframe(df_clean[outliers_mask], use_container_width=True)

            method3 = st.radio(
                "异常值处理方式",
                ["保留（仅查看）", "截断到边界值（Winsorize）", "替换为中位数", "删除含异常值的行"],
                key="num_method",
            )
            if st.button("🧹 应用", disabled=method3 == "保留（仅查看）"):
                before = outlier_cnt
                if method3 == "截断到边界值（Winsorize）":
                    st.session_state["df_clean"][col3] = df_clean[col3].clip(lower=low, upper=high)
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{col3}」异常值 Winsorize 截断：{before} 处")
                elif method3 == "替换为中位数":
                    median_v = df_clean[col3].median()
                    st.session_state["df_clean"].loc[outliers_mask, col3] = median_v
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{col3}」异常值替换为中位数({median_v})：{before} 处")
                elif method3 == "删除含异常值的行":
                    st.session_state["df_clean"] = df_clean[~outliers_mask].reset_index(drop=True)
                    clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{col3}」删除异常值行：{before} 条")
                df_clean = st.session_state["df_clean"]
                st.success("已应用处理")
                st.rerun()
        else:
            st.info("该列无足够数据用于异常值检测")

# -------- 3.4 日期与类型修正 --------
with op_tab4:
    cols4 = list(df_clean.columns)
    col4 = st.selectbox("选择要处理的列", cols4, key="conv_col")
    target_type = st.radio(
        "转换为",
        ["保持不变", "整数 (int)", "浮点 (float)", "日期 (datetime)", "文本 (str)"],
        key="conv_type",
    )
    if target_type != "保持不变" and st.button("🔧 应用类型转换"):
        try:
            if target_type == "整数 (int)":
                st.session_state["df_clean"][col4] = pd.to_numeric(df_clean[col4], errors="coerce").astype("Int64")
            elif target_type == "浮点 (float)":
                st.session_state["df_clean"][col4] = pd.to_numeric(df_clean[col4], errors="coerce")
            elif target_type == "日期 (datetime)":
                st.session_state["df_clean"][col4] = pd.to_datetime(df_clean[col4], errors="coerce")
            elif target_type == "文本 (str)":
                st.session_state["df_clean"][col4] = df_clean[col4].astype(str)
            df_clean = st.session_state["df_clean"]
            clean_log.append(f"[{datetime.now():%H:%M:%S}] 「{col4}」转换为 {target_type}")
            st.success("类型转换完成")
            st.rerun()
        except Exception as e:
            st.error(f"转换失败：{e}")

# ---------- 重置 / 日志 ----------
st.markdown("---")
log_col1, log_col2 = st.columns([4, 1])
with log_col1:
    with st.expander(f"📜 清洗操作日志（共 {len(clean_log)} 条）"):
        if clean_log:
            for log in clean_log:
                st.write("· " + log)
        else:
            st.caption("暂未执行任何清洗操作")
with log_col2:
    if st.button("🔄 重置为原始数据", use_container_width=True):
        st.session_state["df_clean"] = df_original.copy()
        st.session_state["clean_log"] = []
        st.success("已重置")
        st.rerun()

# ---------- 步骤 4：前后对比 & 可视化 ----------
st.markdown('<p class="section-title">步骤 4 · 前后对比 & 可视化</p>', unsafe_allow_html=True)

comp_t1, comp_t2, comp_t3 = st.tabs(["📊 清洗前后指标对比", "📈 数值列分布", "🗂️ 类别分布"])

with comp_t1:
    def report_metrics(d):
        d_diag = diagnose(d)
        return (
            d_diag["n_rows"],
            d_diag["missing_total"],
            d_diag["duplicate_rows"],
            sum(s.get("outliers_iqr", 0) for s in d_diag["col_stats"].values()),
        )

    orig_stats = report_metrics(df_original)
    clean_stats = report_metrics(df_clean)
    labels = ["总行数", "缺失单元格", "重复行", "数值异常值"]

    cmp_df = pd.DataFrame(
        {
            "指标": labels,
            "清洗前": orig_stats,
            "清洗后": clean_stats,
            "变化": [a - b for a, b in zip(orig_stats, clean_stats)],
        }
    )
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    st.markdown("**缺失值 / 重复行对比图**")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="清洗前", x=labels, y=list(orig_stats), text=list(orig_stats)))
    fig_cmp.add_trace(go.Bar(name="清洗后", x=labels, y=list(clean_stats), text=list(clean_stats)))
    fig_cmp.update_layout(barmode="group", height=400)
    st.plotly_chart(fig_cmp, use_container_width=True)

with comp_t2:
    num_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    if not num_cols:
        st.info("当前无数值列")
    else:
        col_n = st.selectbox("选择数值列", num_cols, key="viz_num")
        fig_num = go.Figure()
        fig_num.add_trace(
            go.Histogram(
                x=df_original[col_n].dropna(),
                name="清洗前",
                opacity=0.7,
                marker_color="#FF6B6B",
            )
        )
        fig_num.add_trace(
            go.Histogram(
                x=df_clean[col_n].dropna(),
                name="清洗后",
                opacity=0.7,
                marker_color="#4CAF50",
            )
        )
        fig_num.update_layout(
            title=f"「{col_n}」分布对比", barmode="overlay", height=420
        )
        st.plotly_chart(fig_num, use_container_width=True)

        col_box1, col_box2 = st.columns(2)
        with col_box1:
            fig_box1 = px.box(y=df_original[col_n].dropna(), title="清洗前箱线图")
            st.plotly_chart(fig_box1, use_container_width=True)
        with col_box2:
            fig_box2 = px.box(y=df_clean[col_n].dropna(), title="清洗后箱线图")
            st.plotly_chart(fig_box2, use_container_width=True)

with comp_t3:
    cat_cols = [
        c
        for c in df_clean.columns
        if df_clean[c].dtype == object and 1 < df_clean[c].nunique(dropna=True) <= 50
    ]
    if not cat_cols:
        st.info("未检测到合适的类别列")
    else:
        col_c = st.selectbox("选择类别列", cat_cols, key="viz_cat")
        before_vc = df_original[col_c].fillna("(缺失)").value_counts().head(15)
        after_vc = df_clean[col_c].fillna("(缺失)").value_counts().head(15)
        c1b, c2b = st.columns(2)
        with c1b:
            st.markdown("**清洗前**")
            st.plotly_chart(
                px.bar(x=before_vc.index, y=before_vc.values, text_auto=True),
                use_container_width=True,
            )
        with c2b:
            st.markdown("**清洗后**")
            st.plotly_chart(
                px.bar(x=after_vc.index, y=after_vc.values, text_auto=True, color_discrete_sequence=["#4CAF50"]),
                use_container_width=True,
            )

# ---------- 步骤 5：导出 ----------
st.markdown('<p class="section-title">步骤 5 · 导出结果</p>', unsafe_allow_html=True)

exp_format = st.radio("导出格式", ["CSV (UTF-8)", "Excel (XLSX)"], horizontal=True)
base_name = st.session_state.get("file_name", "cleaned_data").rsplit(".", 1)[0]
out_name = f"{base_name}_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if exp_format == "CSV (UTF-8)":
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载清洗后的 CSV",
        data=csv_bytes,
        file_name=f"{out_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_clean.to_excel(writer, sheet_name="清洗后数据", index=False)
        pd.DataFrame(
            {"操作日志": clean_log if clean_log else ["（无操作）"]}
        ).to_excel(writer, sheet_name="清洗日志", index=False)
    st.download_button(
        "⬇️ 下载清洗后的 Excel",
        data=buf.getvalue(),
        file_name=f"{out_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# 预览 JSON 报告
with st.expander("📄 查看清洗报告（文本）"):
    diag_after = diagnose(df_clean)
    report_lines = [
        f"文件：{st.session_state.get('file_name', '未知')}",
        f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"原始数据：{len(df_original)} 行 × {len(df_original.columns)} 列",
        f"清洗后数据：{len(df_clean)} 行 × {len(df_clean.columns)} 列",
        f"缺失值总数：{diag_after['missing_total']}",
        f"重复行数：{diag_after['duplicate_rows']}",
        "",
        "操作日志：",
    ] + [f"  · {log}" for log in (clean_log or ["（无）"])]
    st.text("\n".join(report_lines))

st.divider()
st.caption("© 2025 脏数据处理工具 · 基于 Streamlit 构建")
