import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
import os

# フォントを Noto Sans CJK JP に変更（matplotlib用）
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

# ロゴ画像のパス（この .py ファイルと同じディレクトリにあると想定）
logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")

# タイトルとロゴを横並びに表示
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<h1 style='text-align: left;'>直線回帰分析</h1>", unsafe_allow_html=True)
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=60)
        st.text("T. Nagashima")
    else:
        st.warning("ロゴ画像が見つかりません: logo.jpg")

# タイトル下に水平線
st.markdown("<hr style='margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# 数式と説明文
st.latex(r'y = ax + b')
st.write("データを入力してから「計算実行」ボタンを押してください．")

# 初期データフレーム（空のデータフレームを用意）
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame({"X": ["0.0"], "Y": ["0.0"]})

# データ入力テーブル（TextColumnで指数表記も許可）
column_config = {
    "X": st.column_config.TextColumn("X（数値）", help="例: 1.2e12"),
    "Y": st.column_config.TextColumn("Y（数値）", help="例: 3.4e-5")
}

edited_df = st.data_editor(
    st.session_state.data,
    column_config=column_config,
    num_rows="dynamic",
    key="data_editor"
)

# ボタン右寄せ配置
button_col1, button_col2, button_col3 = st.columns([6, 1, 2])
with button_col3:
    run = st.button("計算実行", use_container_width=True, type="primary")

# ボタンがクリックされたら計算
if run:
    # NaNを含む行を除外
    cleaned_df = edited_df.dropna()

    # 数値変換を試みる
    try:
        cleaned_df["X"] = cleaned_df["X"].astype(float)
        cleaned_df["Y"] = cleaned_df["Y"].astype(float)
    except ValueError:
        st.error("XまたはYに無効な数値があります（指数表記も可: 1.2e12 等）。")
        st.stop()

    if len(cleaned_df) >= 2:
        X = cleaned_df["X"].values.reshape(-1, 1)
        Y = cleaned_df["Y"].values

        # 線形回帰モデル
        model = LinearRegression()
        model.fit(X, Y)

        # 回帰係数と切片
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, Y)

        # 計算結果を中央に大きく表示
        st.markdown(f"""
        <div style='text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;'>
            傾き a = {slope:.10f}<br>
            切片 b = {intercept:.10f}<br>
        </div>
        <div style='text-align: center; font-size: 18px; font-weight: bold; margin: 14px 0;'>
            回帰直線の式: Y = {slope:.10f} × X + {intercept:.10f}<br>
            決定係数 (R²): {r_squared:.6f}
        </div>
        """, unsafe_allow_html=True)

        # プロット
        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Data")
        ax.plot(X, model.predict(X), color="red", label="y=ax+b")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("2行以上の有効なデータを入力してください。")
