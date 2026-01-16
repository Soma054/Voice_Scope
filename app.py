import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.styles import apply_styles
from src.audio import SUPPORTED_AUDIO_EXTS, load_uploaded_audio, AudioLoadError
from src.analysis import compute_voice_metrics, metric_labels_jp

# -----------------------------
# Page config + styles
# -----------------------------
st.set_page_config(
    page_title="Voice-Scope (Local)",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()

# -----------------------------
# Session state init
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "calibration_done" not in st.session_state:
    st.session_state.calibration_done = False

if "selected_insight" not in st.session_state:
    st.session_state.selected_insight = 0  # 右側の要点カード選択

# audio
for k in ["source_audio", "source_sr", "source_name", "target_audio", "target_sr", "target_name"]:
    if k not in st.session_state:
        st.session_state[k] = None

# metrics cache
for k in ["source_metrics", "target_metrics"]:
    if k not in st.session_state:
        st.session_state[k] = None


def change_page(page: str):
    st.session_state.page = page
    st.rerun()


# -----------------------------
# Radar chart (plotly)
# -----------------------------
def create_radar_chart(source_scores: dict, target_scores: dict | None, highlight_key: str | None = None):
    labels = metric_labels_jp()
    keys = ["stability", "clarity", "resonance", "brightness", "voiced"]
    theta = [labels[k] for k in keys]

    src_values = [int(source_scores.get(k, 0)) for k in keys]
    tgt_values = [int(target_scores.get(k, 0)) for k in keys] if target_scores else [70] * len(keys)

    fig = go.Figure()

    # Source
    fig.add_trace(
        go.Scatterpolar(
            r=src_values,
            theta=theta,
            fill="toself",
            fillcolor="rgba(124, 58, 237, 0.20)",
            line=dict(color="#7C3AED", width=3),
            name=f"自分 (Source)  {st.session_state.source_name or ''}",
            marker=dict(size=7, color="#7C3AED"),
        )
    )

    # Target
    fig.add_trace(
        go.Scatterpolar(
            r=tgt_values,
            theta=theta,
            fill="toself",
            fillcolor="rgba(236, 72, 153, 0.10)",
            line=dict(color="#EC4899", width=3, dash="dash"),
            name=f"理想 (Target)  {st.session_state.target_name or '未設定'}",
            marker=dict(size=7, color="#EC4899"),
        )
    )

    # Highlight (selected axis)
    if highlight_key in keys:
        i = keys.index(highlight_key)
        # draw a spoke line (0 -> max(src,tgt) on that axis)
        rmax = max(src_values[i], tgt_values[i])
        fig.add_trace(
            go.Scatterpolar(
                r=[0, rmax],
                theta=[theta[i], theta[i]],
                mode="lines",
                line=dict(color="rgba(124,58,237,0.55)", width=8),
                name="highlight",
                showlegend=False,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(248, 247, 255, 0.5)",
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        legend=dict(x=0.5, y=-0.12, xanchor="center", orientation="h"),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=20),
        font=dict(size=14, color="#2B2A33"),
    )
    return fig


def insight_defs(src, tgt):
    """
    HTMLで見せてくれた “強み/弱み/差/バランス” をStreamlitで再現するための定義。
    src,tgt: MetricResult
    """
    labels = metric_labels_jp()
    keys = ["stability", "clarity", "resonance", "brightness", "voiced"]

    src_vals = np.array([src.score[k] for k in keys], dtype=float)

    # 強み/弱みは Source 内
    i_strong = int(np.argmax(src_vals))
    i_weak = int(np.argmin(src_vals))

    # 差が大きいは Targetとの差（Targetが無ければ70基準）
    if tgt is not None:
        tgt_vals = np.array([tgt.score[k] for k in keys], dtype=float)
    else:
        tgt_vals = np.array([70] * len(keys), dtype=float)

    gaps = tgt_vals - src_vals
    i_gap = int(np.argmax(gaps))  # targetが上で差が大きい項目

    # バランス（凸凹の少なさ）
    std = float(np.std(src_vals))
    balance = max(0.0, 100.0 - std * 2.2)

    INSIGHTS = [
        {
            "key": "strong",
            "title": f"強み：{labels[keys[i_strong]]}",
            "body": "自分スコアが最も高い軸。ここを維持しつつ他を底上げ。",
            "highlight": keys[i_strong],
        },
        {
            "key": "weak",
            "title": f"改善候補：{labels[keys[i_weak]]}",
            "body": "相対的に低い軸。フォーム/環境要因も含めて優先的に改善。",
            "highlight": keys[i_weak],
        },
        {
            "key": "gap",
            "title": f"理想との差が大きい：{labels[keys[i_gap]]}",
            "body": "Targetが上回っている差分が最大の軸。比較分析の優先度が高い。",
            "highlight": keys[i_gap],
        },
        {
            "key": "balance",
            "title": f"バランス：{balance:.0f} / 100",
            "body": "全体の凸凹の少なさ（ダミーではなく計算値）。一部だけ突出していると低下します。",
            "highlight": None,
        },
    ]
    return INSIGHTS, float(balance)


def training_reco(focus_key: str):
    """
    超ざっくり版。あとであなたの世界観に合わせていくらでも盛れる。
    """
    rec = {
        "stability": {
            "title": "ピッチ安定性トレーニング",
            "methods": ["ロングトーン（一定の息で10〜15秒）", "メトロノームに合わせてビブラート量を一定に", "チューナーで±20cent以内を目標に"],
            "keywords": ["ロング��ーン", "ピッチ 安定", "ビブラート コントロール"],
        },
        "clarity": {
            "title": "クリアさ（ノイズ低減）",
            "methods": ["息を一定に（吐きすぎない）", "ストロー発声で声帯閉鎖を整える", "録音環境ノイズを下げる（距離/部屋/ゲイン）"],
            "keywords": ["息漏れ 改善", "ストロー発声", "声帯閉鎖"],
        },
        "resonance": {
            "title": "響き・抜け",
            "methods": ["ハミングで共鳴位置を前に", "母音を明るく（口腔の形を固定）", "鼻腔/口腔の抜けを意識して録音比較"],
            "keywords": ["共鳴 位置", "ハミング 練習", "抜ける声"],
        },
        "brightness": {
            "title": "明るさ（トーン）",
            "methods": ["笑顔のフォームで発声", "軟口蓋を上げる意識（あくびの感覚）", "軽いアタックで立ち上がりを揃える"],
            "keywords": ["軟口蓋 上げ方", "明るい声", "トーン 改善"],
        },
        "voiced": {
            "title": "発声率（声が乗っている時間）",
            "methods": ["語尾が抜けないように息を支える", "子音で声が切れすぎないようにゆっくり歌う", "録音ゲインを適正化（小さすぎるとvoicedが落ちる）"],
            "keywords": ["声が抜ける", "息の支え", "録音 ゲイン"],
        },
    }
    return rec.get(focus_key, rec["clarity"])


# -----------------------------
# Pages
# -----------------------------
def page_home():
    st.markdown('<div class="main-title">🎤 Voice-Scope</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ローカル開発版：UXを元に戻して機能追加していく</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)

        st.markdown("### Voice Lab")
        st.markdown("あなたの声を分析し、理想の声とのギャップを可視化します。")
        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎤 比較��析\n\n*Calibration First*", key="go_calib"):
                change_page("calibration")

        with col_b:
            if st.button("⚡️ Quick診断\n\n*診断のみ*", key="go_quick"):
                st.session_state.calibration_done = False
                change_page("input")

        st.markdown("---")
        if st.button("📈 成長記録 (Import)", key="import"):
            st.info("この機能は今後実装予定です")

        st.markdown("</div>", unsafe_allow_html=True)


def page_calibration():
    st.markdown(
        """
        <div class="step-indicator">
            <span class="step active">1. 校正</span>
            <span>›</span>
            <span class="step">2. 本番</span>
            <span>›</span>
            <span class="step">3. 解析</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 🔭 校正テスト (推奨)")
        st.markdown("**「あー」と3秒間発声した音声**をアップロードしてください。")

        uploaded_calib = st.file_uploader(
            "校正用音声ファイルをアップロード",
            type=SUPPORTED_AUDIO_EXTS,
            key="calib_upload",
        )

        if uploaded_calib:
            st.success(f"✅ {uploaded_calib.name}")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("校正を確定して次へ", type="primary"):
                    st.session_state.calibration_done = True
                    change_page("input")
            with col_b:
                if st.button("やり直す"):
                    st.session_state.calibration_done = False
                    st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("スキップする"):
                    st.session_state.calibration_done = False
                    change_page("input")
            with col_b:
                if st.button("← 戻る"):
                    change_page("home")

        st.markdown("</div>", unsafe_allow_html=True)


def page_input():
    # ステップインジケーター
    if st.session_state.calibration_done:
        st.markdown(
            """
            <div class="step-indicator">
                <span class="step done">✔ 校正</span>
                <span>›</span>
                <span class="step active">2. 本番</span>
                <span>›</span>
                <span class="step">3. 解析</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="step-indicator">
                <span class="step active">2. 本番</span>
                <span>›</span>
                <span class="step">3. 解析</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 💿 理想の声 (Target)")
        target_file = st.file_uploader(
            "ファイルを選択",
            type=SUPPORTED_AUDIO_EXTS,
            key="target_upload",
            help="伴奏付きOK",
        )
        if target_file:
            st.success(f"✅ {target_file.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 🎤 自分の声 (Source)")
        st.markdown("*※マイク距離15cm推奨 / 環境音OFF*")
        source_file = st.file_uploader(
            "ファイルを選択",
            type=SUPPORTED_AUDIO_EXTS,
            key="source_upload",
        )
        if source_file:
            st.success(f"✅ {source_file.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sr = st.selectbox(
        "変換後サンプルレート（Source/Target共通）",
        options=[16000, 22050, 44100, 48000],
        index=2,
        key="io_sr",
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        if st.button("← 戻る"):
            change_page("home")

    with col_c:
        if st.button("🔬 解析開始", type="primary"):
            if not source_file:
                st.warning("自分の声（Source）をアップロードしてください")
                return

            # Source (required)
            try:
                y, out_sr = load_uploaded_audio(source_file, sr=sr)
            except AudioLoadError as e:
                st.error(str(e))
                return

            st.session_state.source_audio = y
            st.session_state.source_sr = out_sr
            st.session_state.source_name = source_file.name

            # Target (optional)
            if target_file:
                try:
                    ty, tsr = load_uploaded_audio(target_file, sr=sr)
                    st.session_state.target_audio = ty
                    st.session_state.target_sr = tsr
                    st.session_state.target_name = target_file.name
                except AudioLoadError as e:
                    st.warning(f"Targetの読み込みに失敗しました（Sourceのみで進めます）: {e}")
                    st.session_state.target_audio = None
                    st.session_state.target_sr = None
                    st.session_state.target_name = None
            else:
                st.session_state.target_audio = None
                st.session_state.target_sr = None
                st.session_state.target_name = None

            # reset metrics cache
            st.session_state.source_metrics = None
            st.session_state.target_metrics = None

            change_page("loading")


def page_loading():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="neu-card" style="text-align:center; padding:60px;">',
            unsafe_allow_html=True,
        )
        st.markdown("### 🔬 解析中...")
        st.markdown("音声の特徴量を抽出しています（ローカル処理）")

        progress_bar = st.progress(0)
        status_text = st.empty()

        import time

        steps = [
            "音声デコード完了",
            "基本特徴量計算中...",
            "ピッチ推定（pyin）中...",
            "比較メトリクス生成中...",
        ]
        for i, step in enumerate(steps):
            status_text.markdown(f"**{step}**")
            time.sleep(0.4)
            progress_bar.progress(int((i + 1) / len(steps) * 80))

        # --- actual compute here ---
        try:
            st.session_state.source_metrics = compute_voice_metrics(
                st.session_state.source_audio, int(st.session_state.source_sr)
            )
            if st.session_state.target_audio is not None and st.session_state.target_sr is not None:
                st.session_state.target_metrics = compute_voice_metrics(
                    st.session_state.target_audio, int(st.session_state.target_sr)
                )
            else:
                st.session_state.target_metrics = None
        except Exception as e:
            st.error(f"解析に失敗しました: {e}")
            st.stop()

        progress_bar.progress(100)
        status_text.markdown("**完了**")
        time.sleep(0.2)

        st.markdown("</div>", unsafe_allow_html=True)

    change_page("result1")


def page_result1():
    st.markdown('<div class="main-title">📊 解析レポート</div>', unsafe_allow_html=True)

    src = st.session_state.source_metrics
    tgt = st.session_state.target_metrics

    if src is None:
        st.warning("メトリクスがありません。Inputから解析を実行してください。")
        if st.button("入力へ戻る"):
            change_page("input")
        return

    # 信頼度 = voiced_ratioベース（簡易）
    voiced_ratio = float(src.raw.get("voiced_ratio", 0.0))
    if voiced_ratio >= 0.75:
        badge = "信頼度: 高 (A)"
    elif voiced_ratio >= 0.55:
        badge = "信頼度: 中 (B)"
    else:
        badge = "信頼度: 低 (C)"

    st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)

    # Insights (click)
    INSIGHTS, balance = insight_defs(src, tgt)
    sel = int(st.session_state.selected_insight or 0)
    sel = max(0, min(sel, len(INSIGHTS) - 1))
    selected = INSIGHTS[sel]

    highlight_key = selected["highlight"]

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 比較レーダー + ハイライト")
        st.markdown("右の「要点」をクリックすると該当軸を強調します。")

        fig = create_radar_chart(src.score, tgt.score if tgt else None, highlight_key=highlight_key)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "自分=<span class='badge' style='background:rgba(124,58,237,0.2);color:#4c1d95'>紫 実線</span> / "
            "理想=<span class='badge' style='background:rgba(236,72,153,0.15);color:#9d174d'>ピンク 破線</span>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 要点（クリック）")
        st.caption("5指標から見える「強み / 改善 / ギャップ / バランス」")

        for i, it in enumerate(INSIGHTS):
            # Streamlitボタンでも “カードっぽい見た目” に寄せたいので、ラベル短め
            label = f"{it['title']}"
            if st.button(label, key=f"insight_{i}", use_container_width=True):
                st.session_state.selected_insight = i
                st.rerun()

            st.markdown(
                f"<div style='margin-top:-8px; margin-bottom:12px; color:#6B6A77; font-size:13px;'>{it['body']}</div>",
                unsafe_allow_html=True,
            )

        # explain area (簡略/専門)
        tabs = st.tabs(["簡略", "専門"])
        with tabs[0]:
            axis = highlight_key if highlight_key else "all"
            st.markdown("**選択中:** " + selected["title"])
            if axis == "all":
                st.write("全体バランスを見ています。凸凹が大きいほど、一部だけ突出して聴こえやすいです。")
            else:
                st.write("この軸が高い/低いと、聴こえ方の傾向が変わります。次のページで数値根拠も確認できます。")

        with tabs[1]:
            st.caption("※MVP: proxy指標（今後Jitter/HNR等に精密化可能）")
            st.write("voiced_ratio:", f"{voiced_ratio*100:.1f}%")
            st.write("f0 range (p05-p95):", f"{src.raw.get('f0_p05_note','-')} 〜 {src.raw.get('f0_p95_note','-')}")

        st.markdown("</div>", unsafe_allow_html=True)

    # “盛り”: 追加サマリーカード（数値）
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown("### サマリー")

    c1, c2, c3, c4 = st.columns(4)

    # 総合（平均）
    keys = ["stability", "clarity", "resonance", "brightness", "voiced"]
    overall = int(round(np.mean([src.score[k] for k in keys])))

    # 一致度（Targetがある場合のみ：差分平均から算出）
    if tgt is not None:
        diffs = [abs(src.score[k] - tgt.score[k]) for k in keys]
        match = int(round(100 - np.mean(diffs)))
    else:
        match = 0

    with c1:
        st.markdown(
            f"<div class='custom-metric'><div class='custom-metric-value'>{overall}</div><div class='custom-metric-label'>総合スコア</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='custom-metric'><div class='custom-metric-value'>{match if tgt else '-'}"
            f"</div><div class='custom-metric-label'>一致度（参考）</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='custom-metric'><div class='custom-metric-value'>{voiced_ratio*100:.0f}%</div>"
            f"<div class='custom-metric-label'>発声率</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='custom-metric'><div class='custom-metric-value'>{src.raw.get('f0_p05_note','-')}–{src.raw.get('f0_p95_note','-')}</div>"
            f"<div class='custom-metric-label'>音域(参考)</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # “盛り”: トレーニング提案（ギャップ最大 or 弱点）
    focus_key = highlight_key or "clarity"
    rec = training_reco(focus_key)

    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 今日のおすすめトレーニング")
    st.markdown(f"**フォーカス：{rec['title']}**")
    for m in rec["methods"]:
        st.markdown(f"- {m}")
    st.markdown("**検索キーワード:** " + " / ".join(rec["keywords"]))
    st.markdown("</div>", unsafe_allow_html=True)

    # Nav
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("🏠 トップへ戻る"):
            # reset
            st.session_state.calibration_done = False
            for k in ["source_audio", "source_sr", "source_name", "target_audio", "target_sr", "target_name", "source_metrics", "target_metrics"]:
                st.session_state[k] = None
            st.session_state.selected_insight = 0
            change_page("home")

    with col_c:
        if st.button("詳細メトリクス →", type="primary"):
            change_page("result2")


def page_result2():
    st.markdown('<div class="main-title">📈 詳細メトリクス</div>', unsafe_allow_html=True)

    src = st.session_state.source_metrics
    tgt = st.session_state.target_metrics

    if src is None:
        st.warning("メトリクスがありません。Result1から来てください。")
        if st.button("Result1へ戻る"):
            change_page("result1")
        return

    labels = metric_labels_jp()
    keys = ["stability", "clarity", "resonance", "brightness", "voiced"]

    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown("### スコア（0-100）")

    for k in keys:
        s = int(src.score[k])
        t = int(tgt.score[k]) if tgt else 70
        diff = s - t
        diff_color = "#10b981" if diff >= 0 else "#ef4444"

        st.markdown(
            f"""
            <div class="metric-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1;">
                  <div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">{labels[k]}</div>
                  <div style="font-size:0.85em; color:#6B6A77;">Target: {t}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-family:monospace; font-size:1.8em; font-weight:bold; color:#7C3AED;">{s}</div>
                  <div style="font-size:0.9em; color:{diff_color}; font-weight:bold;">({diff:+d})</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # extra info: range & raw
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown("### 参考情報（計算根拠）")
    st.write("Source voiced_ratio:", f"{src.raw.get('voiced_ratio', 0.0)*100:.1f}%")
    st.write("Source pitch range (p05-p95):", f"{src.raw.get('f0_p05_note','-')} 〜 {src.raw.get('f0_p95_note','-')}")
    if tgt:
        st.write("Target voiced_ratio:", f"{tgt.raw.get('voiced_ratio', 0.0)*100:.1f}%")
        st.write("Target pitch range (p05-p95):", f"{tgt.raw.get('f0_p05_note','-')} 〜 {tgt.raw.get('f0_p95_note','-')}")
    st.caption("※ 音域は曲/キーにも依存するので参考値です。")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔧 raw features (debug)"):
        st.write("Source raw:", src.raw)
        if tgt:
            st.write("Target raw:", tgt.raw)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← 要点分析に戻る"):
            change_page("result1")
    with col_b:
        if st.button("🏠 トップへ戻る"):
            change_page("home")


# -----------------------------
# Router
# -----------------------------
page = st.session_state.page
if page == "home":
    page_home()
elif page == "calibration":
    page_calibration()
elif page == "input":
    page_input()
elif page == "loading":
    page_loading()
elif page == "result1":
    page_result1()
elif page == "result2":
    page_result2()
else:
    st.session_state.page = "home"
    st.rerun()