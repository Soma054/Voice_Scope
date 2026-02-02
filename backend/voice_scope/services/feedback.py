from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


MetricKey = Literal["stability", "clarity", "resonance", "brightness", "voiced"]


class FeedbackRequest(BaseModel):
    consultation_text: Optional[str] = None
    focus_key: Optional[MetricKey] = None
    analysis: dict[str, Any] = Field(..., description="`POST /v1/analyze` のレスポンスをそのまま渡す")


def _pick_focus_key(req: FeedbackRequest) -> MetricKey:
    if req.focus_key is not None:
        return req.focus_key
    rk = req.analysis.get("recommended_highlight_key")
    if rk in {"stability", "clarity", "resonance", "brightness", "voiced"}:
        return rk
    # fallback
    return "clarity"


def _queries_for(key: MetricKey) -> list[str]:
    return {
        "stability": [
            "ピッチ 安定 コツ",
            "ビブラート コントロール 方法",
            "ロングトーン ピッチ 安定",
        ],
        "clarity": [
            "息漏れ 改善 発声",
            "録音 ノイズ 減らす マイク距離",
            "声帯閉鎖 感覚",
        ],
        "resonance": [
            "共鳴 位置 見つけ方",
            "響き 抜ける声 コツ",
            "ハミング 共鳴",
        ],
        "brightness": [
            "明るい声 出し方",
            "軟口蓋 上げ方",
            "トーン 明るさ 改善",
        ],
        "voiced": [
            "声が抜ける 原因",
            "息の支え 発声",
            "録音 ゲイン 適正",
        ],
    }[key]


def build_feedback(req: FeedbackRequest) -> dict[str, Any]:
    analysis = req.analysis or {}
    summary = analysis.get("summary") or {}
    source = analysis.get("source") or {}
    source_score = source.get("score") or {}
    target = analysis.get("target")
    target_score = (target or {}).get("score") or {}

    focus_key = _pick_focus_key(req)

    observations: list[str] = []
    grade = summary.get("reliability_grade")
    voiced_ratio = summary.get("voiced_ratio")
    if grade in {"B", "C"}:
        observations.append(
            "音声からの推定が不安定になりやすい状態です（発声区間が短い/環境音/伴奏などで `voiced_ratio` が下がる場合があります）。"
        )
    if isinstance(voiced_ratio, (int, float)):
        observations.append(f"推定された発声率（voiced_ratio）は {float(voiced_ratio)*100:.0f}% です。")

    # 低めの指標を簡易に言語化（断定しない）
    low_keys = []
    for k in ["stability", "clarity", "resonance", "brightness", "voiced"]:
        v = source_score.get(k)
        if isinstance(v, (int, float)) and float(v) <= 40:
            low_keys.append(k)
    if low_keys:
        observations.append(
            "相対的に低めに出ている軸があります（録音条件や曲調の影響も受けます）：" + ", ".join(low_keys)
        )

    if target_score:
        # gapの大きい軸
        best_gap_k: Optional[MetricKey] = None
        best_gap = -1e9
        for k in ["stability", "clarity", "resonance", "brightness", "voiced"]:
            sv = source_score.get(k)
            tv = target_score.get(k)
            if isinstance(sv, (int, float)) and isinstance(tv, (int, float)):
                gap = float(tv) - float(sv)
                if gap > best_gap:
                    best_gap = gap
                    best_gap_k = k  # type: ignore[assignment]
        if best_gap_k is not None:
            observations.append(f"Targetと比べて差が出やすい軸は `{best_gap_k}` です（参考）。")

    # 相談文を「観察」に寄せる（断定/医療助言にしない）
    if req.consultation_text:
        observations.append(
            "相談内容は“状況の整理”として扱い、ここでは一般的な情報の探し方（検索クエリ）を提示します。"
        )

    return {
        "policy": {
            "note": "これは医療的助言ではなく、音声特徴量にもとづく一般的な情報整理と検索の補助です。症状や不調がある場合は専門家に相談してください。"
        },
        "focus_key": focus_key,
        "observations": observations,
        "search_queries": _queries_for(focus_key),
        "optional_queries": [
            "録音 ノイズ 低減 方法",
            "マイク 距離 ゲイン 適正",
            "伴奏 あり 音声 ピッチ 推定 精度",
        ],
    }
