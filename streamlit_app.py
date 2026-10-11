"""
業務タイムスタディ記録アプリ
看護業務の実施時間を記録し、AIで自動分類する
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ===== 設定 =====
MODEL = "gemini-3-flash-preview"
USE_AUDIO = True  # 音声入力を使わない場合は False にする

LOCATIONS = ["病室", "ナースステーション", "廊下", "検査室", "その他"]
SHIFTS = ["日勤", "夜勤", "時短（13時まで）", "時短（15時まで）", "時短（16時まで）"]
LADDERS = ["未認定", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ"]
ROLES = ["リーダー", "スタッフ", "フリー"]

# 業務分類マスタ（中分類: 必要度タグ）
# ※必要度タグは暫定。最新の評価票で要確認
TAXONOMY = {
    "直接ケア": {
        "観察・バイタル測定": "A関連あり",
        "創傷処置": "A",
        "褥瘡処置": "なし",
        "呼吸ケア": "A",
        "注射・点滴の管理": "A",
        "輸血の管理": "A",
        "与薬（内服・外用）": "一部A",
        "移乗・移動の介助": "B",
        "口腔ケア": "B",
        "清拭・入浴の介助": "なし",
        "食事の介助": "B",
        "更衣の介助": "B",
        "トイレ歩行の介助": "なし",
        "おむつ交換": "なし",
        "危険行動への対応・見守り": "B関連あり",
        "検査の説明": "なし",
        "IC同席": "なし",
        "精神的支援": "なし",
        "検査・手術への送迎": "C関連あり",
        "急変対応": "一部A",
    },
    "入退院関連": {
        "予定入院": "なし",
        "緊急入院": "なし",
        "予定退院": "なし",
        "転院": "なし",
    },
    "間接ケア": {
        "看護計画の立案": "なし",
        "SOAP記録": "なし",
        "カンファレンス記録": "なし",
        "看護サマリー": "なし",
        "申し送り": "なし",
        "カンファレンス": "なし",
        "医師指示の受け・確認": "なし",
        "ケア・処置の準備": "なし",
        "調剤": "なし",
        "内服薬の確認": "なし",
        "他職種との連絡・調整": "なし",
        "家族への対応": "なし",
        "退院支援": "なし",
    },
}


def taxonomy_text() -> str:
    """AIに渡す分類一覧のテキスト"""
    lines = []
    for major, minors in TAXONOMY.items():
        lines.append(f"{major}: " + " / ".join(minors.keys()))
    return "\n".join(lines)


PROMPT = f"""あなたは看護業務の記録を分類する担当です。
入力された業務の記録を、以下のルールで処理してください。

【ルール】
1. 患者名・職員名・部屋番号などの固有名詞は必ず削除する。
   例「301号室の田中さんの清拭」→「清拭」
2. 場所は次の区分のいずれかに割り当てる: {" / ".join(LOCATIONS)}
3. 複数の業務が含まれる場合は、業務ごとに分けて出力する。
4. 表記ゆれは同じ中分類にまとめる（例：体を拭く → 清拭・入浴の介助）。
5. 大分類・中分類は必ず下記の一覧から選ぶ。一覧にない業務は
   最も近いものを選び、confidence を低くする。

【業務分類の一覧】
{taxonomy_text()}

【出力形式】
JSONの配列のみを出力する。説明文やコードブロックの記号は付けない。
各要素は次のキーを持つ:
  "業務内容": 固有名詞を削除した短い業務名（文字列）
  "大分類": 一覧の大分類名（文字列）
  "中分類": 一覧の中分類名（文字列）
  "場所区分": 場所の区分（文字列）
  "confidence": 分類の確信度 0.0〜1.0（数値）

例: [{{"業務内容":"清拭","大分類":"直接ケア","中分類":"清拭・入浴の介助","場所区分":"病室","confidence":0.95}}]
"""


def classify(client, text=None, audio_bytes=None):
    """AIで業務を分類する。結果のリストを返す"""
    parts = [PROMPT]
    if audio_bytes:
        parts.append("次の音声は業務の記録です。聞き取って処理してください。")
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
    else:
        parts.append(f"業務の記録：{text}")

    response = client.models.generate_content(
        model=MODEL,
        contents=parts,
        config={"response_mime_type": "application/json"},
    )
    raw = (response.text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    if isinstance(result, dict):
        result = [result]
    return result


# ===== 画面 =====
st.set_page_config(page_title="業務タイムスタディ記録", page_icon="⏱")
st.title("⏱ 業務タイムスタディ記録")
st.caption("話すだけ・押すだけで、看護業務の時間を記録します")

with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini APIキー", type="password")
    st.caption("[Google AI Studio](https://aistudio.google.com/apikey) で取得できます")

    st.divider()
    st.subheader("記録者の条件")
    staff = st.text_input("職員の記号", value="看護師A")
    shift = st.selectbox("勤務帯", SHIFTS)
    ladder = st.selectbox("ラダー", LADDERS)
    years = st.number_input("経験年数", min_value=0, max_value=30, value=5, step=1)
    n_patients = st.number_input(
        "受け持ち人数", min_value=0, max_value=30, value=7, step=1
    )
    role = st.selectbox("役割", ROLES)

    st.divider()
    st.caption("※研修用の試作です。患者情報・職員の実名は入力しないでください。")

if not api_key:
    st.info("左の設定欄にGemini APIキーを入力してください。", icon="🗝️")
    st.stop()

client = genai.Client(api_key=api_key)

# セッション状態の初期化
if "records" not in st.session_state:
    st.session_state.records = []
if "running" not in st.session_state:
    st.session_state.running = None
if "pending" not in st.session_state:
    st.session_state.pending = None
if "audio_round" not in st.session_state:
    st.session_state.audio_round = 0

# --- 記録中でないとき：開始の入力 ---
if st.session_state.running is None and st.session_state.pending is None:
    st.subheader("1. これから行う業務を入力")

    audio = None
    memo = ""
    if USE_AUDIO:
        tab1, tab2 = st.tabs(["🎤 音声で入力", "⌨ 文字で入力"])
        with tab1:
            st.caption(
                "🔴 マイクを押す → 業務名を話す → ⏹ もう一度押して止める。"
                "止めた時点で記録が始まります。"
            )
            audio = st.audio_input(
                "業務名を話してください",
                key=f"audio_{st.session_state.audio_round}",
            )
            # 録音が止まった時点で自動的に記録開始
            if audio is not None:
                st.session_state.running = {
                    "開始時刻": datetime.now(),
                    "memo": None,
                    "audio": audio.getvalue(),
                }
                st.session_state.audio_round += 1
                st.rerun()
        with tab2:
            memo = st.text_input("業務内容", placeholder="例：301号室の清拭をします")
            if st.button("▶ 開始", type="primary"):
                if not memo:
                    st.warning("業務内容を入力してください。")
                else:
                    st.session_state.running = {
                        "開始時刻": datetime.now(),
                        "memo": memo,
                        "audio": None,
                    }
                    st.rerun()
    else:
        memo = st.text_input("業務内容", placeholder="例：301号室の清拭をします")
        if st.button("▶ 開始", type="primary"):
            if not memo:
                st.warning("業務内容を入力してください。")
            else:
                st.session_state.running = {
                    "開始時刻": datetime.now(),
                    "memo": memo,
                    "audio": None,
                }
                st.rerun()

# --- 記録中 ---
elif st.session_state.running is not None:
    start = st.session_state.running["開始時刻"]
    st.subheader("2. 記録中")
    st.metric("開始時刻", start.strftime("%H:%M:%S"))
    st.info("ケアが終わったら下のボタンを押してください。", icon="⏳")

    if st.button("■ 終了", type="primary"):
        end = datetime.now()
        run = st.session_state.running
        with st.spinner("AIが業務を分類しています..."):
            try:
                items = classify(client, text=run["memo"], audio_bytes=run["audio"])
            except Exception as e:
                st.error(f"分類に失敗しました：{e}")
                items = [
                    {
                        "業務内容": run["memo"] or "（聞き取り失敗）",
                        "大分類": "直接ケア",
                        "中分類": "観察・バイタル測定",
                        "場所区分": "その他",
                        "confidence": 0.0,
                    }
                ]
        st.session_state.pending = {"開始時刻": start, "終了時刻": end, "items": items}
        st.session_state.running = None
        st.rerun()

# --- 確認画面 ---
elif st.session_state.pending is not None:
    p = st.session_state.pending
    start, end = p["開始時刻"], p["終了時刻"]
    total = (end - start).total_seconds() / 60
    n = max(len(p["items"]), 1)
    multi = n > 1

    st.subheader("3. 内容の確認")
    st.write(
        f"{start.strftime('%H:%M:%S')} 〜 {end.strftime('%H:%M:%S')}　"
        f"（{total:.1f} 分{'／' + str(n) + '件に分割' if multi else ''}）"
    )

    confirmed = []
    for i, item in enumerate(p["items"]):
        conf = float(item.get("confidence", 0))
        label = f"{i + 1}. {item.get('業務内容', '')}"
        if conf < 0.7:
            label += "　⚠ 要確認"
        with st.expander(label, expanded=(conf < 0.7)):
            majors = list(TAXONOMY.keys())
            major = st.selectbox(
                "大分類",
                majors,
                index=majors.index(item["大分類"]) if item.get("大分類") in majors else 0,
                key=f"maj{i}",
            )
            minors = list(TAXONOMY[major].keys())
            minor = st.selectbox(
                "中分類",
                minors,
                index=minors.index(item["中分類"]) if item.get("中分類") in minors else 0,
                key=f"min{i}",
            )
            place = st.selectbox(
                "場所区分",
                LOCATIONS,
                index=LOCATIONS.index(item["場所区分"])
                if item.get("場所区分") in LOCATIONS
                else 0,
                key=f"loc{i}",
            )
            st.caption(f"必要度タグ：{TAXONOMY[major][minor]}")
        confirmed.append(
            {
                "開始時刻": start.strftime("%H:%M:%S"),
                "終了時刻": end.strftime("%H:%M:%S"),
                "所要時間(分)": round(total / n, 1),
                "大分類": major,
                "中分類": minor,
                "必要度タグ": TAXONOMY[major][minor],
                "場所区分": place,
                "複数業務": "○" if multi else "",
                "職員": staff,
                "勤務帯": shift,
                "ラダー": ladder,
                "経験年数": years,
                "受け持ち人数": n_patients,
                "役割": role,
            }
        )

    c1, c2 = st.columns(2)
    if c1.button("✓ この内容で記録", type="primary"):
        st.session_state.records.extend(confirmed)
        st.session_state.pending = None
        st.rerun()
    if c2.button("✗ 取り消す"):
        st.session_state.pending = None
        st.rerun()

# --- 記録一覧 ---
st.divider()
st.subheader("📋 本日の記録")

if not st.session_state.records:
    st.caption("まだ記録がありません。")
else:
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, hide_index=True)

    c1, c2 = st.columns(2)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    c1.download_button(
        "⬇ CSVでダウンロード",
        csv,
        file_name=f"timestudy_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    if c2.button("🤖 AIで傾向を要約"):
        with st.spinner("集計しています..."):
            major_sum = df.groupby("大分類")["所要時間(分)"].sum().to_dict()
            minor_sum = df.groupby("中分類")["所要時間(分)"].sum().to_dict()
            try:
                res = client.models.generate_content(
                    model=MODEL,
                    contents=(
                        "次は看護師の業務時間の集計です。"
                        "業務量の傾向と、業務整理の観点で気づく点を"
                        "3つ以内の短い箇条書きで述べてください。\n"
                        f"大分類別の合計(分): {major_sum}\n"
                        f"中分類別の合計(分): {minor_sum}"
                    ),
                )
                st.success(res.text)
            except Exception as e:
                st.error(f"要約に失敗しました：{e}")

    with st.expander("集計を見る"):
        st.bar_chart(df.groupby("大分類")["所要時間(分)"].sum())
