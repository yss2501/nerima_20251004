import streamlit as st
import requests
import folium
import pandas as pd
from datetime import datetime  # 日付取得用
from streamlit_folium import st_folium
import time
from openai import OpenAI  # GPTコメント生成用ライブラリ（新しいバージョン）
from PIL import Image  # 画像処理用ライブラリ
import io

# OpenRouterクライアントの初期化
def get_openrouter_client():
    """OpenRouterクライアントを取得"""
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        return None
    
    return OpenAI(
        api_key=st.secrets["openai"]["api_key"],
        base_url="https://openrouter.ai/api/v1"
    )

# 利用可能なAIモデル一覧（OpenRouter対応）
AVAILABLE_MODELS = {
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4o": "openai/gpt-4o", 
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gemini-pro": "google/gemini-pro",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "deepseek-chat": "deepseek/deepseek-chat"
}

API_KEY = "AIzaSyAf_qxaXszMB2YmNUYrSlocBrf53b7Al6U"  # ここに有効なAPIキーを記入

# APIのURLと都市コード（東京固定）
city_code = "130010"  # 東京の都市コード
url = f"https://weather.tsukumijima.net/api/forecast/city/{city_code}"  # リクエストURL

# 画像の自動リサイズ・圧縮関数
def optimize_image(image_file, max_width=800, max_height=600, quality=85, max_size_mb=0.8):
    """
    画像をWeb表示に適したサイズに自動変換
    - max_width: 最大幅（デフォルト800px）
    - max_height: 最大高さ（デフォルト600px）
    - quality: JPEG品質（デフォルト85）
    - max_size_mb: 最大ファイルサイズ（MB）
    """
    try:
        # 画像を開く
        image = Image.open(image_file)
        
        # 元のサイズを記録
        original_size = image_file.getvalue().__len__()
        original_width, original_height = image.size
        
        # RGBモードに変換（JPEG保存のため）
        if image.mode in ('RGBA', 'LA', 'P'):
            # 透明部分を白で埋める
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # アスペクト比を保ちながらリサイズ
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # ファイルサイズをチェックして品質を調整
        current_quality = quality
        while current_quality > 10:
            # メモリ上でJPEGとして保存してサイズをチェック
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=current_quality, optimize=True)
            output_size = output.getbuffer().nbytes
            
            # 目標サイズ（MB）に収まっているかチェック
            if output_size <= max_size_mb * 1024 * 1024:
                break
            
            # 品質を下げて再試行
            current_quality -= 10
        
        # 最終的な画像データを取得
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=current_quality, optimize=True)
        optimized_data = output.getvalue()
        
        # 最適化結果を返す
        return {
            'data': optimized_data,
            'original_size': original_size,
            'optimized_size': len(optimized_data),
            'original_dimensions': (original_width, original_height),
            'optimized_dimensions': image.size,
            'quality_used': current_quality,
            'compression_ratio': round((1 - len(optimized_data) / original_size) * 100, 1)
        }
        
    except Exception as e:
        raise Exception(f"画像の最適化に失敗しました: {str(e)}")

# 天気情報を取得する関数
def get_weather(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.sidebar.error("天気情報を取得できませんでした。")
        return None

# コメント生成関数
def generate_gpt_comment(destinations, model_name="claude-3-haiku"):
    try:
        # APIキーの存在チェック
        if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
            return "⚠️ APIキーが設定されていません。管理者メニューで設定してください。"
        
        # OpenRouterクライアントを取得
        client = get_openrouter_client()
        if client is None:
            return "⚠️ OpenRouterクライアントの初期化に失敗しました。"
        
        # モデル名を取得
        model = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["claude-3-haiku"])
        
        # プロンプトの作成
        messages = [
            {"role": "system", "content": "あなたは練馬の地元旅行ガイドのネリーです。"},
            {"role": "user", "content": (
                f"以下の情報を元に、場所1と場所2を組み合わせた冒険や旅行の提案を、100字以内でユニークでわくわくするコメントを作成してください。\n\n" +
                f"場所1: {destinations[0]['場所']}\n解説: {destinations[0]['解説']}\n\n" +
                f"場所2: {destinations[1]['場所']}\n解説: {destinations[1]['解説']}\n\n" +
                "まとめコメント:"
            )}
        ]

        # OpenRouterのAPI呼び出し（複数モデル対応）
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "https://nerima-wanderland.streamlit.app",
                "X-Title": "練馬ワンダーランド"
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "User not found" in error_msg:
            return "⚠️ APIキーが無効です。OpenRouterでAPIキーを確認してください。"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "⚠️ APIの使用制限に達しました。しばらく待ってから再試行してください。"
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            return "⚠️ APIアクセスが拒否されました。APIキーの権限を確認してください。"
        else:
            return f"⚠️ コメント生成中にエラーが発生しました: {error_msg}"

# CSVデータを読み込み（文字化け対応）
try:
    # UTF-8 BOM付きを試行
    data = pd.read_csv("destinations.csv", encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        # Shift_JISを試行
        data = pd.read_csv("destinations.csv", encoding='shift_jis')
    except UnicodeDecodeError:
        # CP932を試行
        data = pd.read_csv("destinations.csv", encoding='cp932')

# 固定された出発地
fixed_origin = "豊島園駅"
st.session_state.setdefault("fixed_origin", fixed_origin)

# Streamlitアプリ
st.title("練馬ワンダーランド")

# 管理者認証状態の初期化
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

# メインタブと管理者タブを作成
tab1, tab2 = st.tabs(["🏠 メイン", "⚙️ 管理者メニュー"])

# サイドバーにウィジェットを配置
with st.sidebar:
    st.header("設定")
    
    # 気分の選択肢を表示
    if "今の気持ち" in data.columns:
        selected_mood = st.selectbox("今の気分を選んでください", data["今の気持ち"].unique())
    else:
        st.error("CSVファイルに「今の気持ち」カラムが見つかりません。")
    
    # 移動手段の選択肢を表示
    transport_mode = st.radio("移動手段を選んでください", ["徒歩", "自転車", "タクシー"])
    mode_map = {"徒歩": "walking", "自転車": "bicycling", "タクシー": "driving"}
    selected_mode = mode_map[transport_mode]
    
    # 確定ボタン
    search_button = st.button("ルートを検索")

    # サイドバーの下部に天気情報を表示
    st.markdown("---")  # 水平線で区切りを追加
    st.subheader("練馬の天気（3日分）")

    # 天気情報の取得と表示
    weather_json = get_weather(url)
    if weather_json:
        # 天気情報を3日分表示
        for i in range(3):  # 今日、明日、明後日
            forecast_date = weather_json['forecasts'][i]['dateLabel']
            weather = weather_json['forecasts'][i]['telop']
            icon_url = weather_json['forecasts'][i]['image']['url']
            st.image(icon_url, width=85)
            st.write(f"{forecast_date}: {weather}")
    else:
        st.write("天気情報を取得できませんでした。")

with tab1:
    # 以下はスライドショーやルート検索の処理
    if "search_completed" not in st.session_state:
        st.session_state["search_completed"] = False

    if not search_button and not st.session_state["search_completed"]:
        image_placeholder = st.empty()
        images = ["pic/0.png", "pic/1.png", "pic/2.png"]
        for img in images:
            image_placeholder.image(img, use_container_width=True)
            time.sleep(1)
            if st.session_state["search_completed"]:
                break
    else:
        st.session_state["search_completed"] = True

    if search_button:
        st.session_state["search_completed"] = True
        # 新しい検索時はコメントをリセット
        if "adventure_comment" in st.session_state:
            del st.session_state["adventure_comment"]

        if selected_mood:
            selected_data = data[data["今の気持ち"] == selected_mood].iloc[0]

            # 保存用データをセッションに記録
            st.session_state["selected_data"] = {
                "場所1": selected_data["場所1"],
                "画像1": selected_data["画像1"],
                "解説1": selected_data["解説1"],
                "場所2": selected_data["場所2"],
                "画像2": selected_data["画像2"],
                "解説2": selected_data["解説2"]
            }

            origin = fixed_origin
            destination1 = selected_data["住所1"]
            destination2 = selected_data["住所2"]

            directions_url1 = (
                f"https://maps.googleapis.com/maps/api/directions/json"
                f"?origin={origin}&destination={destination1}&mode={selected_mode}&key={API_KEY}"
            )
            directions_url2 = (
                f"https://maps.googleapis.com/maps/api/directions/json"
                f"?origin={destination1}&destination={destination2}&mode={selected_mode}&key={API_KEY}"
            )
            directions_url3 = (
                f"https://maps.googleapis.com/maps/api/directions/json"
                f"?origin={destination2}&destination={origin}&mode={selected_mode}&key={API_KEY}"
            )

            res1 = requests.get(directions_url1)
            res2 = requests.get(directions_url2)
            res3 = requests.get(directions_url3)

            if res1.status_code == 200 and res2.status_code == 200 and res3.status_code == 200:
                data1 = res1.json()
                data2 = res2.json()
                data3 = res3.json()

                if "routes" in data1 and len(data1["routes"]) > 0 and "routes" in data2 and len(data2["routes"]) > 0 and "routes" in data3 and len(data3["routes"]) > 0:
                    route1 = data1["routes"][0]["overview_polyline"]["points"]
                    route2 = data2["routes"][0]["overview_polyline"]["points"]
                    route3 = data3["routes"][0]["overview_polyline"]["points"]

                    # Decode polyline
                    def decode_polyline(polyline_str):
                        index, lat, lng, coordinates = 0, 0, 0, []
                        while index < len(polyline_str):
                            b, shift, result = 0, 0, 0
                            while True:
                                b = ord(polyline_str[index]) - 63
                                index += 1
                                result |= (b & 0x1F) << shift
                                shift += 5
                                if b < 0x20:
                                    break
                            dlat = ~(result >> 1) if result & 1 else (result >> 1)
                            lat += dlat
                            shift, result = 0, 0
                            while True:
                                b = ord(polyline_str[index]) - 63
                                index += 1
                                result |= (b & 0x1F) << shift
                                shift += 5
                                if b < 0x20:
                                    break
                            dlng = ~(result >> 1) if result & 1 else (result >> 1)
                            lng += dlng
                            coordinates.append((lat / 1e5, lng / 1e5))
                        return coordinates

                    route_coords1 = decode_polyline(route1)
                    route_coords2 = decode_polyline(route2)
                    route_coords3 = decode_polyline(route3)

                    # セッションにルートデータ保存
                    st.session_state["route_coords1"] = route_coords1
                    st.session_state["route_coords2"] = route_coords2
                    st.session_state["route_coords3"] = route_coords3

                    # 移動時間を取得
                    duration1 = data1["routes"][0]["legs"][0]["duration"]["text"]
                    duration2 = data2["routes"][0]["legs"][0]["duration"]["text"]
                    duration3 = data3["routes"][0]["legs"][0]["duration"]["text"]

                    st.session_state["route_table"] = pd.DataFrame({
                        "出発地": [fixed_origin, selected_data["場所1"], selected_data["場所2"]],
                        "目的地": [selected_data["場所1"], selected_data["場所2"], fixed_origin],
                        "所要時間": [duration1, duration2, duration3]
                    })

                    # 地図データを保存
                    m = folium.Map(location=route_coords1[0], zoom_start=13)
                    folium.PolyLine(route_coords1, color="blue", weight=5, opacity=0.7).add_to(m)
                    folium.PolyLine(route_coords2, color="purple", weight=5, opacity=0.7).add_to(m)
                    folium.PolyLine(route_coords3, color="red", weight=5, opacity=0.7).add_to(m)

                    # Add markers
                    folium.Marker(
                        location=route_coords1[0], popup="出発地: " + origin, icon=folium.Icon(color="green")
                    ).add_to(m)
                    folium.Marker(
                        location=route_coords1[-1], popup="目的地1: " + selected_data["場所1"], icon=folium.Icon(color="orange")
                    ).add_to(m)
                    folium.Marker(
                        location=route_coords2[-1], popup="目的地2: " + selected_data["場所2"], icon=folium.Icon(color="red")
                    ).add_to(m)
                    folium.Marker(
                        location=route_coords3[-1], popup="戻り: " + origin, icon=folium.Icon(color="blue")
                    ).add_to(m)

                    st.session_state["map"] = m
                    
    # メイン画面に状態を再表示
    if "selected_data" in st.session_state:
        selected_data = st.session_state["selected_data"]

        st.write("### あなたの気分にあった冒険プランは、こちらです！")
        # 目的地情報リスト
        destinations = [
            {"場所": selected_data["場所1"], "解説": selected_data["解説1"]},
            {"場所": selected_data["場所2"], "解説": selected_data["解説2"]},
        ]
        
        # GPTコメント生成（一度だけ実行）
        if "adventure_comment" not in st.session_state:
            # 選択されたモデルを取得（デフォルトはclaude-3-haiku）
            selected_model = st.session_state.get("selected_model", "claude-3-haiku")
            with st.spinner(f"コメントを生成中です（{selected_model}）..."):
                st.session_state["adventure_comment"] = generate_gpt_comment(destinations, selected_model)
        
        adventure_comment = st.session_state["adventure_comment"]

        # 場所1の情報を表示
        st.write(f"#### {selected_data['場所1']}")
        col1, col2 = st.columns([1, 3])  # カラムを分割してレイアウト調整
        with col1:
            st.image(selected_data['画像1'], caption=selected_data['場所1'], width=150)
        with col2:
            st.write(selected_data['解説1'])
        
        # 場所2の情報を表示
        st.write(f"#### {selected_data['場所2']}")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(selected_data['画像2'], caption=selected_data['場所2'], width=150)
        with col2:
            st.write(selected_data['解説2'])

        # GPTコメントを表示
        st.write("### ネリーからの提案")
        st.write(adventure_comment)

    # 保存された表を表示
    if "route_table" in st.session_state:
        st.write("### ルート情報")
        st.table(st.session_state["route_table"])

    if "map" in st.session_state:
        st.write("### 地図")
        st_folium(st.session_state["map"], width=725)

with tab2:
    st.header("⚙️ 管理者メニュー")
    
    # 管理者認証チェック
    if not st.session_state["admin_authenticated"]:
        st.warning("⚠️ 管理者認証が必要です。")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("🔐 管理者認証")
            password = st.text_input("管理者パスワード", type="password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("認証", use_container_width=True):
                    # パスワードは secrets.toml で管理（デフォルト: admin123）
                    admin_password = st.secrets.get("admin", {}).get("password", "admin123")
                    if password == admin_password:
                        st.session_state["admin_authenticated"] = True
                        st.success("認証成功！")
                        st.rerun()
                    else:
                        st.error("パスワードが間違っています")
            
            with col_b:
                if st.button("キャンセル", use_container_width=True):
                    st.info("認証をキャンセルしました")
    
    else:
        # 認証済みの場合の管理者機能
        st.success("✅ 管理者として認証されています。")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("ログアウト", use_container_width=True):
                st.session_state["admin_authenticated"] = False
                st.rerun()
        
        st.markdown("---")
        
        # CSV管理セクション
        st.subheader("📁 CSV管理")
        
        # CSVダウンロード機能
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 現在のCSVをダウンロード", use_container_width=True):
                # UTF-8 BOM付きでCSVデータを作成（Excel対応）
                csv_data = '\ufeff' + data.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="CSVファイルをダウンロード",
                    data=csv_data,
                    file_name="destinations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📋 現在のCSV内容を表示", use_container_width=True):
                st.write("**現在のデータ:**")
                st.dataframe(data, use_container_width=True)
        
        st.markdown("---")
        
        # CSVアップロード機能（追記・上書き選択）
        st.subheader("📤 CSVアップロード")
        
        upload_mode = st.radio(
            "アップロードモードを選択",
            ["🔄 完全上書き", "➕ 追記（既存データに追加）"],
            help="完全上書き: 既存データを削除して新しいデータで置き換え\n追記: 既存データに新しいデータを追加"
        )
        
        uploaded_file = st.file_uploader(
            "CSVファイルをアップロード", 
            type=['csv'],
            help="destinations.csvを更新します"
        )
        
        if uploaded_file is not None:
            try:
                # アップロードされたCSVを読み込み（文字化け対応）
                try:
                    # UTF-8 BOM付きを試行
                    new_data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        # Shift_JISを試行
                        new_data = pd.read_csv(uploaded_file, encoding='shift_jis')
                    except UnicodeDecodeError:
                        # CP932を試行
                        new_data = pd.read_csv(uploaded_file, encoding='cp932')
                
                # 必要なカラムが存在するかチェック
                required_columns = ["今の気持ち", "場所1", "画像1", "解説1", "住所1", "場所2", "画像2", "解説2", "住所2"]
                if all(col in new_data.columns for col in required_columns):
                    
                    if upload_mode == "➕ 追記（既存データに追加）":
                        # 追記モード: 既存データに新しいデータを追加
                        combined_data = pd.concat([data, new_data], ignore_index=True)
                        
                        # 重複チェック（同じ「今の気持ち」がある場合）
                        duplicate_moods = combined_data[combined_data.duplicated(subset=['今の気持ち'], keep=False)]
                        if not duplicate_moods.empty:
                            st.warning("⚠️ 重複する「今の気持ち」が見つかりました:")
                            st.dataframe(duplicate_moods[['今の気持ち']], use_container_width=True)
                            
                            if st.button("🔄 重複を上書きして続行"):
                                # 重複を削除（最初のものを残す）
                                combined_data = combined_data.drop_duplicates(subset=['今の気持ち'], keep='first')
                                final_data = combined_data
                            else:
                                st.info("処理をキャンセルしました。重複を解決してから再アップロードしてください。")
                                final_data = None
                        else:
                            final_data = combined_data
                    else:
                        # 完全上書きモード
                        final_data = new_data
                    
                    if final_data is not None:
                        # CSVファイルを保存（UTF-8 BOM付きで保存）
                        final_data.to_csv("destinations.csv", index=False, encoding='utf-8-sig')
                        
                        # 結果を表示
                        if upload_mode == "➕ 追記（既存データに追加）":
                            st.success(f"✅ CSVファイルに {len(new_data)} 件のデータを追記しました！")
                            st.info(f"**合計データ件数:** {len(final_data)} 件")
                        else:
                            st.success("✅ CSVファイルが完全に更新されました！")
                            st.info(f"**新しいデータ件数:** {len(final_data)} 件")
                        
                        st.info("ページを再読み込みしてください。")
                        
                        # セッション状態をリセット
                        for key in list(st.session_state.keys()):
                            if key not in ["fixed_origin", "admin_authenticated"]:
                                del st.session_state[key]
                            
                else:
                    missing_cols = [col for col in required_columns if col not in new_data.columns]
                    st.error(f"❌ 必要なカラムが不足しています: {', '.join(missing_cols)}")
                    st.info("**必要なカラム:** " + ", ".join(required_columns))
                    
            except Exception as e:
                st.error(f"❌ ファイルの読み込みに失敗しました: {e}")
        
        st.markdown("---")
        
        # 画像管理セクション
        st.subheader("🖼️ 画像管理")
        
        uploaded_images = st.file_uploader(
            "画像ファイルをアップロード", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="複数の画像を一度にアップロードできます"
        )
        
        if uploaded_images:
            for uploaded_image in uploaded_images:
                try:
                    # 画像を最適化
                    with st.spinner(f"{uploaded_image.name} を最適化中..."):
                        optimization_result = optimize_image(uploaded_image)
                    
                    # 最適化された画像を保存
                    image_path = f"pic/{uploaded_image.name}"
                    with open(image_path, "wb") as f:
                        f.write(optimization_result['data'])
                    
                    # 最適化結果を表示
                    st.success(f"✅ {uploaded_image.name} を保存しました")
                    
                    # 最適化詳細を表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("元のサイズ", f"{optimization_result['original_size']/1024:.1f} KB")
                    with col2:
                        st.metric("最適化後", f"{optimization_result['optimized_size']/1024:.1f} KB")
                    with col3:
                        st.metric("圧縮率", f"{optimization_result['compression_ratio']}%")
                    
                    # サイズ変更情報
                    original_w, original_h = optimization_result['original_dimensions']
                    optimized_w, optimized_h = optimization_result['optimized_dimensions']
                    st.info(f"サイズ: {original_w}×{original_h} → {optimized_w}×{optimized_h} (品質: {optimization_result['quality_used']})")
                    
                except Exception as e:
                    st.error(f"❌ {uploaded_image.name} の保存に失敗: {e}")
        
        # 現在の画像一覧を表示
        if st.button("📷 現在の画像一覧を表示"):
            import os
            if os.path.exists("pic"):
                image_files = [f for f in os.listdir("pic") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    st.write("**保存されている画像:**")
                    for img_file in image_files:
                        st.write(f"- {img_file}")
                else:
                    st.write("画像ファイルが見つかりません")
            else:
                st.write("picフォルダが存在しません")
        
        st.markdown("---")
        
        # APIキー管理セクション
        st.subheader("🔑 APIキー管理")
        
        # 現在のAPIキー状態を表示
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            api_key = st.secrets["openai"]["api_key"]
            if api_key.startswith("sk-or-v1-"):
                st.success("✅ OpenRouter APIキーが設定されています")
                st.info(f"キー: {api_key[:20]}...")
            else:
                st.warning("⚠️ APIキーの形式が正しくない可能性があります")
        else:
            st.error("❌ APIキーが設定されていません")
        
        st.info("**APIキーの設定方法:**\n1. [OpenRouter](https://openrouter.ai/)でアカウント作成\n2. APIキーを生成\n3. Streamlit CloudのSecretsに設定")
        
        # AIモデル選択セクション
        st.subheader("🤖 AIモデル選択")
        
        # デフォルトモデルを設定
        if "selected_model" not in st.session_state:
            st.session_state["selected_model"] = "claude-3-haiku"
        
        # モデル選択
        selected_model = st.selectbox(
            "使用するAIモデルを選択",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state["selected_model"]),
            help="コストと性能のバランスを考慮して選択してください"
        )
        
        # モデル情報を表示
        model_info = {
            "claude-3-haiku": "🚀 高速・低コスト（推奨）",
            "claude-3.5-sonnet": "🧠 高性能・中コスト",
            "gpt-3.5-turbo": "⚡ OpenAI標準モデル",
            "gpt-4o": "💎 OpenAI最高性能",
            "gemini-pro": "🔍 Google Gemini",
            "llama-3.1-8b": "🦙 Meta Llama（無料枠あり）",
            "qwen-2.5-7b": "🌟 Alibaba Qwen",
            "deepseek-chat": "🎯 DeepSeek（高品質）"
        }
        
        st.info(f"**選択中のモデル:** {model_info.get(selected_model, selected_model)}")
        
        # モデルをセッションに保存
        st.session_state["selected_model"] = selected_model
        
        st.markdown("---")
        
        # システム情報
        st.subheader("ℹ️ システム情報")
        st.write(f"**現在のデータ件数:** {len(data)} 件")
        st.write(f"**利用可能な気分:** {', '.join(data['今の気持ち'].unique()) if '今の気持ち' in data.columns else 'なし'}")
        
        # APIキー設定の詳細説明
        with st.expander("🔧 APIキー設定の詳細"):
            st.markdown("""
            **Streamlit CloudでのAPIキー設定:**
            
            1. **Streamlit Cloudにアクセス**
               - [https://share.streamlit.io/](https://share.streamlit.io/)
            
            2. **アプリの管理画面を開く**
               - 「Manage App」をクリック
            
            3. **Secrets設定**
               - 「Settings」→「Secrets」
               - 以下を追加:
               ```toml
               [openai]
               api_key = "sk-or-v1-あなたのAPIキー"
               
               [admin]
               password = "admin123"
               ```
            
            4. **保存して再起動**
               - 「Save」をクリック
               - 「Reboot app」をクリック
            """)