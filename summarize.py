import requests
from datetime import datetime
import urllib.parse

API_SERVER_URL = "http://localhost:11434/api/generate"

def main():    
    today = datetime.today().strftime('%Y/%m/%d')
    response = requests.get(f'https://scrapbox.io/api/pages/villagepump/{urllib.parse.quote_plus(today)}/text')

    headers = {"Content-Type": "application/json"}
    json = {
        "model": "gemma3:12b",
        "system": '''あなたは優秀な要約生成アシスタントです。
次のテキストはコラボレーションツールCosenseに作成された「井戸端」プロジェクトの日記ページの内容です。

内容を読み込み、主要なポイントと重要な情報を網羅した要約を、日本語で出力例に即して作成してください。
また、元の文書の文脈やニュアンスをできるだけ正確に反映するよう心掛けてください。

## 注意事項
- できるだけ冗長にならず、主要な点を簡潔に記述する。
- 数字や固有名詞など重要な情報は正確に残す。
- 誤解が生じないよう、必要な説明を適宜補足する。
- Cosenseは、次の特徴があることに留意すること。
    - [LINK]は、ページへのリンクを表現する。
    - [example.icon]は、アイコンを表現する。

## 出力例
概要: 全体概要のまとめ
主な話題: 主な話題をピックアップして、それぞれの要約
全体的な傾向: 全体的な傾向についてコメント''',
        "prompt": response.text,
        "stream": False
    }

    response = requests.post(API_SERVER_URL, headers=headers, json=json)
    response.raise_for_status()
    
    print(response.json()['response'])

main()

