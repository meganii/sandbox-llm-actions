from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

def main():
    # MODEL
    model = OllamaLLM(model="gemma3:4b")

    # PROMPT
    prompt = PromptTemplate(
        input_variables=["text"],
        template='''あなたは優秀な要約生成アシスタントです。
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
全体的な傾向: 全体的な傾向についてコメント

## テキスト内容
{text}

'''
    )
    # Webページの内容をロード
    loader = WebBaseLoader("https://scrapbox.io/api/pages/villagepump/2025%2F03%2F21/text")
    docs = loader.load()

    # CharacterTextSplitter の設定
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",  # セパレータ
        chunk_size=5000,  # チャンクの最大サイズ (必要に応じて調整)
        # chunk_overlap=200, # チャンク間のオーバーラップ (必要に応じて調整)
        # length_function=len, # 文字列の長さを測る関数
    )

    # テキストを分割
    # split_docs = text_splitter.split_documents(docs)
    # print('\nCharacterTextSplitterによる分割結果: \n')
    # for i, chunk in enumerate(split_docs):
    #     print(f"Chunk {i+1} (\n{chunk}\n")

    # CHAIN
    chain = load_summarize_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )
    result = chain.invoke(docs)
    print(result['output_text'])

if __name__ == "__main__":
    main()