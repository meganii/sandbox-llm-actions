from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain



def main():
    # MODEL
    model = OllamaLLM(model="gemma3:4b")

    # PROMPT
    prompt = PromptTemplate(
        input_variables=["text"],
        template="以下のWEBページの本文部分を日本語で要約してください:\n\n{text}\n\n要約:"
    )
    # Webページの内容をロード
    loader = WebBaseLoader("https://scrapbox.io/api/pages/villagepump/2025%2F03%2F21/text")
    docs = loader.load()

    # CHAIN
    chain = load_summarize_chain(model, chain_type="stuff", prompt=prompt)
    result = chain.invoke(docs)
    print(result['output_text'])

if __name__ == "__main__":
    main()