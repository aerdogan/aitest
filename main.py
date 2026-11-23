
import os
import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 1. Ortam değişkenlerini yükle
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY bulunamadı. .env dosyasını kontrol et.")

# 2. Yapay zeka modeli
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


#web araması

@tool
def search_on_web(query) -> str:
    """
    Müşteri tipini doğrulamak için webden findex puanına bakan web metodu.
    """    
    print(f"Web araması yapılıyor..... : {query}")
    return f"Web araması yapıldı: {query}. (Bu bir simülasyondur, gerçek arama yapılmadı.)"


# 3. İş kuralları

@tool
def calculate_order(amount: float, is_vip: bool) -> str:
    """
    Sipariş tutarına ve müşteri tipine göre
    indirim ve nihai tutarı hesaplar.
    """

    if amount < 500:
        return json.dumps({
            "approved": False,
            "reason": "Minimum sipariş tutarı 500 TL"
        }, ensure_ascii=False)

    discount = 0

    if amount > 1000:
        discount += 10

    if is_vip:
        discount += 5

    final_amount = amount * (1 - discount / 100)

    return json.dumps({
        "approved": True,
        "original_amount": amount,
        "discount": discount,
        "final_amount": round(final_amount, 2)
    }, ensure_ascii=False)


# 4. Agent sistem talimatları

system_prompt = """
Sen bir sipariş değerlendirme yapay zeka ajanısın.

Kurallar:

- Kullanıcının sipariş talebini analiz et.
- Sipariş tutarını ve VIP bilgisini belirle.
- Hesaplama işlemini mutlaka calculate_order
  aracıyla gerçekleştir.
- Web araması yapmak için search_on_web
  aracıyla gerçekleştir.
- Kesinlikle müşterileri web araması ile doğrulama yapmadan siparişi onaylama!
- Kendin indirim hesabı yapma.
- Araçtan dönen sonucu değiştirme.
- Eksik bilgi varsa kullanıcıdan bilgi iste.
- Sonucu anlaşılır şekilde açıkla.
- Gerçekleşmemiş bir siparişi onaylanmış gibi gösterme.
"""

# 5. Agent oluştur
agent = create_agent(
    model=llm,
    tools=[search_on_web, calculate_order],
    system_prompt=system_prompt
)


# 6. Komut satırı döngüsü

def main():

    print("=" * 50)
    print("SİPARİŞ DEĞERLENDİRME SİSTEMİ")
    print("=" * 50)
    print("Sipariş bilgilerini yazabilirsin.")
    print("Çıkmak için 'exit' veya 'quit' yaz.\n")

    while True:

        user_input = input("Sipariş > ").strip()

        # Boş giriş kontrolü
        if not user_input:
            continue

        # Çıkış kontrolü
        if user_input.lower() in ["exit", "quit", "çıkış"]:
            print("\nProgram kapatılıyor...")
            break

        try:
            response = agent.invoke({
                "messages": [
                    ("user", user_input)
                ]
            })

            final_message = response["messages"][-1]

            print("\nAI:", final_message.content)
            print("-" * 50)

        except Exception as e:
            print("\nHATA:", type(e).__name__, str(e))
            print("-" * 50)


if __name__ == "__main__":
    main()
