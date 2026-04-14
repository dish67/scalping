import json
import time
from urllib.error import URLError
from urllib.request import urlopen


API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"


def get_bitcoin_price():
    with urlopen(API_URL, timeout=10) as response:
        data = json.load(response)
    return data["bitcoin"]["usd"]


print("Scalping prêt. Récupération du prix du Bitcoin...")

while True:
    try:
        price = get_bitcoin_price()
        print(f"Prix du Bitcoin: {price} USD")
    except (URLError, KeyError, ValueError) as error:
        print(f"Erreur lors de la récupération du prix: {error}")

    time.sleep(5)
