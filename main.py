import asyncio
from functions import procesar_excel, WebSocketClient, WS_URL_REPHRASE, WS_URL_SIMILARITY, WS_AUTH_TOKEN

if __name__ == "__main__":
    archivo_excel = "archivo.xlsx"

    async def main():
        ws_rephrase = WebSocketClient(WS_URL_REPHRASE, WS_AUTH_TOKEN)
        ws_similarity = WebSocketClient(WS_URL_SIMILARITY, WS_AUTH_TOKEN)
        await ws_rephrase.connect()
        await ws_similarity.connect()
        try:
            await procesar_excel(archivo_excel, ws_rephrase, ws_similarity)
        finally:
            await ws_rephrase.close()
            await ws_similarity.close()

    asyncio.run(main())
