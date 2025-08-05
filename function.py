import os
import time
import asyncio
import json
import ssl
import signal
import sys
import uuid

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import websockets

load_dotenv()

# SBERT para similitud coseno local
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Parámetros
N_REPHRASES = 3
SIMILARITY_THRESHOLD = 0.8

# WebSocket config
WS_URL_REPHRASE = os.getenv("WEBSOCKET_URL_REPHRASE")
WS_URL_SIMILARITY = os.getenv("WEBSOCKET_URL_SIMILARITY")
WS_AUTH_TOKEN = os.getenv("WS_AUTH_TOKEN")

# Control de cierre
stop = False
def handle_exit(signum, frame):
    global stop
    stop = True
    print("Señal de salida recibida, cerrando...")

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


class WebSocketClient:
    def __init__(self, url, token=None):
        self.url = url
        self.token = token
        self.ws = None
        self.pending = {}
        self.lock = asyncio.Lock()

async def connect(self):
    headers = {}
    if self.token:
        headers["Authorization"] = f"Bearer {self.token}"
    ssl_context = None
    if self.url.startswith("wss://"):
        ssl_context = ssl.create_default_context()
    try:
        self.ws = await websockets.connect(self.url, extra_headers=headers, ssl=ssl_context)
        print(f"[WS] Conectado a {self.url}")
        asyncio.create_task(self._listener())
    except Exception as e:
        print(f"[WS] Error conectando: {e}")
        self.ws = None


    async def _listener(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    print("[WS] Mensaje no JSON:", raw)
                    continue
                corr = msg.get("correlation_id")
                if corr and corr in self.pending:
                    fut = self.pending.pop(corr)
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    print("[WS] Mensaje sin pending match:", msg)
        except Exception as e:
            print(f"[WS] Listener terminado: {e}")

    async def request(self, event_type, payload, timeout=30):
        if not self.url:
            raise RuntimeError("No WEB_SOCKET URL configurado")
        if self.ws is None:
            await self.connect()
        if self.ws is None:
            raise RuntimeError("No se pudo conectar al WebSocket")
        correlation_id = str(uuid.uuid4())
        message = {
            "type": event_type,
            "correlation_id": correlation_id,
            "payload": payload,
            "timestamp": time.time()
        }
        fut = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.pending[correlation_id] = fut
            try:
                await self.ws.send(json.dumps(message))
            except Exception as e:
                self.pending.pop(correlation_id, None)
                raise
        try:
            response = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending.pop(correlation_id, None)
            raise RuntimeError(f"Timeout esperando respuesta para {event_type}")
        return response

    async def close(self):
        if self.ws:
            await self.ws.close()
            print("[WS] Cerrado")


# Funciones locales
async def compute_similarity_cosine(text1, text2):
    def sync():
        emb1 = model_sbert.encode(text1, convert_to_tensor=True)
        emb2 = model_sbert.encode(text2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    return await asyncio.to_thread(sync)

# Wrappers
async def rephrase_question_ws(question, previous_rephrasings, ws_client: WebSocketClient):
    payload = {
        "question": question,
        "previous_rephrasings": previous_rephrasings
    }
    resp = await ws_client.request("rephrase_question", payload)
    return resp.get("result", "").strip()

async def generate_answer_ws(question, ws_client: WebSocketClient):
    start = time.time()
    payload = {"question": question}
    resp = await ws_client.request("generate_answer", payload)
    answer = resp.get("result", "").strip()
    elapsed = round(time.time() - start, 2)
    return answer, elapsed

async def compute_similarity_llm_ws(pregunta, respuesta1, respuesta2, ws_client: WebSocketClient):
    payload = {
        "pregunta": pregunta,
        "respuesta1": respuesta1,
        "respuesta2": respuesta2
    }
    resp = await ws_client.request("similarity_llm", payload)
    try:
        return float(resp.get("result", 0.0))
    except Exception:
        return 0.0

async def procesar_excel(file_path, ws_rephrase: WebSocketClient, ws_similarity: WebSocketClient):
    xls = pd.ExcelFile(file_path)
    resultados = []
    for sheet_name in xls.sheet_names:
        print(f"Procesando hoja: {sheet_name}")
        df_input = pd.read_excel(xls, sheet_name=sheet_name)
        df_input["Respuesta esperada"] = df_input["Respuesta esperada"].astype(str)
        for idx, row in df_input.iterrows():
            if stop:
                break
            fila_num = idx + 2
            pregunta = row.get("Pregunta")
            esperada = row.get("Respuesta esperada")
            print(f"Procesando fila {fila_num}: {pregunta}")
            if pd.isna(pregunta) or pd.isna(esperada):
                print(f"⚠️  Hoja {sheet_name} - Fila {fila_num}: falta pregunta o respuesta esperada")
                continue
            similitudes_coseno = []
            similitudes_llm = []
            previous_rephrasings = []
            for i in range(N_REPHRASES):
                try:
                    # Usar ws_rephrase y ws_similarity según corresponda
                    pregunta_reformulada = await rephrase_question_ws(pregunta, previous_rephrasings, ws_rephrase)
                    respuesta, tiempo = await generate_answer_ws(pregunta_reformulada, ws_rephrase)
                    similitud_coseno = await compute_similarity_cosine(respuesta, esperada)
                    similitud_llm = await compute_similarity_llm_ws(pregunta, respuesta, esperada, ws_similarity)
                    similitudes_coseno.append(similitud_coseno)
                    similitudes_llm.append(similitud_llm)
                    previous_rephrasings.append(pregunta_reformulada)
                    entry = {
                        "Hoja": sheet_name,
                        "Fila": fila_num,
                        "Pregunta original": pregunta,
                        "Respuesta esperada": esperada,
                        "Pregunta reformulada": pregunta_reformulada,
                        "Respuesta generada": respuesta,
                        "Similitud Coseno": round(similitud_coseno, 4),
                        "Similitud LLM": round(similitud_llm, 4),
                        "Tiempo (s)": tiempo,
                        ">0.8 Coseno": similitud_coseno > SIMILARITY_THRESHOLD,
                        ">0.8 LLM": similitud_llm > SIMILARITY_THRESHOLD
                    }
                    resultados.append(entry)
                    print(f"  Reformulación {i+1}: coseno {similitud_coseno:.4f}, llm {similitud_llm:.4f}, tiempo {tiempo}s")
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Error en hoja {sheet_name} - fila {fila_num}: {error_msg}")
                    err_entry = {
                        "Hoja": sheet_name,
                        "Fila": fila_num,
                        "Pregunta original": pregunta,
                        "Pregunta reformulada": previous_rephrasings[-1] if previous_rephrasings else "",
                        "Respuesta generada": "[Error]",
                        "Respuesta esperada": esperada,
                        "Similitud Coseno": 0.0,
                        "Similitud LLM": 0.0,
                        "Tiempo (s)": "",
                        ">0.8 Coseno": False,
                        ">0.8 LLM": False,
                        "Error": error_msg
                    }
                    resultados.append(err_entry)
            # Agregar promedios
            if similitudes_coseno or similitudes_llm:
                prom_coseno = round(np.mean(similitudes_coseno), 4) if similitudes_coseno else 0.0
                prom_llm = round(np.mean(similitudes_llm), 4) if similitudes_llm else 0.0
                promedio_entry = {
                    "Hoja": sheet_name,
                    "Fila": fila_num,
                    "Pregunta original": pregunta,
                    "Pregunta reformulada": "- PROMEDIO -",
                    "Respuesta generada": "",
                    "Respuesta esperada": esperada,
                    "Similitud Coseno": prom_coseno,
                    "Similitud LLM": prom_llm,
                    "Tiempo (s)": "",
                    ">0.8 Coseno": prom_coseno > SIMILARITY_THRESHOLD,
                    ">0.8 LLM": prom_llm > SIMILARITY_THRESHOLD
                }
                resultados.append(promedio_entry)
        if stop:
            break
    df_result = pd.DataFrame(resultados)
    output_path = "reporte_ws_only.xlsx"
    df_result.to_excel(output_path, index=False)
    print(f"✅ Reporte generado: {output_path}")

async def main():
    ws_rephrase = WebSocketClient(WS_URL_REPHRASE, WS_AUTH_TOKEN)
    ws_similarity = WebSocketClient(WS_URL_SIMILARITY, WS_AUTH_TOKEN)
    await ws_rephrase.connect()
    await ws_similarity.connect()
    try:
        await procesar_excel("input.xlsx", ws_rephrase, ws_similarity)
    finally:
        await ws_rephrase.close()
        await ws_similarity.close()

if __name__ == "__main__":
    asyncio.run(main())
