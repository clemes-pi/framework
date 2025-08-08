import os
import uuid
import time
import re
import unicodedata
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
API_URL = os.getenv("API_URL", "http://localhost:7071/api/orc")  # igual que Postman
SIMILARITY_THRESHOLD = 0.7
TIMEOUT_SEC = 60
MAX_RETRIES = 3

# SBERT para similitud coseno local
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(s: str) -> str:
    """Normaliza unicode, colapsa saltos raros y elimina caracteres de control."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\r\n", "\n")
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return s.strip()

def preguntar_api(question: str):
    """POST al endpoint con headers ‘seguros’, conversation_id único y reintentos."""
    url = API_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",  # evita compresión rara
        "Connection": "close"           # evita keep-alive en servers sensibles
    }
    question = clean_text(question)

    for intento in range(MAX_RETRIES):
        body = {
            "conversation_id": str(uuid.uuid4()),  # único por intento
            "question": question
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SEC)
            print(f"[HTTP] intento {intento+1}/{MAX_RETRIES} -> {resp.status_code}")
            if resp.status_code == 200:
                # Nota: si el backend no devuelve JSON válido, esto levantará excepción (y reintentará)
                rj = resp.json()
                return rj.get("answer", ""), None
            # Log de error del servidor/cliente
            print("[HTTP] error body:", (resp.text or "")[:2000])
            # Reintenta solo en 5xx
            if 500 <= resp.status_code < 600 and intento < MAX_RETRIES-1:
                time.sleep(2 * (intento + 1))
                continue
            return None, f"HTTP {resp.status_code}: {resp.text}"
        except Exception as ex:
            print("[HTTP] exception:", ex)
            if intento < MAX_RETRIES-1:
                time.sleep(2 * (intento + 1))
                continue
            return None, f"Exception: {ex}"

def compute_similarity_cosine(text1, text2):
    try:
        emb1 = model_sbert.encode(text1 or "", convert_to_tensor=True)
        emb2 = model_sbert.encode(text2 or "", convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except Exception as e:
        print(f"[SIM] Error calculando similitud: {e}")
        return 0.0

def procesar_excel_http(file_path):
    xls = pd.ExcelFile(file_path)
    resultados = []

    for sheet_name in xls.sheet_names:
        print(f"Procesando hoja: {sheet_name}")
        df_input = pd.read_excel(xls, sheet_name=sheet_name)
        # Asegurar columnas
        if "Pregunta" not in df_input.columns or "Respuesta esperada" not in df_input.columns:
            print(f"⚠️  La hoja '{sheet_name}' no tiene columnas requeridas: 'Pregunta' y 'Respuesta esperada'")
            continue

        df_input["Respuesta esperada"] = df_input["Respuesta esperada"].astype(str)

        for idx, row in df_input.iterrows():
            fila_num = idx + 2
            pregunta = row.get("Pregunta")
            esperada = row.get("Respuesta esperada")

            if pd.isna(pregunta) or pd.isna(esperada):
                print(f"⚠️  Hoja {sheet_name} - Fila {fila_num}: falta pregunta o respuesta esperada")
                continue

            pregunta = str(pregunta)
            print(f"\nFila {fila_num} - Pregunta: {pregunta[:140]}{'...' if len(pregunta)>140 else ''}")

            start_time = time.time()
            respuesta, error = preguntar_api(pregunta)
            tiempo = round(time.time() - start_time, 2)

            # Log en consola (recorta para no inundar)
            if respuesta:
                print(f"Fila {fila_num} - Respuesta (inicio): {respuesta[:200]}{'...' if len(respuesta)>200 else ''}")
                similitud_coseno = compute_similarity_cosine(respuesta, esperada)
                resultados.append({
                    "Hoja": sheet_name,
                    "Fila": fila_num,
                    "Pregunta": pregunta,
                    "Respuesta esperada": esperada,
                    "Respuesta API": respuesta,
                    "Similitud Coseno": round(similitud_coseno, 4),
                    "Tiempo (s)": tiempo,
                    f">{SIMILARITY_THRESHOLD} Coseno": similitud_coseno > SIMILARITY_THRESHOLD,
                })
            else:
                print(f"Fila {fila_num} - Error: {error}")
                resultados.append({
                    "Hoja": sheet_name,
                    "Fila": fila_num,
                    "Pregunta": pregunta,
                    "Respuesta esperada": esperada,
                    "Respuesta API": f"[Error] {error}",
                    "Similitud Coseno": 0.0,
                    "Tiempo (s)": tiempo,
                    f">{SIMILARITY_THRESHOLD} Coseno": False,
                })

    df_result = pd.DataFrame(resultados)
    output_path = "reporte_http_api.xlsx"
    df_result.to_excel(output_path, index=False)
    print(f"\n✅ Reporte generado: {output_path}")

if __name__ == "__main__":
    procesar_excel_http("input.xlsx")
