import os
import uuid
import time
import re
import unicodedata
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
API_URL = os.getenv("API_URL", "http://localhost:7071/api/orc")  # igual que Postman
SIMILARITY_THRESHOLD = 0.7
TIMEOUT_SEC = 60
MAX_RETRIES = 3

# -----------------------
# Utilidades
# -----------------------
def clean_text(s: str) -> str:
    """Normaliza unicode, colapsa saltos raros y elimina caracteres de control."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\r\n", "\n")
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return s.strip()

def compute_similarity_cosine(text1: str, text2: str) -> float:
    """Similitud coseno TF‑IDF (sin torch)."""
    try:
        vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        X = vect.fit_transform([text1 or "", text2 or ""])
        sim = cosine_similarity(X[0], X[1])[0, 0]
        return float(sim)
    except Exception as e:
        print(f"[SIM] Error calculando similitud TF-IDF: {e}")
        return 0.0

# -----------------------
# HTTP client
# -----------------------
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
                rj = resp.json()  # si no es JSON válido, disparará excepción y reintentará
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

# -----------------------
# Pipeline Excel
# -----------------------
def procesar_excel_http(file_path: str):
    xls = pd.ExcelFile(file_path)
    resultados = []

    for sheet_name in xls.sheet_names:
        print(f"\n====================")
        print(f"Procesando hoja: {sheet_name}")
        print(f"====================")
        df_input = pd.read_excel(xls, sheet_name=sheet_name)

        # Validación de columnas
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
