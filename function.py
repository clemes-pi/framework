import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import requests

load_dotenv()

# SBERT para similitud coseno local
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Parámetros
SIMILARITY_THRESHOLD = 0.8
API_URL = "http://localhost:7071/api/orc"  # Cambia esto si tu endpoint es otro

def preguntar_api(question, conversation_id="cualquier_ID"):
    url = API_URL
    headers = {"Content-Type": "application/json"}
    body = {
        "conversation_id": conversation_id,
        "question": question
    }
    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        if response.status_code == 200:
            return response.json().get("answer", ""), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, str(e)

def compute_similarity_cosine(text1, text2):
    try:
        emb1 = model_sbert.encode(text1, convert_to_tensor=True)
        emb2 = model_sbert.encode(text2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except Exception as e:
        print(f"Error calculando similitud: {e}")
        return 0.0

def procesar_excel_http(file_path):
    xls = pd.ExcelFile(file_path)
    resultados = []

    for sheet_name in xls.sheet_names:
        print(f"Procesando hoja: {sheet_name}")
        df_input = pd.read_excel(xls, sheet_name=sheet_name)
        df_input["Respuesta esperada"] = df_input["Respuesta esperada"].astype(str)

        for idx, row in df_input.iterrows():
            fila_num = idx + 2
            pregunta = row.get("Pregunta")
            esperada = row.get("Respuesta esperada")

            if pd.isna(pregunta) or pd.isna(esperada):
                print(f"⚠️  Hoja {sheet_name} - Fila {fila_num}: falta pregunta o respuesta esperada")
                continue

            print(f"Procesando fila {fila_num}: {pregunta}")
            start_time = time.time()
            respuesta, error = preguntar_api(pregunta)
            tiempo = round(time.time() - start_time, 2)

            if respuesta:
                similitud_coseno = compute_similarity_cosine(respuesta, esperada)
                resultados.append({
                    "Hoja": sheet_name,
                    "Fila": fila_num,
                    "Pregunta": pregunta,
                    "Respuesta esperada": esperada,
                    "Respuesta API": respuesta,
                    "Similitud Coseno": round(similitud_coseno, 4),
                    "Tiempo (s)": tiempo,
                    ">0.8 Coseno": similitud_coseno > SIMILARITY_THRESHOLD,
                })
            else:
                resultados.append({
                    "Hoja": sheet_name,
                    "Fila": fila_num,
                    "Pregunta": pregunta,
                    "Respuesta esperada": esperada,
                    "Respuesta API": f"[Error] {error}",
                    "Similitud Coseno": 0.0,
                    "Tiempo (s)": tiempo,
                    ">0.8 Coseno": False,
                })

    df_result = pd.DataFrame(resultados)
    output_path = "reporte_http_api.xlsx"
    df_result.to_excel(output_path, index=False)
    print(f"✅ Reporte generado: {output_path}")

if __name__ == "__main__":
    procesar_excel_http("input.xlsx")
