import requests
import uuid

API_URL = "http://localhost:7071/api/orc"  # igual que Postman

def preguntar_api():
    body = {
        "conversation_id": str(uuid.uuid4()),  # evita problemas por reuso de ID
        "question": "¿Dónde puedo encontrar datos del uso de los medios de pagos de estacionamientos (por aeropuerto)? (Detallado: medio de mayor uso, dentro de cada medio las ubicaciones más usadas. Información diaria)."
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",  # evita compresión rara
        "Connection": "close"
    }
    r = requests.post(API_URL, headers=headers, json=body, timeout=60)
    print("STATUS:", r.status_code)
    print("BODY:", r.text)

preguntar_api()
