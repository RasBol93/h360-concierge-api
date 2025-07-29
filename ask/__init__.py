import json
import azure.functions as func
from main import ask_bot  # o donde esté tu lógica

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON", status_code=400)
    result = ask_bot(payload)
    return func.HttpResponse(json.dumps(result), mimetype="application/json")
