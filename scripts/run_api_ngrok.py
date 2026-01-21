import nest_asyncio
nest_asyncio.apply()

import socket, time
import uvicorn
from pyngrok import ngrok, conf
from api.server import app
from config.configs import NGROK_TOKEN

def free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

port = free_port()
conf.get_default().auth_token = NGROK_TOKEN
public_url = ngrok.connect(port)

print("🚀 Public API URL:", public_url.public_url + "/generate")
uvicorn.run(app, host="0.0.0.0", port=port)
