
from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the streamlit app"""
    Popen(
        [
            "streamlit", 
            "run", 
            "CoVasc_DataApp.py", 
            "--browser.serverAddress=0.0.0.0", 
            "--server.enableCORS=False",
            "--server.enableWebsocketCompression=false",
            "--server.enableXsrfProtection=false",
        ]
    )
