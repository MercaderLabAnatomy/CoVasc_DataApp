from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the streamlit app"""
    Popen(["streamlit", "Untitled", "--browser.serverAddress=0.0.0.0", "--server.enableCORS=False"])
