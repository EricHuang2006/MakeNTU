import ssl

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="server.crt", keyfile="server.key")

raw_client, addr = server_socket.accept()
client_socket = context.wrap_socket(raw_client, server_side=True)

