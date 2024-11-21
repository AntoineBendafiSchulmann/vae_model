from app import create_app

if __name__ == "__main__":
    app = create_app()

    port = 5004
    host = "127.0.0.1"

    print(f"API disponible à l'adresse : http://{host}:{port}/")
    print(f"Endpoint pour la reconstruction : http://{host}:{port}/api/image/reconstruct")

    app.run(host=host, port=port, debug=True, use_reloader=False)
