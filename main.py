import logging
import os

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    """Start the uvicorn server serving the FastAPI application.

    Host and port can be overridden via the environment variables ``HOST``
    and ``PORT``. Defaults: 0.0.0.0:8000.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    try:
        logging.getLogger("uvicorn.error").propagate = True
        uvicorn.run("app.api:app", host=host, port=port, reload=True)
        return 0
    except Exception:
        logging.exception("Failed to start uvicorn. Ensure 'uvicorn' is installed.")
        return 1


if __name__ == "__main__":
    main()
