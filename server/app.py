from gst_env.main import app
import uvicorn

__all__ = ["app"]


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()