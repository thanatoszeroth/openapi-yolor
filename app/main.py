import os
import time
import sys
from fastapi import FastAPI
from fastapi import Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 

from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('.env')
load_dotenv(dotenv_path = dotenv_path)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title = "CL Hsiao API",
        version = "1.0.0",
        description = "This is a very custom OpenAPI schema",
        contact = {
            "name": "CL, Hsiao",
            "url": "https://github.com/thanatoszeroth",
            "email": "thanatoszeroth@gmail.com.tw",
        },
        license_info = {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        routes = app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Main APP
app = FastAPI()
# Swagger UI Note
app.openapi = custom_openapi
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# Static File
app.mount("/static", StaticFiles(directory = "static"), name = "static")


@app.get("/", tags = ["Index"])
async def index():
    """
    Export swagger docs to json file.   
    File : openapi.json  
    ```sh
    curl -O ServiceIP:ServicePort/openapi.json
    ```
    """
    ServiceIP = os.getenv("ServiceIP")
    ServicePort = os.getenv("ServicePort")
    return {"api_docs": f"http://{ServiceIP}:{ServicePort}/docs"}

@app.get("/readme", tags = ["Index"])
async def readme():
    """
    """
    try:
        return {"readme": f"© 2022 CL Hsiao. All rights reserved."}
    except Exception as e:
        return {"message": f"{e}"}


@app.get("/get_ip", tags = ["Index"])
def get_ip(request: Request):
    """
    Get IP  
    """
    return {
        "ip": f"{request.client.host}",
        # "x-real-ip": request.header.get("x-real-ip", ""),
        # "x-forwarded-for": request.header.get("x-forwarded-for", "")}
    }

@app.get("/get_server_time", tags = ["Index"])
def get_server_time(request: Request):
    """
    Get Server Time  
    """
    return {
        "time": f"{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())}",
    }

# API Routes and Modules
from YOLOR import api4yolor
try:
    os.makedirs(f"static/tmp")
except FileExistsError:
    print("Folder Exist")
app.include_router(api4yolor.router)
