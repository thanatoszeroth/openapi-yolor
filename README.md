# openapi-yolor

Package
```sh
pip install -r requirements.txt
```

Run on Service 
```sh
ServiceIP=0.0.0.0
ServicePort=19191
uvicorn main:app --host ${ServiceIP} --port ${ServicePort} --reload
```

Export openapi.json
```sh
curl -O ServiceIP:ServicePort/openapi.json
```

Reference
- [ ] https://github.com/WongKinYiu/yolor