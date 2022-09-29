echo "========== ========== =========="
echo "Init ENV"
if [ ${ServiceIP} == '' ]
then
  ServiceIP = 0.0.0.0
fi
if [ ${ServicePort} == '' ]
then
  ServicePort = 19191
fi
echo "Service listening on http://${ServiceIP}:${ServicePort}"
echo "========== ========== =========="
cd /app
echo "Run API YOLOR Service"
uvicorn main:app --host ${ServiceIP} --port ${ServicePort}
echo "========== ========== =========="
