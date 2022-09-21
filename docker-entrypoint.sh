echo "========== ========== =========="
echo "Init ENV"
if [ ${ServiceDomainName} == '' ]
then
  ServiceDomainName = 0.0.0.0
fi
if [ ${ServiceDomainPort} == '' ]
then
  ServiceDomainPort = 19191
fi
echo "Service listening on http://${ServiceDomainName}:${ServiceDomainPort}"
echo "========== ========== =========="
cd /app
echo "Run API YOLOR Service"
uvicorn main:app --host ${ServiceDomainName} --port ${ServiceDomainPort}
echo "========== ========== =========="
