# Usa una imagen base de Python oficial, especifica la versión
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia tus archivos de requisitos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación
COPY . .

# Comando para iniciar la aplicación (usa gunicorn)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
