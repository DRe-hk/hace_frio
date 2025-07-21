# app.py
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, render_template
import datetime
import os

# --- 1. CONFIGURACIÓN (sin cambios) ---
RUTA_DATASET_ENTRADA = 'entrada.csv'
NOMBRE_MODELO_H5 = 'modelo_lstm_tiempo.h5'
NOMBRE_SCALER_PKL = 'scaler_tiempo.pkl'
HORAS_VENTANA_ENTRADA = 72
HORAS_VENTANA_SALIDA = 24
COLUMNAS_OBJETIVO = ['temperatura', 'precipitacion', 'wind_speed']

app = Flask(__name__)

# --- 2. FUNCIONES AUXILIARES (sin cambios) ---
def calcular_sensacion_termica(T, V):
    if V < 4.8 or T > 10:
        return T
    V = max(V, 0.1)
    wci = 13.12 + 0.6215 * T - 11.37 * (V**0.16) + 0.3965 * T * (V**0.16)
    return round(wci, 1)

def obtener_info_lluvia(precipitacion):
    if precipitacion < 0.2:
        return "Nula", None
    elif precipitacion < 1.0:
        return "Baja", f"Puede que haya un riego por aspersión meteorológico."
    elif precipitacion < 4.0:
        return "Moderada", f"Es probable que te resfríes si no llevas un paraguas."
    else:
        return "Extrema", f"Lleva un arca en la mochila porque se espera un diluvio."

def obtener_info_abrigo(sensacion_minima_dia):
    if sensacion_minima_dia < -5:
        imagen = "muy_abrigado.png"
        texto = f"No te recomiendo salir después de las 10 de la noche, no quiero asustarte pero podría darte una hipotermia. La mínima sensación será de {sensacion_minima_dia}°C."
    elif sensacion_minima_dia < 5:
        imagen = "muy_abrigado.png"
        texto = f"Hará mucho frío hoy, si estarás fuera abrígate bien. ¡Si es que no, también! La mínima sensación será de {sensacion_minima_dia}°C."
    elif sensacion_minima_dia < 10:
        imagen = "abrigado.png"
        texto = f"El día estará fresco. Un buen abrigo será tu mejor amigo. La mínima sensación térmica esperada es de {sensacion_minima_dia}°C."
    else:
        imagen = "normal.png"
        texto = f"Hoy no hará demasiado frío, así que puedes salir sin parecer un oso polar. La mínima sensación será de {sensacion_minima_dia}°C."
    return imagen, texto

# --- 3. LÓGICA DE PREDICCIÓN (sin cambios) ---
def realizar_prediccion():
    try:
        model = load_model(NOMBRE_MODELO_H5)
        with open(NOMBRE_SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)

        df_input = pd.read_csv(RUTA_DATASET_ENTRADA).tail(HORAS_VENTANA_ENTRADA).copy()
        df_input['fecha_hora'] = pd.to_datetime(df_input['fecha_hora'])
        df_input = df_input.sort_values(by='fecha_hora').reset_index(drop=True)
        df_input = df_input.replace('S/D', np.nan).dropna()
        if len(df_input) < HORAS_VENTANA_ENTRADA:
            raise ValueError(f"No hay suficientes datos. Se necesitan {HORAS_VENTANA_ENTRADA}, hay {len(df_input)}.")

        df_input['Año'] = df_input['fecha_hora'].dt.year
        df_input['Mes'] = df_input['fecha_hora'].dt.month
        df_input['Dia'] = df_input['fecha_hora'].dt.day
        df_input['Hora'] = df_input['fecha_hora'].dt.hour
        df_input['Hora_sin'] = np.sin(2 * np.pi * df_input['Hora'] / 24)
        df_input['Hora_cos'] = np.cos(2 * np.pi * df_input['Hora'] / 24)

        feature_columns = scaler.feature_names_in_
        df_features = df_input[feature_columns]
        input_scaled = scaler.transform(df_features)
        input_for_prediction = input_scaled.reshape(1, HORAS_VENTANA_ENTRADA, len(feature_columns))

        prediction_scaled_flat = model.predict(input_for_prediction)
        prediction_scaled = prediction_scaled_flat.reshape(HORAS_VENTANA_SALIDA, len(COLUMNAS_OBJETIVO))

        target_indices = [list(feature_columns).index(col) for col in COLUMNAS_OBJETIVO]
        dummy_array = np.zeros((HORAS_VENTANA_SALIDA, len(feature_columns)))
        for i, target_idx in enumerate(target_indices):
            dummy_array[:, target_idx] = prediction_scaled[:, i]
        predictions_descaled = scaler.inverse_transform(dummy_array)
        
        final_predictions = pd.DataFrame(predictions_descaled[:, target_indices], columns=COLUMNAS_OBJETIVO)
        return final_predictions
    except Exception as e:
        print(f"ERROR DURANTE LA PREDICCIÓN: {e}")
        return pd.DataFrame(columns=COLUMNAS_OBJETIVO)

# --- 4. RUTA PRINCIPAL DE LA APLICACIÓN (MODIFICADA) ---

@app.route('/')
def home():
    predicciones_df = realizar_prediccion()
    if predicciones_df.empty:
        return "Error al procesar los datos o cargar el modelo. Revisa la consola.", 500

    now = datetime.datetime.now()
    forecast_list = []
    start_time = now.replace(minute=0, second=0, microsecond=0)
    
    for i in range(HORAS_VENTANA_SALIDA):
        hora_actual = start_time + datetime.timedelta(hours=i)
        pred = predicciones_df.iloc[i]
        temp = round(pred['temperatura'], 1)
        precip = pred['precipitacion']
        wind_speed = pred['wind_speed']
        lluvia_cat, _ = obtener_info_lluvia(precip)

        forecast_list.append({
            "hora_obj": hora_actual,
            "hora_str": hora_actual.strftime('%H:%M'),
            "temperatura": temp,
            "sensacion_termica": calcular_sensacion_termica(temp, wind_speed),
            "prob_lluvia_cat": lluvia_cat
        })

    datos_hora_actual = forecast_list[0]
    sensacion_actual = datos_hora_actual['sensacion_termica']
    
    if sensacion_actual < 0: background_class = "bg-extremo"
    elif sensacion_actual < 5: background_class = "bg-muy-frio"
    elif sensacion_actual < 10: background_class = "bg-frio"
    elif sensacion_actual < 15: background_class = "bg-fresco"
    else: background_class = "bg-templado"

    # Recomendación de abrigo (sin cambios)
    min_sensacion_dia = min(f['sensacion_termica'] for f in forecast_list)
    main_image, main_text_abrigo = obtener_info_abrigo(min_sensacion_dia)

    # *** CAMBIO AQUÍ: Generar texto de lluvia por separado ***
    main_text_lluvia = None
    max_precip_index = predicciones_df['precipitacion'].idxmax()
    max_precip_value = predicciones_df['precipitacion'][max_precip_index]
    
    if max_precip_value >= 0.2: # Si hay alguna probabilidad de lluvia
        hora_lluvia = forecast_list[max_precip_index]['hora_obj'].strftime('%I:%M %p')
        _, texto_lluvia_generado = obtener_info_lluvia(max_precip_value)
        main_text_lluvia = f"¡Ojo! {texto_lluvia_generado} alrededor de las {hora_lluvia}."

    return render_template(
        'index.html',
        forecast=forecast_list,
        current_hour_str=now.strftime('%H:%M'),
        background_class=background_class,
        main_image_url=f'img/{main_image}',
        # Pasar ambos textos a la plantilla
        recommendation_abrigo=main_text_abrigo,
        recommendation_lluvia=main_text_lluvia
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)