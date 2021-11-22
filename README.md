# Proyecto 2 - Data Science.
Predicción de clientes que incurren en default payment basado en su historial crediticio.

* José Miguel Castañeda 18282
* Douglas de León       18037
* Rodrigo Garoz         18102
* Gerardo Méndez        18239

## Reproducción
1. Descargar o clonar el repositorio
2. Instalar librerías. Para instalarlas con pip, en el directorio principal del repositorio ejecutar: <pre>C:\ pip install -r requirements.txt</pre>
3. Para revisar el análisis exploratorio y evaluación de modelos ejecutar el comando y correr los notebooks 'proyecto2.ipynb' y 'models.ipynb'<pre>C:\ jupyter notebook</pre> 
5. Para resultados interactivos: <pre>C:\ streamlit run resultados.py</pre>

**Nota:** Si la instrucción `run_app(data, mode='external')` en el notebook 'models.ipynb' produce error, es posible que se deba a un problema de compatibilidad con la librería dash. En ese caso es necesario ejecutar: <pre>C:\ pip install "dash-bootstrap-components<1"</pre>
