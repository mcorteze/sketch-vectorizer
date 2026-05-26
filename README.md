# sketch-vectorizer

Limpia bocetos escaneados y los convierte a trazos profesionales para colorear.

Entrada: foto/escaneo de dibujo a mano (jpg, png, bmp, tiff)
Salida: PNG limpio en blanco y negro + SVG vectorial

## Instalación

```bash
cd c:\Proyectos\sketch-vectorizer
pip install -r requirements.txt
```

## Uso básico

```bash
python vectorizar.py mi_boceto.jpg
```

Genera `mi_boceto_limpio.png` y `mi_boceto_limpio.svg` en la misma carpeta.

## Opciones

```bash
# Guardar en otra carpeta
python vectorizar.py boceto.jpg --output-dir resultados/

# Ajustar limpieza (para imágenes muy sucias)
python vectorizar.py boceto.jpg --denoise 20 --c-value 12

# Vectorización poligonal (trazos más angulares, menos suavizado)
python vectorizar.py boceto.jpg --svg-mode polygon

# Ver todas las opciones
python vectorizar.py --help
```

## Parámetros de ajuste fino

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `--denoise N` | 10 | Fuerza anti-ruido. Subir si hay mucho grano. |
| `--block-size N` | 15 | Sensibilidad al umbral local. Subir para iluminación desigual. |
| `--c-value N` | 8 | Cuánto gris eliminar. Subir para limpiar más, bajar si se pierden trazos finos. |
| `--noise-size N` | 2 | Tamaño mínimo de manchas a eliminar (px). |
| `--speckle N` | 4 | Elimina manchas del SVG menores a N px. |
| `--svg-mode` | spline | `spline` = curvas Bezier suaves, `polygon` = poligonal. |
