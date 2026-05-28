"""
sketch-vectorizer — servidor web
Carga un boceto, lo limpia automaticamente y sirve la interfaz de pintura por capas.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def limpiar_trazo(img_bgr: np.ndarray, umbral_manual: int = None) -> np.ndarray:
    """
    Limpia el boceto y retorna escala de grises: 0=trazo, 255=fondo.
    umbral_manual: 0-255. Bajo = trazo fino (solo negro puro),
                           Alto = trazo grueso (incluye grises oscuros).
                   None  = Otsu automatico (bocetos a lapiz/tinta).
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gris.shape
    if max(h, w) < 1200:
        factor = 1200 / max(h, w)
        gris = cv2.resize(gris, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)

    suave = cv2.bilateralFilter(gris, d=5, sigmaColor=20, sigmaSpace=5)

    kernel_fondo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    fondo = cv2.morphologyEx(suave, cv2.MORPH_DILATE, kernel_fondo)
    normalizado = cv2.divide(suave.astype(np.float32), fondo.astype(np.float32) + 1e-6)
    normalizado = np.clip(normalizado * 255, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contraste = clahe.apply(normalizado)

    blur_aa = cv2.GaussianBlur(contraste, (3, 3), 0.8)

    if umbral_manual is not None:
        # Umbral fijo: el usuario controla qué tan oscuro tiene que ser un pixel
        # para considerarse trazo. Valor alto = captura trazos mas grises/gruesos.
        _, umbral = cv2.threshold(blur_aa, 255 - umbral_manual, 255, cv2.THRESH_BINARY)
    else:
        # Otsu automatico — bueno para bocetos b/n con fondo claro
        _, umbral = cv2.threshold(blur_aa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_limpieza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    limpio = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel_limpieza)

    limpio_f = limpio.astype(np.float32)
    suavizado = cv2.GaussianBlur(limpio_f, (0, 0), 1.2)
    final = np.clip(suavizado * 0.7 + limpio_f * 0.3, 0, 255).astype(np.uint8)

    return final


def img_a_base64(img_bw: np.ndarray) -> str:
    pil = Image.fromarray(img_bw)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def flood_fill_zona(img_bw: np.ndarray, x: int, y: int, color_rgba: list, tolerancia: int = 15) -> np.ndarray:
    """
    Rellena la zona contigua al punto (x,y) con el color dado.
    Retorna imagen RGBA con la zona rellena y el resto transparente.
    Tolerancia: cuantos px de gris/huecos se atraviesan al expandir.
    """
    h, w = img_bw.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return None

    # Crear mascara de trazo: 255=trazo (bloqueado), 0=espacio libre
    # Umbral duro para que grises intermedios no frenen el fill
    _, trazo_mask = cv2.threshold(img_bw, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilatar el trazo levemente para cerrar huecos pequeños
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerancia // 3 + 1, tolerancia // 3 + 1))
    trazo_cerrado = cv2.dilate(trazo_mask, kernel, iterations=1)

    # Flood fill sobre la mascara cerrada
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    fill_input = trazo_cerrado.copy()

    seed_val = int(fill_input[y, x])
    if seed_val == 255:
        # Click sobre un trazo: no hacer nada
        return None

    cv2.floodFill(fill_input, fill_mask, (x, y), 128, loDiff=tolerancia, upDiff=tolerancia)

    zona = (fill_input == 128).astype(np.uint8) * 255

    # Revertir la dilatacion para no pintar encima del trazo original
    zona = cv2.erode(zona, kernel, iterations=1)

    # Construir RGBA
    r, g, b, a = color_rgba
    resultado = np.zeros((h, w, 4), dtype=np.uint8)
    resultado[zona == 255] = [r, g, b, a]

    buf = io.BytesIO()
    Image.fromarray(resultado, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cargar", methods=["POST"])
def cargar():
    if "imagen" not in request.files:
        return jsonify({"error": "No se recibio imagen"}), 400

    archivo = request.files["imagen"]
    if not archivo or archivo.filename == "":
        return jsonify({"error": "Archivo vacio o sin nombre"}), 400

    # Leer directamente en memoria — evita problemas con nombres de archivo
    # especiales, rutas con espacios, o caracteres unicode
    try:
        buf = archivo.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"No se pudo decodificar la imagen: {e}"}), 400

    if img is None:
        return jsonify({"error": "Formato de imagen no soportado"}), 400

    # Parametros de limpieza enviados desde el modal
    umbral_manual = request.form.get("umbral", None)
    umbral = int(umbral_manual) if umbral_manual else None

    try:
        limpia = limpiar_trazo(img, umbral_manual=umbral)
    except Exception as e:
        return jsonify({"error": f"Error al procesar imagen: {e}"}), 500

    h, w = limpia.shape
    return jsonify({
        "trazo_b64": img_a_base64(limpia),
        "ancho": w,
        "alto": h,
    })


@app.route("/fill", methods=["POST"])
def fill():
    data = request.get_json()
    trazo_b64 = data["trazo_b64"]
    x = int(data["x"])
    y = int(data["y"])
    color = data["color"]   # [r, g, b, a]
    tolerancia = int(data.get("tolerancia", 15))

    img_bytes = base64.b64decode(trazo_b64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_bw = np.array(img_pil)

    resultado = flood_fill_zona(img_bw, x, y, color, tolerancia)
    if resultado is None:
        return jsonify({"error": "click sobre trazo o fuera de imagen"}), 400

    return jsonify({"capa_b64": resultado})


@app.route("/varita", methods=["POST"])
def varita():
    """
    Devuelve la mascara de la region delimitada por el trazo en el punto (x,y).
    El frontend la usa para borrar (destination-out) lo que haya en la capa activa
    dentro de esa region.
    """
    data = request.get_json()
    trazo_b64 = data["trazo_b64"]
    x = int(data["x"])
    y = int(data["y"])
    tolerancia = int(data.get("tolerancia", 8))

    img_bytes = base64.b64decode(trazo_b64)
    img_bw = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))
    h, w = img_bw.shape

    if x < 0 or x >= w or y < 0 or y >= h:
        return jsonify({"error": "coordenadas fuera de imagen"}), 400

    # Construir mascara de trazo binaria dura (sin anti-alias) para flood fill fiable
    _, trazo_bin = cv2.threshold(img_bw, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilatar el trazo para cerrar huecos de anti-alias
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerancia // 2 + 2, tolerancia // 2 + 2))
    trazo_cerrado = cv2.dilate(trazo_bin, k, iterations=1)

    if int(trazo_cerrado[y, x]) == 255:
        return jsonify({"error": "click sobre trazo"}), 400

    # Flood fill para obtener la region
    fill_input = trazo_cerrado.copy()
    fill_mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(fill_input, fill_mask, (x, y), 128)
    region = ((fill_input == 128).astype(np.uint8) * 255)

    # Erosionar levemente para no comerse el borde del trazo
    region = cv2.erode(region, k, iterations=1)

    # Devolver la mascara como PNG RGBA: blanco opaco donde se debe borrar
    mascara_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mascara_rgba[region == 255] = [0, 0, 0, 255]

    buf = io.BytesIO()
    Image.fromarray(mascara_rgba, "RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"mascara_b64": b64})


@app.route("/exportar", methods=["POST"])
def exportar():
    data = request.get_json()
    trazo_b64 = data["trazo_b64"]
    capas = data["capas"]   # lista de {b64, blending, opacidad}
    ancho = int(data["ancho"])
    alto = int(data["alto"])

    # Fondo blanco
    resultado = Image.new("RGBA", (ancho, alto), (255, 255, 255, 255))

    for capa in capas:
        if not capa.get("visible", True):
            continue
        capa_bytes = base64.b64decode(capa["b64"])
        capa_img = Image.open(io.BytesIO(capa_bytes)).convert("RGBA").resize((ancho, alto))

        opacidad = capa.get("opacidad", 100) / 100
        if opacidad < 1.0:
            r, g, b, a = capa_img.split()
            a = a.point(lambda p: int(p * opacidad))
            capa_img = Image.merge("RGBA", (r, g, b, a))

        blending = capa.get("blending", "normal")
        if blending == "multiplicar":
            resultado = Image.composite(
                ImageChops_blend(resultado, capa_img, "multiply"),
                resultado,
                capa_img.split()[3],
            )
        elif blending == "screen":
            resultado = Image.composite(
                ImageChops_blend(resultado, capa_img, "screen"),
                resultado,
                capa_img.split()[3],
            )
        elif blending == "overlay":
            resultado = Image.composite(
                ImageChops_blend(resultado, capa_img, "overlay"),
                resultado,
                capa_img.split()[3],
            )
        else:
            resultado = Image.alpha_composite(resultado, capa_img)

    # Trazo encima de todo
    trazo_bytes = base64.b64decode(trazo_b64)
    trazo_pil = Image.open(io.BytesIO(trazo_bytes)).convert("L").resize((ancho, alto))
    trazo_rgba = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    trazo_rgba.paste(Image.new("RGB", (ancho, alto), (0, 0, 0)), mask=Image.eval(trazo_pil, lambda p: 255 - p))
    resultado = Image.alpha_composite(resultado, trazo_rgba)

    buf = io.BytesIO()
    resultado.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"imagen_b64": b64})


def ImageChops_blend(base: Image.Image, capa: Image.Image, modo: str) -> Image.Image:
    from PIL import ImageChops
    import numpy as np
    base_rgb = np.array(base.convert("RGB")).astype(np.float32) / 255
    capa_rgb = np.array(capa.convert("RGB")).astype(np.float32) / 255
    if modo == "multiply":
        out = base_rgb * capa_rgb
    elif modo == "screen":
        out = 1 - (1 - base_rgb) * (1 - capa_rgb)
    elif modo == "overlay":
        out = np.where(
            base_rgb < 0.5,
            2 * base_rgb * capa_rgb,
            1 - 2 * (1 - base_rgb) * (1 - capa_rgb),
        )
    else:
        out = capa_rgb
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    resultado = Image.fromarray(out, "RGB").convert("RGBA")
    resultado.putalpha(capa.split()[3])
    return resultado


if __name__ == "__main__":
    app.run(debug=True, port=5050)
