"""
sketch-vectorizer
Limpia un boceto escaneado y genera PNG limpio + SVG vectorial.
Uso: python vectorizar.py <imagen> [opciones]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import vtracer


def limpiar_boceto(img_bgr: np.ndarray, args) -> np.ndarray:
    """
    Transforma una imagen de boceto sucio a trazos limpios en blanco y negro.
    Retorna imagen binaria (0=negro trazo, 255=fondo blanco).
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Escalar si la imagen es muy pequeña (mejora calidad de trazo)
    h, w = gris.shape
    if max(h, w) < 1000:
        factor = 1000 / max(h, w)
        gris = cv2.resize(gris, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)

    # Reducir ruido manteniendo bordes
    denoised = cv2.fastNlMeansDenoising(gris, h=args.denoise_strength, templateWindowSize=7, searchWindowSize=21)

    # Umbralizado adaptativo — maneja variaciones de iluminación del escáner
    umbral = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        args.block_size,   # tamaño del bloque vecino (debe ser impar)
        args.c_value,      # constante que se resta: más alto = elimina más gris
    )

    # Invertir para trabajar con trazos blancos sobre negro (morfología)
    trazos = cv2.bitwise_not(umbral)

    # Limpiar manchas pequeñas (ruido puntual)
    kernel_limpieza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.noise_size, args.noise_size))
    trazos = cv2.morphologyEx(trazos, cv2.MORPH_OPEN, kernel_limpieza)

    # Cerrar huecos pequeños dentro de los trazos
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    trazos = cv2.morphologyEx(trazos, cv2.MORPH_CLOSE, kernel_cierre)

    # Adelgazar trazos solo si el usuario lo pide (skeleton)
    if args.thin:
        trazos = cv2.ximgproc.thinning(trazos) if hasattr(cv2, 'ximgproc') else trazos

    # Volver a imagen final: fondo blanco, trazos negros
    resultado = cv2.bitwise_not(trazos)
    return resultado


def guardar_png(img_bw: np.ndarray, ruta: Path) -> None:
    Image.fromarray(img_bw).save(str(ruta), "PNG", optimize=True)
    print(f"  PNG limpio  -> {ruta}")


def guardar_svg(img_bw: np.ndarray, ruta: Path, args) -> None:
    """Convierte el PNG limpio a SVG usando vtracer."""
    # vtracer necesita un PNG temporal en disco
    tmp = ruta.with_suffix(".tmp.png")
    Image.fromarray(img_bw).save(str(tmp))

    vtracer.convert_image_to_svg_py(
        str(tmp),
        str(ruta),
        colormode="binary",
        hierarchical="stacked",
        mode=args.svg_mode,          # "spline" = curvas suaves, "polygon" = poligonal
        filter_speckle=args.speckle, # elimina manchas menores a N px
        color_precision=6,
        layer_difference=16,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=8,
    )
    tmp.unlink()
    print(f"  SVG vectorial -> {ruta}")


def procesar(ruta_entrada: str, args) -> None:
    entrada = Path(ruta_entrada)
    if not entrada.exists():
        print(f"Error: no se encontró el archivo '{entrada}'")
        sys.exit(1)

    # Directorio de salida
    if args.output_dir:
        salida_dir = Path(args.output_dir)
        salida_dir.mkdir(parents=True, exist_ok=True)
    else:
        salida_dir = entrada.parent

    nombre_base = entrada.stem + "_limpio"

    print(f"\nProcesando: {entrada.name}")

    img = cv2.imread(str(entrada))
    if img is None:
        print(f"Error: no se pudo leer la imagen (formato no soportado o archivo corrupto)")
        sys.exit(1)

    print(f"  Resolución original: {img.shape[1]}×{img.shape[0]}px")

    limpio = limpiar_boceto(img, args)

    guardar_png(limpio, salida_dir / (nombre_base + ".png"))
    guardar_svg(limpio, salida_dir / (nombre_base + ".svg"), args)

    print("Listo.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Limpia bocetos escaneados y genera PNG + SVG vectorial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python vectorizar.py boceto.jpg
  python vectorizar.py boceto.png --output-dir resultados/
  python vectorizar.py boceto.jpg --denoise 15 --c-value 10
  python vectorizar.py boceto.jpg --svg-mode polygon
        """
    )
    parser.add_argument("imagen", help="Ruta a la imagen de entrada (jpg, png, bmp, tiff, etc.)")
    parser.add_argument("-o", "--output-dir", metavar="DIR", help="Carpeta de salida (por defecto: misma carpeta que la imagen)")

    grupo_limpieza = parser.add_argument_group("Parámetros de limpieza")
    grupo_limpieza.add_argument("--denoise", dest="denoise_strength", type=int, default=10,
        metavar="N", help="Fuerza de reducción de ruido 1-30 (default: 10)")
    grupo_limpieza.add_argument("--block-size", type=int, default=15,
        metavar="N", help="Tamaño de bloque para umbralizado adaptativo, impar (default: 15)")
    grupo_limpieza.add_argument("--c-value", type=int, default=8,
        metavar="N", help="Constante de umbral, más alto elimina más grises (default: 8)")
    grupo_limpieza.add_argument("--noise-size", type=int, default=2,
        metavar="N", help="Tamaño mínimo de manchas a eliminar en px (default: 2)")
    grupo_limpieza.add_argument("--thin", action="store_true",
        help="Adelgazar trazos a 1px de grosor (skeleton, requiere opencv-contrib)")

    grupo_svg = parser.add_argument_group("Parámetros de vectorización")
    grupo_svg.add_argument("--svg-mode", choices=["spline", "polygon", "none"], default="spline",
        help="Modo de vectorización: spline=curvas suaves, polygon=poligonal (default: spline)")
    grupo_svg.add_argument("--speckle", type=int, default=4,
        metavar="N", help="Elimina manchas SVG menores a N px (default: 4)")

    args = parser.parse_args()

    # block_size debe ser impar y >= 3
    if args.block_size % 2 == 0:
        args.block_size += 1

    procesar(args.imagen, args)


EXTENSIONS_SOPORTADAS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def procesar_carpeta(carpeta: str) -> None:
    entrada = Path(carpeta)
    if not entrada.is_dir():
        print(f"Error: '{entrada}' no es una carpeta valida.")
        sys.exit(1)

    salida = entrada / "procesados_vectorizar"
    salida.mkdir(exist_ok=True)

    imagenes = [f for f in entrada.iterdir() if f.suffix.lower() in EXTENSIONS_SOPORTADAS]
    if not imagenes:
        print(f"No se encontraron imagenes en '{entrada}'.")
        return

    print(f"Carpeta de entrada : {entrada}")
    print(f"Carpeta de salida  : {salida}")
    print(f"Imagenes encontradas: {len(imagenes)}\n")

    # Parametros por defecto
    class ArgDefaults:
        denoise_strength = 10
        block_size = 15
        c_value = 8
        noise_size = 2
        thin = False
        svg_mode = "spline"
        speckle = 4
        output_dir = str(salida)

    args = ArgDefaults()

    for img_path in sorted(imagenes):
        procesar(str(img_path), args)


if __name__ == "__main__":
    # --- CONFIGURACION ---
    # Cambia esta ruta a la carpeta donde tienes tus bocetos.
    # El script procesara todas las imagenes que encuentre y guardara
    # los resultados en una subcarpeta llamada "procesados_vectorizar".
    INPUT_DIR = r"C:\Users\mcorteze\Pictures\Screenshot HD"
    # ---------------------
    if len(sys.argv) > 1:
        # Permite seguir usando la linea de comandos normalmente
        main()
    else:
        procesar_carpeta(INPUT_DIR)
