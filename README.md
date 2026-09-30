# translate_video.py — Transcripción, traducción y subtitulado de video (offline)

Script en Python 3 que procesa un video de forma local y sin servicios en la nube:

1. Extrae el audio con **ffmpeg** (WAV mono, 16 kHz).
2. Transcribe el audio con **faster-whisper** (modelo `base`, CPU, `int8`) con marcas de tiempo.
3. Traduce cada segmento con **argos-translate** (offline).
4. Genera dos subtítulos `.srt`: original y traducido.
5. Genera un video con los subtítulos traducidos incrustados y el audio original.

## 1. Requisitos

| Componente | Versión mínima | Observación |
|---|---|---|
| Python | 3.8 | Se recomienda 3.10 o superior |
| ffmpeg | 4.x | Compilado con `libx264` y `libass` (filtro `subtitles`) |
| faster-whisper | 1.0.0 | Ver `requirements.txt` |
| argostranslate | 1.9.0 | Ver `requirements.txt` |

## 2. Instalación

### 2.1 ffmpeg

```bash
# Debian / Ubuntu
sudo apt install ffmpeg
# macOS (Homebrew)
brew install ffmpeg
# Windows (winget)
winget install Gyan.FFmpeg
```

### 2.2 Dependencias de Python

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 Modelos (descarga única)

Los modelos se descargan automáticamente en la **primera ejecución** y quedan
en caché local; a partir de ese momento el proceso funciona sin conexión:

- Whisper `base`: `~/.cache/huggingface/` (≈ 145 MB).
- Paquete de idioma argos (p. ej. `en→es`): `~/.local/share/argos-translate/` (≈ 100 MB).

Para dejar el equipo preparado antes de trabajar sin red, puede instalar el
paquete de idioma manualmente:

```bash
argospm update
argospm install translate-en_es
```

## 3. Uso

```bash
python translate_video.py VIDEO [IDIOMA_ORIGEN] [IDIOMA_DESTINO] [-o DIRECTORIO_SALIDA]
```

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `VIDEO` | — | Ruta del video de entrada |
| `IDIOMA_ORIGEN` | `en` | Código ISO 639-1 del idioma hablado |
| `IDIOMA_DESTINO` | `es` | Código ISO 639-1 del idioma de los subtítulos |
| `-o, --output-dir` | directorio del video | Carpeta donde se escriben los resultados |

### 3.1 Ejemplo

```bash
python translate_video.py charla.mp4             # inglés -> español
python translate_video.py charla.mp4 en pt -o salida/
```

Salida por consola:

```
[1/5] Extrayendo audio de 'charla.mp4'...
[2/5] Transcribiendo (base, cpu, int8, idioma 'en')...
      00:00:00,000 Hello everyone.
      ...
      42 segmentos transcritos
[3/5] Traduciendo en -> es con argos-translate...
[4/5] Generando archivos SRT...
      Original:  charla.en.srt
      Traducido: charla.es.srt
[5/5] Incrustando subtítulos traducidos en el video...
      Video:     charla.es.subtitulado.mp4
Proceso completado en 183.4 s.
```

### 3.2 Archivos generados

| Archivo | Contenido |
|---|---|
| `<nombre>.<origen>.srt` | Subtítulos en el idioma original |
| `<nombre>.<destino>.srt` | Subtítulos traducidos (mismos tiempos y numeración) |
| `<nombre>.<destino>.subtitulado.mp4` | Video H.264 con subtítulos incrustados y audio original copiado sin recodificar |

### 3.3 Manejo de errores

Ante un error controlado el script muestra un mensaje `ERROR: ...` en `stderr`
y finaliza con código `1` (código `130` si se cancela con Ctrl+C). Se validan,
entre otros: existencia del video, disponibilidad de ffmpeg, ausencia de pista
de audio, idiomas iguales, fallos al cargar o descargar modelos, paquete de
idioma inexistente y errores de ffmpeg (se incluye su detalle).

> **Nota:** argos-translate requiere un paquete directo entre ambos idiomas.
> Si no existe, el script lo indica explícitamente.

## 4. Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

Las pruebas (`tests/test_srt.py`) verifican el formato de tiempos
`HH:MM:SS,mmm`, la numeración consecutiva de los bloques SRT, la alineación
entre el SRT original y el traducido, el uso de la configuración
`base / cpu / int8` y el manejo de errores.

**La transcripción y la traducción se simulan con mocks**, por lo que las
pruebas no descargan modelos, no requieren `faster-whisper` ni `argostranslate`
instalados y no invocan ffmpeg. En el entorno de desarrollo en la nube donde se
construyó este script no fue posible descargar los modelos; la **ejecución real
de extremo a extremo se realiza en el equipo local** del usuario siguiendo la
sección 2. Las etapas de ffmpeg (extracción de audio e incrustación de
subtítulos) sí se validaron con un video sintético.

## 5. Consideraciones de rendimiento

- El modelo `base` en CPU con `int8` procesa aproximadamente entre 0,2x y 0,5x
  la duración del audio en un equipo moderno, según núcleos disponibles.
- La incrustación de subtítulos recodifica el video (`libx264`, `crf 20`); es la
  etapa más costosa en videos largos o de alta resolución.
