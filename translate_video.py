#!/usr/bin/env python3
"""Transcribe, traduce y subtitula un video de forma local (offline).

Flujo de procesamiento:
    1. Extracción del audio con ffmpeg (WAV mono 16 kHz).
    2. Transcripción con faster-whisper (modelo "base", CPU, int8) con marcas de tiempo.
    3. Traducción de cada segmento con argos-translate (offline).
    4. Generación de los archivos .srt original y traducido.
    5. Generación de un video con los subtítulos traducidos incrustados,
       conservando el audio original.

Uso:
    python translate_video.py VIDEO [--src en] [--dst es]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
TOTAL_STAGES = 5


class TranslateVideoError(Exception):
    """Error controlado del proceso; se muestra al usuario sin traza completa."""


@dataclass
class Segment:
    """Segmento de texto con marcas de tiempo en segundos."""

    start: float
    end: float
    text: str


# --------------------------------------------------------------------------- #
# Utilidades de progreso
# --------------------------------------------------------------------------- #
def stage(number: int, message: str) -> None:
    print(f"[{number}/{TOTAL_STAGES}] {message}", flush=True)


def info(message: str) -> None:
    print(f"      {message}", flush=True)


# --------------------------------------------------------------------------- #
# Generación de SRT
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT ``HH:MM:SS,mmm``."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(segments: Iterable[Segment]) -> str:
    """Construye el contenido SRT; omite segmentos vacíos y numera desde 1."""
    blocks = []
    index = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        end = max(seg.end, seg.start)
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(seg.start)} --> {format_timestamp(end)}\n"
            f"{text}\n"
        )
        index += 1
    return "\n".join(blocks)


def write_srt(segments: Iterable[Segment], path: Path) -> Path:
    path = Path(path)
    path.write_text(build_srt(segments), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranslateVideoError(
            "No se encontró 'ffmpeg' en el PATH. Instálelo "
            "(p. ej. 'sudo apt install ffmpeg') y vuelva a intentarlo."
        )


def run_ffmpeg(args: List[str], description: str, cwd: Optional[Path] = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    except OSError as exc:
        raise TranslateVideoError(f"No se pudo ejecutar ffmpeg ({description}): {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "sin detalle"
        raise TranslateVideoError(f"ffmpeg falló durante {description}:\n{detail}")


def extract_audio(video: Path, audio_out: Path) -> Path:
    """Extrae el audio como WAV PCM mono 16 kHz (formato óptimo para Whisper)."""
    run_ffmpeg(
        ["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_out)],
        "la extracción de audio",
    )
    if not audio_out.exists() or audio_out.stat().st_size == 0:
        raise TranslateVideoError("El video no contiene una pista de audio utilizable.")
    return audio_out


def burn_subtitles(video: Path, srt: Path, output: Path) -> Path:
    """Incrusta los subtítulos en el video y copia el audio original sin recodificar.

    El filtro ``subtitles`` de ffmpeg requiere escapar rutas con caracteres
    especiales; para evitarlo se copia el SRT a un directorio temporal con un
    nombre simple y se ejecuta ffmpeg desde ese directorio.
    """
    with tempfile.TemporaryDirectory(prefix="subs_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copyfile(srt, tmp_dir / "subs.srt")
        run_ffmpeg(
            [
                "-i", str(video.resolve()),
                "-vf", "subtitles=subs.srt:charenc=UTF-8",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "copy",
                str(output.resolve()),
            ],
            "la incrustación de subtítulos",
            cwd=tmp_dir,
        )
    return output


# --------------------------------------------------------------------------- #
# Transcripción (faster-whisper)
# --------------------------------------------------------------------------- #
def transcribe(audio: Path, language: str, model_size: str = WHISPER_MODEL) -> List[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranslateVideoError(
            "No está instalado 'faster-whisper'. Ejecute: pip install -r requirements.txt"
        ) from exc

    try:
        model = WhisperModel(model_size, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    except Exception as exc:  # descarga del modelo, red, disco, etc.
        raise TranslateVideoError(
            f"No se pudo cargar el modelo Whisper '{model_size}': {exc}"
        ) from exc

    try:
        raw_segments, meta = model.transcribe(str(audio), language=language, vad_filter=True)
        segments = []
        for seg in raw_segments:  # generador: la transcripción ocurre aquí
            segments.append(Segment(float(seg.start), float(seg.end), seg.text.strip()))
            info(f"{format_timestamp(seg.start)} {seg.text.strip()[:60]}")
    except Exception as exc:
        raise TranslateVideoError(f"Error durante la transcripción: {exc}") from exc

    duration = getattr(meta, "duration", None)
    if duration:
        info(f"Duración de audio procesada: {format_timestamp(duration)}")
    return segments


# --------------------------------------------------------------------------- #
# Traducción (argos-translate)
# --------------------------------------------------------------------------- #
def ensure_argos_package(src: str, dst: str) -> None:
    """Instala el paquete de idioma src->dst si no está disponible localmente.

    La descarga solo se requiere la primera vez; después funciona offline.
    """
    try:
        import argostranslate.package as argos_package
        import argostranslate.translate as argos_translate
    except ImportError as exc:
        raise TranslateVideoError(
            "No está instalado 'argostranslate'. Ejecute: pip install -r requirements.txt"
        ) from exc

    installed = {lang.code: lang for lang in argos_translate.get_installed_languages()}
    if src in installed and dst in installed and installed[src].get_translation(installed[dst]):
        return

    info(f"Paquete de traducción {src}->{dst} no instalado; descargando (una sola vez)...")
    try:
        argos_package.update_package_index()
        available = argos_package.get_available_packages()
        pkg = next((p for p in available if p.from_code == src and p.to_code == dst), None)
        if pkg is None:
            raise TranslateVideoError(
                f"argos-translate no ofrece un paquete directo {src}->{dst}."
            )
        argos_package.install_from_path(pkg.download())
    except TranslateVideoError:
        raise
    except Exception as exc:
        raise TranslateVideoError(
            f"No se pudo descargar el paquete de idioma {src}->{dst}: {exc}"
        ) from exc


def argos_translator(src: str, dst: str) -> Callable[[str], str]:
    ensure_argos_package(src, dst)
    import argostranslate.translate as argos_translate

    return lambda text: argos_translate.translate(text, src, dst)


def translate_segments(
    segments: Iterable[Segment], translator: Callable[[str], str]
) -> List[Segment]:
    """Traduce el texto de cada segmento conservando sus marcas de tiempo."""
    segments = list(segments)
    result = []
    for i, seg in enumerate(segments, start=1):
        try:
            text = translator(seg.text) if seg.text.strip() else seg.text
        except Exception as exc:
            raise TranslateVideoError(f"Error al traducir el segmento {i}: {exc}") from exc
        result.append(Segment(seg.start, seg.end, text.strip()))
        info(f"Segmento {i}/{len(segments)} traducido")
    return result


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def output_paths(video: Path, src: str, dst: str, output_dir: Optional[Path] = None):
    base_dir = output_dir or video.parent
    stem = video.stem
    return (
        base_dir / f"{stem}.{src}.srt",
        base_dir / f"{stem}.{dst}.srt",
        base_dir / f"{stem}.{dst}.subtitulado.mp4",
    )


def run(video: Path, src: str, dst: str, output_dir: Optional[Path] = None) -> dict:
    video = Path(video)
    if not video.is_file():
        raise TranslateVideoError(f"No existe el archivo de video: {video}")
    if src == dst:
        raise TranslateVideoError("El idioma de origen y destino deben ser distintos.")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    check_ffmpeg()

    srt_src, srt_dst, video_out = output_paths(video, src, dst, output_dir)

    with tempfile.TemporaryDirectory(prefix="translate_video_") as tmp:
        t0 = time.monotonic()

        stage(1, f"Extrayendo audio de '{video.name}'...")
        audio = extract_audio(video, Path(tmp) / "audio.wav")

        stage(2, f"Transcribiendo ({WHISPER_MODEL}, {WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE}, idioma '{src}')...")
        segments = transcribe(audio, src)
        if not segments:
            raise TranslateVideoError("La transcripción no produjo texto (¿audio sin voz?).")
        info(f"{len(segments)} segmentos transcritos")

        stage(3, f"Traduciendo {src} -> {dst} con argos-translate...")
        translated = translate_segments(segments, argos_translator(src, dst))

        stage(4, "Generando archivos SRT...")
        write_srt(segments, srt_src)
        write_srt(translated, srt_dst)
        info(f"Original:  {srt_src}")
        info(f"Traducido: {srt_dst}")

        stage(5, "Incrustando subtítulos traducidos en el video...")
        burn_subtitles(video, srt_dst, video_out)
        info(f"Video:     {video_out}")

        print(f"Proceso completado en {time.monotonic() - t0:.1f} s.", flush=True)

    return {"srt_original": srt_src, "srt_translated": srt_dst, "video": video_out}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe, traduce y subtitula un video de forma local."
    )
    parser.add_argument("video", type=Path, help="Ruta del video de entrada")
    parser.add_argument("src", nargs="?", default="en", help="Idioma de origen (por defecto: en)")
    parser.add_argument("dst", nargs="?", default="es", help="Idioma de destino (por defecto: es)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Directorio de salida (por defecto: el del video)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.video, args.src, args.dst, args.output_dir)
    except TranslateVideoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nProceso cancelado por el usuario.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
