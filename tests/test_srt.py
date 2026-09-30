"""Pruebas de generación de SRT.

La transcripción (faster-whisper) y la traducción (argos-translate) se simulan
con mocks: las pruebas no descargan modelos ni requieren ffmpeg.
"""

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translate_video as tv  # noqa: E402

SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")

FAKE_TRANSCRIPTION = [
    (0.0, 2.5, " Hello everyone."),
    (2.5, 5.123, " Welcome to the network operations briefing."),
    (65.4, 70.0, " BGP sessions are stable."),
    (3725.999, 3730.0, " Thank you."),
]

FAKE_DICTIONARY = {
    "Hello everyone.": "Hola a todos.",
    "Welcome to the network operations briefing.": "Bienvenidos a la reunión de operaciones de red.",
    "BGP sessions are stable.": "Las sesiones BGP están estables.",
    "Thank you.": "Gracias.",
}


def fake_translator(text):
    return FAKE_DICTIONARY[text]


@pytest.fixture
def fake_whisper(monkeypatch):
    """Inyecta un módulo 'faster_whisper' simulado en sys.modules."""
    calls = {}

    class FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            calls["init"] = (size, device, compute_type)

        def transcribe(self, audio, language=None, **kwargs):
            calls["transcribe"] = (audio, language)
            segs = (SimpleNamespace(start=s, end=e, text=t) for s, e, t in FAKE_TRANSCRIPTION)
            return segs, SimpleNamespace(duration=3730.0, language=language)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


def parse_srt(content):
    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    return [b.split("\n") for b in blocks]


# --------------------------------------------------------------------------- #
# Formato de tiempos
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00,000"),
        (0.001, "00:00:00,001"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),
        (60, "00:01:00,000"),
        (65.4, "00:01:05,400"),
        (3599.9994, "00:59:59,999"),
        (3599.9996, "01:00:00,000"),  # el redondeo propaga el acarreo
        (3725.999, "01:02:05,999"),
        (36000, "10:00:00,000"),
        (-3, "00:00:00,000"),  # tiempos negativos se acotan a cero
    ],
)
def test_format_timestamp(seconds, expected):
    assert tv.format_timestamp(seconds) == expected


# --------------------------------------------------------------------------- #
# Construcción del SRT
# --------------------------------------------------------------------------- #
def test_build_srt_numbering_and_format():
    segments = [tv.Segment(s, e, t) for s, e, t in FAKE_TRANSCRIPTION]
    blocks = parse_srt(tv.build_srt(segments))

    assert [b[0] for b in blocks] == ["1", "2", "3", "4"]
    for block in blocks:
        assert len(block) == 3
        assert SRT_TIME.match(block[1]), block[1]
    assert blocks[0][1] == "00:00:00,000 --> 00:00:02,500"
    assert blocks[1][1] == "00:00:02,500 --> 00:00:05,123"
    assert blocks[3][1] == "01:02:05,999 --> 01:02:10,000"
    assert blocks[0][2] == "Hello everyone."  # texto sin espacios sobrantes


def test_build_srt_skips_empty_segments_and_keeps_consecutive_numbering():
    segments = [
        tv.Segment(0, 1, "Uno"),
        tv.Segment(1, 2, "   "),
        tv.Segment(2, 3, ""),
        tv.Segment(3, 4, "Dos"),
    ]
    blocks = parse_srt(tv.build_srt(segments))
    assert [b[0] for b in blocks] == ["1", "2"]
    assert [b[2] for b in blocks] == ["Uno", "Dos"]


def test_build_srt_end_never_before_start():
    blocks = parse_srt(tv.build_srt([tv.Segment(5.0, 4.0, "x")]))
    assert blocks[0][1] == "00:00:05,000 --> 00:00:05,000"


def test_build_srt_empty_input():
    assert tv.build_srt([]) == ""


def test_write_srt_utf8(tmp_path):
    path = tv.write_srt([tv.Segment(0, 1, "Configuración ñandú")], tmp_path / "a.srt")
    assert path.read_text(encoding="utf-8").splitlines()[2] == "Configuración ñandú"


# --------------------------------------------------------------------------- #
# Transcripción y traducción simuladas
# --------------------------------------------------------------------------- #
def test_transcribe_uses_base_cpu_int8(fake_whisper, tmp_path):
    segments = tv.transcribe(tmp_path / "audio.wav", "en")
    assert fake_whisper["init"] == ("base", "cpu", "int8")
    assert fake_whisper["transcribe"][1] == "en"
    assert [s.text for s in segments] == [t.strip() for _, _, t in FAKE_TRANSCRIPTION]


def test_transcribe_without_library_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # fuerza ImportError
    with pytest.raises(tv.TranslateVideoError, match="faster-whisper"):
        tv.transcribe(tmp_path / "audio.wav", "en")


def test_translate_segments_keeps_timestamps():
    original = [tv.Segment(s, e, t.strip()) for s, e, t in FAKE_TRANSCRIPTION]
    translated = tv.translate_segments(original, fake_translator)
    assert [(s.start, s.end) for s in translated] == [(s.start, s.end) for s in original]
    assert translated[0].text == "Hola a todos."


def test_translate_segments_wraps_errors():
    def broken(_):
        raise RuntimeError("modelo no disponible")

    with pytest.raises(tv.TranslateVideoError, match="segmento 1"):
        tv.translate_segments([tv.Segment(0, 1, "Hi")], broken)


def test_original_and_translated_srt_are_aligned(fake_whisper, tmp_path):
    segments = tv.transcribe(tmp_path / "audio.wav", "en")
    translated = tv.translate_segments(segments, fake_translator)

    src = parse_srt(tv.write_srt(segments, tmp_path / "v.en.srt").read_text(encoding="utf-8"))
    dst = parse_srt(tv.write_srt(translated, tmp_path / "v.es.srt").read_text(encoding="utf-8"))

    assert len(src) == len(dst) == 4
    for a, b in zip(src, dst):
        assert a[0] == b[0]  # misma numeración
        assert a[1] == b[1]  # mismos tiempos
    assert dst[2][2] == "Las sesiones BGP están estables."


# --------------------------------------------------------------------------- #
# Orquestación y CLI (sin ffmpeg ni modelos reales)
# --------------------------------------------------------------------------- #
def test_run_pipeline_with_mocks(fake_whisper, monkeypatch, tmp_path, capsys):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(tv, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(tv, "extract_audio", lambda v, out: out)
    monkeypatch.setattr(tv, "argos_translator", lambda s, d: fake_translator)
    burned = {}

    def fake_burn(v, srt, out):
        burned["srt"] = srt
        out.write_bytes(b"video")
        return out

    monkeypatch.setattr(tv, "burn_subtitles", fake_burn)

    result = tv.run(video, "en", "es")

    assert result["srt_original"] == tmp_path / "demo.en.srt"
    assert result["srt_translated"] == tmp_path / "demo.es.srt"
    assert result["video"].exists()
    assert burned["srt"] == result["srt_translated"]
    assert "Hola a todos." in result["srt_translated"].read_text(encoding="utf-8")
    out = capsys.readouterr().out
    for n in range(1, 6):
        assert f"[{n}/5]" in out


def test_parse_args_defaults():
    args = tv.parse_args(["video.mp4"])
    assert (args.src, args.dst) == ("en", "es")
    args = tv.parse_args(["video.mp4", "fr", "en"])
    assert (args.src, args.dst) == ("fr", "en")


def test_main_missing_video_returns_error(tmp_path, capsys):
    assert tv.main([str(tmp_path / "no_existe.mp4")]) == 1
    assert "No existe el archivo" in capsys.readouterr().err


def test_main_same_languages_returns_error(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert tv.main([str(video), "es", "es"]) == 1
    assert "distintos" in capsys.readouterr().err
