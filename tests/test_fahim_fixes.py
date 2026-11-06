"""Regression tests for the fahim-fixes branch: dubbed captions, rolling captions,
rate-limit retries, Gemini retries and opt-in fallback, spoken visual cues, batch mode."""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

import config
import download
import gemini
import transcribe
import watch
import watch_batch


def _tracks(*keys):
    return {key: [{'ext': 'vtt'}] for key in keys}


# --- Caption language on auto-dubbed videos -------------------------------------------------

DUBBED = ('ar-orig', 'bn-orig', 'de-DE-orig', 'en-orig', 'es-US-orig', 'ja-orig')


def test_dubbed_video_uses_the_original_audio_language_not_alphabetical_first():
    info = {'extractor_key': 'Youtube', 'language': 'en-US',
            'automatic_captions': _tracks(*DUBBED, 'ar', 'en', 'fr')}
    track = download.select_caption(info)
    assert track['key'] == 'en-orig'
    assert track['provenance'] == 'original'


def test_audio_track_marked_original_beats_the_reported_language():
    info = {'extractor_key': 'Youtube', 'language': 'en',
            'automatic_captions': _tracks(*DUBBED),
            'formats': [{'acodec': 'opus', 'language': 'ar', 'format_note': 'Arabic - dubbed-auto'},
                        {'acodec': 'opus', 'language': 'ja', 'format_note': 'Japanese original (default)'}]}
    assert download.select_caption(info)['key'] == 'ja-orig'


def test_ambiguous_originals_without_evidence_prefer_english_over_alphabetical():
    info = {'extractor_key': 'Youtube', 'automatic_captions': _tracks('ar-orig', 'en-orig', 'ko-orig')}
    assert download.select_caption(info)['key'] == 'en-orig'


def test_single_original_behaviour_is_unchanged():
    info = {'extractor_key': 'Youtube', 'language': 'en', 'automatic_captions': _tracks('ko-orig', 'en', 'ko')}
    assert download.select_caption(info)['key'] == 'ko-orig'


# --- YouTube rolling auto-captions ----------------------------------------------------------

# Real YouTube files hold a single-space line inside cues; "~" marks it so editors can't strip it.
ROLLING = """WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.270 align:start position:0%
~
Please<00:00:00.400><c> welcome</c><00:00:00.720><c> everyone</c>

00:00:02.270 --> 00:00:02.280 align:start position:0%
Please welcome everyone
~

00:00:02.280 --> 00:00:04.110 align:start position:0%
Please welcome everyone
to<00:00:02.500><c> the</c><00:00:02.800><c> talk</c>

00:00:04.110 --> 00:00:04.120 align:start position:0%
to the talk
~

00:00:04.120 --> 00:00:06.000 align:start position:0%
to the talk
today<00:00:04.500><c> okay</c>
""".replace('\n~\n', '\n \n')


def test_rolling_auto_captions_keep_each_line_once(tmp_path):
    path = tmp_path / 'auto.vtt'
    path.write_bytes(ROLLING.encode('utf-8'))
    out = transcribe.parse_vtt(str(path))
    assert [s['text'] for s in out] == ['Please welcome everyone', 'to the talk', 'today okay']
    assert [s['start'] for s in out] == [0.16, 2.28, 4.12]


def test_plain_captions_with_repeated_speech_are_untouched(tmp_path):
    path = tmp_path / 'manual.vtt'
    path.write_bytes(b'WEBVTT\n\n00:01.000 --> 00:02.000\nNo.\n\n00:03.000 --> 00:04.000\nNo.\n')
    assert [s['text'] for s in transcribe.parse_vtt(str(path))] == ['No.', 'No.']


# --- yt-dlp rate limits ---------------------------------------------------------------------

def _proc(code, stderr=''):
    return subprocess.CompletedProcess([], code, '', stderr)


def test_caption_download_retries_after_429(monkeypatch):
    replies = [_proc(1, 'ERROR: Unable to download video subtitles: HTTP Error 429: Too Many Requests'), _proc(0)]
    waits = []
    monkeypatch.setattr(download, 'run_text', lambda cmd: replies.pop(0))
    monkeypatch.setattr(download, '_sleep', waits.append)
    assert download._run_retrying(['yt-dlp']).returncode == 0
    assert waits == [download.RATE_LIMIT_WAITS[0]]


def test_non_rate_limit_errors_are_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(download, 'run_text', lambda cmd: calls.append(cmd) or _proc(1, 'ERROR: HTTP Error 403: Forbidden'))
    monkeypatch.setattr(download, '_sleep', lambda s: pytest.fail('must not wait on a 403'))
    with pytest.raises(SystemExit, match='403'):
        download._run_retrying(['yt-dlp'])
    assert len(calls) == 1


def test_rate_limit_gives_up_after_the_retries(monkeypatch):
    monkeypatch.setattr(download, 'run_text', lambda cmd: _proc(1, 'HTTP Error 429: Too Many Requests'))
    waits = []
    monkeypatch.setattr(download, '_sleep', waits.append)
    with pytest.raises(SystemExit, match='rate limiting'):
        download._run_retrying(['yt-dlp'])
    assert waits == list(download.RATE_LIMIT_WAITS)


# --- Gemini retries ------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, body):
        self.status, self.headers, self._body = 200, {}, body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def _http(code, message):
    body = ('{"error": {"message": "%s"}}' % message).encode()
    return HTTPError('https://example.invalid', code, 'error', {}, io.BytesIO(body))


ANSWER = (b'{"status": "completed", "steps": [{"type": "model_output", '
          b'"content": [{"type": "text", "text": "ok"}]}]}')


def _replies(monkeypatch, *replies):
    queue = list(replies)
    def fake(request, timeout=None):
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(gemini, 'urlopen', fake)
    return queue


def test_gemini_retries_an_overload_then_answers(monkeypatch):
    _replies(monkeypatch, _http(503, 'high demand'), FakeResponse(ANSWER))
    waits = []
    result = gemini.ask({'uri': 'u'}, None, model='m', key='k', retries=2, sleep=waits.append)
    assert result['text'] == 'ok' and waits == [gemini.RETRY_WAITS[0]]


def test_gemini_honours_the_suggested_retry_delay(monkeypatch):
    _replies(monkeypatch, _http(429, 'Please retry in 38s or upgrade'), FakeResponse(ANSWER))
    waits = []
    gemini.ask({'uri': 'u'}, None, model='m', key='k', retries=1, sleep=waits.append)
    assert waits == [40.0]


def test_gemini_does_not_retry_auth_failures(monkeypatch):
    _replies(monkeypatch, _http(403, 'API key not valid'))
    with pytest.raises(gemini.GeminiError) as caught:
        gemini.ask({'uri': 'u'}, None, model='m', key='k', retries=2, sleep=lambda s: pytest.fail('no retry'))
    assert caught.value.category == 'auth'


def test_gemini_raises_after_exhausting_retries(monkeypatch):
    _replies(monkeypatch, _http(503, 'busy'), _http(503, 'busy'), _http(503, 'busy'))
    with pytest.raises(SystemExit, match='Gemini service'):
        gemini.ask({'uri': 'u'}, None, model='m', key='k', retries=2, sleep=lambda s: None)


# --- Settings and opt-in local fallback ---------------------------------------------------

def test_new_settings_defaults_and_validation(monkeypatch):
    cfg = config.get_config()
    assert cfg['gemini_fallback'] == 'none' and cfg['gemini_retries'] == 2 and cfg['auto_cues'] is False
    monkeypatch.setenv('WATCH_GEMINI_FALLBACK', 'somewhere')
    with pytest.raises(config.ConfigError):
        config.get_config()


def _failing_gemini(monkeypatch):
    def boom(*a, **kw):
        raise SystemExit('Gemini service: Gemini had a server error; retry shortly. Detail: HTTP 503')
    monkeypatch.setattr(gemini, 'ask', boom)
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')


def test_fallback_runs_the_local_engine_and_says_so(monkeypatch, static_clip, capsys):
    _failing_gemini(monkeypatch)
    monkeypatch.setattr(gemini, 'upload_file', lambda *a, **kw: {'name': 'files/x', 'uri': 'u', 'mime_type': 'video/mp4'})
    monkeypatch.setattr(gemini, 'delete_file', lambda *a: None)
    monkeypatch.setenv('WATCH_GEMINI_FALLBACK', 'local')
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--no-whisper'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert '**Engine:** local (fallback)' in out and 'WATCH_GEMINI_FALLBACK=local' in out and '## Frames' in out


def test_explicit_engine_gemini_never_falls_back(monkeypatch, capsys):
    _failing_gemini(monkeypatch)
    monkeypatch.setenv('WATCH_GEMINI_FALLBACK', 'local')
    monkeypatch.setattr(watch, 'fetch_captions', lambda *a, **kw: pytest.fail('explicit gemini must not fall back'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://youtu.be/abc', '--engine', 'gemini'])
    assert watch.main() == 1
    assert 'Gemini service' in capsys.readouterr().out


# --- Spoken visual cues -------------------------------------------------------------------

def test_visual_cues_find_pointing_phrases_spaced_and_capped():
    segments = [{'start': 5, 'end': 6, 'text': 'As you can see the chart'},
                {'start': 8, 'end': 9, 'text': 'look at this'},           # within 10s of the last cue
                {'start': 30, 'end': 31, 'text': 'nothing visual here mate'},
                {'start': 40, 'end': 41, 'text': "Here's the result"},
                {'start': 90, 'end': 91, 'text': 'Take a look at the settings'}]
    assert transcribe.visual_cues(segments, 10) == [6.0, 41.0, 91.0]
    assert transcribe.visual_cues(segments, 2) == [6.0, 91.0]
    assert transcribe.visual_cues(segments, 0) == []


def test_auto_cues_add_frames_when_enabled(monkeypatch, tmp_path, cut_clip, capsys):
    subtitle = tmp_path / 'c.vtt'
    subtitle.write_text('WEBVTT\n\n00:01.000 --> 00:02.000\nas you can see it turns red\n')
    monkeypatch.setattr(watch, 'fetch_captions', lambda *a, **kw: {
        'subtitle_path': str(subtitle), 'info': {'title': 't', 'duration': 5},
        'caption_track': {'language': 'en', 'kind': 'manual', 'provenance': 'original'}})
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: {'video_path': str(cut_clip)})
    monkeypatch.setenv('WATCH_AUTO_CUES', '1')
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://example.com/v', '--engine', 'local', '--no-whisper'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert '**Auto cues:** 1' in out and 'cue_' in out


# --- Batch mode ---------------------------------------------------------------------------

def test_batch_reads_report_fields():
    text = '# watch: video report\n\n- **Source:** x\n- **Title:** My Video\n- **Engine:** gemini-3.7-flash (agentic)\n'
    assert watch_batch.report_field(text, 'Title') == 'My Video'
    assert watch_batch.report_field(text, 'Uploader') == ''


def test_batch_runs_each_source_and_writes_an_index(monkeypatch, tmp_path, capsys):
    calls = []
    class Done:
        returncode = 0
    def fake_run(cmd, stdout=None, **kw):
        calls.append(cmd)
        stdout.write(f'# watch: video report\n\n- **Title:** Video {len(calls)}\n')
        return Done()
    monkeypatch.setattr(watch_batch.subprocess, 'run', fake_run)
    monkeypatch.setattr(watch_batch.time, 'sleep', lambda s: None)
    monkeypatch.setattr(sys, 'argv', ['watch_batch', 'https://a.test/1', 'https://a.test/2', 'https://a.test/1',
                                      '--out-dir', str(tmp_path), '--question', 'Q', '--', '--detail', 'efficient'])
    assert watch_batch.main() == 0
    assert len(calls) == 2  # duplicate source skipped
    assert calls[0][-4:] == ['--detail', 'efficient', '--question', 'Q']
    out = capsys.readouterr().out
    assert '| 1 | https://a.test/1 | Video 1 |' in out and (tmp_path / 'report-02.md').exists()
