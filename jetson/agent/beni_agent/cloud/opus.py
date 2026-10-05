"""Minimal libopus encoder via ctypes (optional uplink codec; apt: libopus0). ~3% of one A57 core at 16 kHz VOIP."""
import ctypes
import ctypes.util

_lib = ctypes.CDLL(ctypes.util.find_library("opus") or "libopus.so.0")
_lib.opus_encoder_create.restype = ctypes.c_void_p
_lib.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.opus_encode.restype = ctypes.c_int32
_lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int32]
_lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]

APPLICATION_VOIP = 2048
SET_BITRATE = 4002
SET_COMPLEXITY = 4010


class Encoder:
    def __init__(self, rate=16000, bitrate=24000, complexity=5):
        err = ctypes.c_int()
        self.enc = _lib.opus_encoder_create(rate, 1, APPLICATION_VOIP, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError("opus_encoder_create: %d" % err.value)
        _lib.opus_encoder_ctl(ctypes.c_void_p(self.enc), SET_BITRATE, ctypes.c_int32(bitrate))
        _lib.opus_encoder_ctl(ctypes.c_void_p(self.enc), SET_COMPLEXITY, ctypes.c_int32(complexity))
        self.out = ctypes.create_string_buffer(1275)

    def encode(self, pcm16_le):
        """One 20 ms frame (640 bytes at 16 kHz) -> Opus packet bytes."""
        n = _lib.opus_encode(self.enc, pcm16_le, len(pcm16_le) // 2, self.out, len(self.out))
        if n < 0:
            raise RuntimeError("opus_encode: %d" % n)
        return self.out.raw[:n]

    def __del__(self):
        if getattr(self, "enc", None):
            _lib.opus_encoder_destroy(self.enc)
            self.enc = None
