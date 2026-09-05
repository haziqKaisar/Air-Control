"""
Modul kontrol sistem:
- Volume: disimulasikan lewat MEDIA KEY (pyautogui volumeup/volumedown),
  bukan lewat pycaw langsung -- ini jauh lebih reliable di banyak laptop
  Windows karena tidak bergantung COM/driver audio spesifik.
  pycaw tetap dipakai (kalau tersedia) HANYA untuk membaca persentase
  volume yang sebenarnya; kalau tidak tersedia, dipakai estimasi internal.
- Brightness: via screen_brightness_control (sudah terbukti jalan).
"""

import pyautogui


class VolumeController:
    STEP_PERCENT = 2          # perkiraan persen per 1x tombol volume ditekan (khas Windows)
    MAX_STEPS_PER_FRAME = 6   # batas jumlah keypress per panggilan, biar ga nge-spam

    def __init__(self):
        self.pycaw_available = False
        self.interface = None
        self.last_known_percent = 50
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.interface = cast(interface, POINTER(IAudioEndpointVolume))
            self.pycaw_available = True
            self.last_known_percent = self.get_volume_percent()
            print(f"[Volume] Baca level volume via pycaw OK (awal: {self.last_known_percent}%). "
                  f"Kontrol naik/turunnya pakai simulasi tombol volume fisik.")
        except Exception as e:
            print(f"[Volume] pycaw tidak tersedia buat baca level ({e}); pakai estimasi internal.")
            print("[Volume] Kontrol naik/turun tetap jalan via simulasi tombol volume fisik.")

    def get_volume_percent(self):
        if self.pycaw_available:
            try:
                return int(self.interface.GetMasterVolumeLevelScalar() * 100)
            except Exception:
                pass
        return self.last_known_percent

    def nudge_to_percent(self, target_percent):
        """Gerakkan volume sistem mendekati target_percent pakai simulasi tombol fisik."""
        current = self.get_volume_percent()
        delta = target_percent - current
        if abs(delta) < self.STEP_PERCENT:
            return  # deadzone kecil, biar ga getar2 kirim keypress terus-terusan

        steps = int(delta / self.STEP_PERCENT)
        steps = max(-self.MAX_STEPS_PER_FRAME, min(self.MAX_STEPS_PER_FRAME, steps))

        key = "volumeup" if steps > 0 else "volumedown"
        try:
            for _ in range(abs(steps)):
                pyautogui.press(key)
        except Exception as e:
            print(f"[Volume] Gagal kirim tombol volume: {e}")
            return

        self.last_known_percent = max(0, min(100, current + steps * self.STEP_PERCENT))


class BrightnessController:
    def __init__(self):
        self.available = False
        try:
            import screen_brightness_control as sbc
            self.sbc = sbc
            sbc.get_brightness()
            self.available = True
            print("[Brightness] Kontrol brightness sistem aktif.")
        except Exception as e:
            print(f"[Brightness] Tidak bisa akses kontrol brightness sistem: {e}")
            print("[Brightness] Mode tampilan tetap jalan, tapi brightness asli tidak berubah.")

    def set_brightness_percent(self, percent):
        if not self.available:
            return
        try:
            self.sbc.set_brightness(int(max(0, min(100, percent))))
        except Exception:
            pass

    def get_brightness_percent(self):
        if not self.available:
            return 0
        try:
            vals = self.sbc.get_brightness()
            return vals[0] if isinstance(vals, list) else vals
        except Exception:
            return 0