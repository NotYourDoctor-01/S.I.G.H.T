"""
Text-to-speech for detected signs.

    from speech import SignSpeaker
    speaker = SignSpeaker()
    speaker.speak("thank_you")      # says "thank you" (non-blocking)
    speaker.close()

Test the audio on its own (speaks three phrases in a row):
    python speech.py

Backends:
  * Windows: PowerShell + System.Speech, one separate process per phrase.
    No extra packages, and no engine state that can go silent after the first
    phrase (a known pyttsx3 problem).
  * Other systems: pyttsx3 (Linux also needs: sudo apt install espeak-ng).

Optional: customize what is said per sign with a phrases.json next to the
model, e.g.  {"thank_you": "Thank you very much", "hello": "Hi there!"}
Signs not listed are spoken as their folder name with underscores -> spaces.
"""
import json
import os
import queue
import subprocess
import sys
import threading

IS_WINDOWS = sys.platform.startswith("win")


class SignSpeaker:
    def __init__(self, phrases_file=None, rate=160, volume=1.0):
        self.phrases = {}
        if phrases_file and os.path.exists(phrases_file):
            with open(phrases_file, encoding="utf-8") as f:
                self.phrases = json.load(f)
        self.rate, self.volume = rate, volume
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def text_for(self, sign):
        return self.phrases.get(sign, sign.replace("_", " "))

    def speak(self, sign):
        """Queue a detected sign to be spoken. Returns immediately."""
        text = self.text_for(sign)
        print(f"[speech] queued: {text}")
        self._q.put(text)

    # -- backends ----------------------------------------------------------
    def _say_windows(self, text):
        sapi_rate = max(-10, min(10, round((self.rate - 160) / 15)))
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Rate = {sapi_rate}; "
                  f"$s.Volume = {int(self.volume * 100)}; "
                  "$s.Speak($env:SIGN_TEXT)")
        # Text goes through an environment variable, never into the command
        # string, so odd characters in a phrase can't break the command.
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "SIGN_TEXT": text},
            capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            print(f"[speech error] {proc.stderr.strip()}")

    def _say_pyttsx3(self, text):
        import pyttsx3
        engine = pyttsx3.init()          # fresh engine per phrase
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _run(self):
        say = self._say_windows if IS_WINDOWS else self._say_pyttsx3
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                say(text)
            except Exception as e:
                print(f"[speech error] {e}")

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=10)


if __name__ == "__main__":
    sp = SignSpeaker()
    for phrase in ["hello", "thank_you", "see_you_later"]:
        sp.speak(phrase)
    sp.close()
    print("Done. You should have heard three phrases.")