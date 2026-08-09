# OBS Studio Integration & Audio Routing Guide

This guide details how to integrate **Real-Time AI Voice Changer** with **OBS Studio** for live streaming on Twitch, YouTube, Kick, or recording content.

---

## 1. System Audio Routing Architecture

Understanding how audio flows through your system prevents common issues like audio echo, feedback loops, and uncaptured sound.

```
+-------------------+
|  Physical Mic     |
| (Hardware Input)  |
+---------+---------+
          |
          v
+-------------------+
|  AI Voice Changer |  <-- Applies Noise Removal, AI Voice Conversion & DSP
+---------+---------+
          |
          v (Sends transformed voice)
+-------------------+
|  Virtual Cable    |
| (VB-Cable / Black)|
+---------+---------+
          |
          +-----------------------------------+
          |                                   |
          v                                   v
+-------------------+               +-------------------+
|    OBS Studio     |               |  Discord / Zoom   |
|  (Stream / Record)|               |  (In-Game Voice)  |
+-------------------+               +-------------------+
```

---

## 2. Step-by-Step OBS Studio Configuration

### Step 1: Set Up Virtual Audio Driver
Before launching OBS Studio, ensure your virtual audio driver is installed and recognized by your operating system:
- **Windows**: Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/). Verify `CABLE Input` and `CABLE Output` appear in your Windows Sound Control Panel.
- **macOS**: Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole). Verify `BlackHole 2ch` appears in Audio MIDI Setup.
- **Linux**: Load the PulseAudio loopback module:
  ```bash
  pactl load-module module-null-sink sink_name=Virtual_Mic sink_properties=device.description="Virtual_Mic"
  ```

---

### Step 2: Configure AI Voice Changer Settings
1. Open the AI Voice Changer application or Web Interface (`http://localhost:8000`).
2. Set **Input Device**: Select your physical hardware microphone (e.g., *Elgato Wave 3*, *Focusrite Scarlett*, *USB Mic*).
3. Set **Output Device**:
   - **Windows**: Select `CABLE Input (VB-Audio Virtual Cable)`.
   - **macOS**: Select `BlackHole 2ch`.
   - **Linux**: Select `Virtual_Mic`.
4. Click **Start Stream**. Speak into your microphone and confirm the audio meters show activity without clipping.

![AI Voice Changer Output Configuration](images/app_device_settings.png)
*Figure 1: Setting application output to the virtual audio device.*

---

### Step 3: Configure OBS Audio Settings

1. Launch **OBS Studio**.
2. Go to **Settings** -> **Audio**.
3. In the **Global Audio Devices** section:
   - Locate **Mic/Auxiliary Audio**.
   - Set it to **CABLE Output (VB-Audio Virtual Cable)** (Windows) or **BlackHole 2ch** (macOS).
   - Ensure **Desktop Audio** is set to your regular Headphones/Speakers (so you can hear game sound and desktop audio).

![OBS Global Audio Settings](images/obs_global_audio.png)
*Figure 2: OBS Studio Audio Settings panel showing Mic/Auxiliary mapped to CABLE Output.*

4. Click **Apply** and **OK**.

---

### Step 4: Adding an Audio Input Capture Source (Alternative Method)

If you prefer adding the AI voice as a scene-specific source rather than a global mic:

1. In OBS Studio, navigate to the **Sources** dock.
2. Click the **+** button -> Select **Audio Input Capture**.
3. Name the source (e.g., `AI Voice Changer Mic`).
4. Select **Device**: `CABLE Output (VB-Audio Virtual Cable)` or `BlackHole 2ch`.
5. Click **OK**.

![OBS Audio Input Capture Source](images/obs_audio_source.png)
*Figure 3: Adding Audio Input Capture source in OBS.*

---

### Step 5: Audio Monitoring & Level Verification

1. Speak into your microphone.
2. Check the **Audio Mixer** panel in OBS Studio. You should see yellow/green activity on your `AI Voice Changer Mic` track.
3. **Important**: To monitor your transformed voice in your headphones:
   - Click the gear icon ⚙️ under the Audio Mixer -> Select **Advanced Audio Properties**.
   - Set **Audio Monitoring** for `AI Voice Changer Mic` to **Monitor Off** (recommended during live streaming to prevent hearing yourself delayed) OR **Monitor and Output** (if you wish to hear your voice while testing).

![OBS Advanced Audio Properties](images/obs_advanced_audio.png)
*Figure 4: OBS Advanced Audio Properties configuration.*

---

## 3. Troubleshooting & Common Issues

### Issue 1: Audio Feedback Loop or Echo
- **Symptom**: You hear an escalating infinite echo or stuttering repetition of your voice.
- **Cause**: AI Voice Changer output is feeding into an input device that OBS or your headphones are feeding back into the app.
- **Fix**:
  1. Ensure AI Voice Changer's **Input Device** is explicitly set to your physical microphone (NOT "Default" or "CABLE Output").
  2. Ensure OBS **Desktop Audio** does NOT capture the virtual cable directly.
  3. Set OBS Audio Monitoring to **Monitor Off** for the voice track during streaming.

---

### Issue 2: Audio Delay / Latency Desync with Video
- **Symptom**: Your voice audio is slightly delayed relative to your webcam video feed.
- **Fix**:
  1. Measure total voice conversion latency from the AI Voice Changer dashboard (e.g., 40ms).
  2. In OBS Studio, open **Advanced Audio Properties**.
  3. Add a **Sync Offset** to your Camera video source (e.g., `40 ms`) OR negative offset if video lags audio.

---

### Issue 3: Stuttering, Crackling, or Popping Audio
- **Symptom**: Audio sounds distorted or choppy with robotic glitch artifacts.
- **Cause**: Buffer underrun, CPU thread contention, or mismatched sample rates across devices.
- **Fix**:
  1. Open Windows Sound Control Panel -> Recording/Playback Properties -> Advanced.
  2. Ensure ALL devices (Mic, Headphones, Virtual Cable) are set to **2-channel, 16-bit/24-bit, 48000 Hz (Studio Quality)**.
  3. In AI Voice Changer config, increase block size from `256` to `512` samples.
  4. Ensure GPU acceleration (CUDA / TensorRT) is active.

---

### Issue 4: Virtual Audio Cable Not Appearing in Device List
- **Symptom**: `CABLE Input` or `BlackHole 2ch` does not show up in AI Voice Changer or OBS dropdowns.
- **Fix**:
  1. Restart the AI Voice Changer background service after installing drivers.
  2. On macOS, grant OBS and Terminal permissions under **System Settings** -> **Privacy & Security** -> **Microphone**.
  3. Reinstall VB-Cable as Administrator on Windows and reboot.
