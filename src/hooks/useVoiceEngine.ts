import { useState, useEffect, useCallback, useRef } from 'react';
import { AudioSettings, AudioDevice, VoicePreset, AudioLevels, LatencyInfo, EngineStatus } from '../types';

const DEFAULT_SETTINGS: AudioSettings = {
  gain: 1.0,
  noiseSuppression: true,
  noiseSuppressionMode: 'rnnoise',
  echoCancellation: true,
  compressorEnabled: true,
  compressorThreshold: -18,
  compressorRatio: 4,
  deEsserEnabled: false,
  reverbAmount: 0.15,
  eqLow: 2,
  eqMid: 0,
  eqHigh: 1,
  limiterThreshold: -1,
};

const DEFAULT_PRESETS: VoicePreset[] = [
  {
    id: 'cyber-hero',
    name: 'Cybernetic Hero',
    description: 'Deep futuristic sci-fi voice with subtle harmonic resonance',
    category: 'Character',
    pitchShift: -4,
    gender: 'Male',
    latencyMs: 38,
    sampleRate: 48000,
  },
  {
    id: 'anime-waifu',
    name: 'Anime Companion',
    description: 'Bright high-pitch anime character tone with formants shift',
    category: 'Anime',
    pitchShift: 5,
    gender: 'Female',
    latencyMs: 45,
    sampleRate: 48000,
  },
  {
    id: 'radio-broadcaster',
    name: 'Broadcast Announcer',
    description: 'Rich warm radio announcer with dynamic compression and bass warmth',
    category: 'Celebrity',
    pitchShift: -2,
    gender: 'Male',
    latencyMs: 32,
    sampleRate: 48000,
  },
  {
    id: 'mecha-robot',
    name: 'Mecha Dreadnought',
    description: 'Heavy metallic robotic voice with dual pitch shift oscillators',
    category: 'Robotic',
    pitchShift: -8,
    gender: 'Alien',
    latencyMs: 52,
    sampleRate: 48000,
  },
  {
    id: 'elf-mage',
    name: 'Elven Sorceress',
    description: 'Ethereal airy voice with soft reverb and crystal high end',
    category: 'Fantasy',
    pitchShift: 3,
    gender: 'Female',
    latencyMs: 41,
    sampleRate: 48000,
  },
  {
    id: 'custom-v1',
    name: 'Custom Trained V1',
    description: 'User-trained RVC model loaded from local weights file',
    category: 'Custom',
    pitchShift: 0,
    gender: 'Neutral',
    latencyMs: 65,
    sampleRate: 48000,
    isCustom: true,
  },
];

export function useVoiceEngine() {
  const [isConverting, setIsConverting] = useState<boolean>(false);
  const [isPipelineRunning, setIsPipelineRunning] = useState<boolean>(true);
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [presets, setPresets] = useState<VoicePreset[]>(DEFAULT_PRESETS);
  const [activePresetId, setActivePresetId] = useState<string>('cyber-hero');
  const [settings, setSettings] = useState<AudioSettings>(DEFAULT_SETTINGS);
  
  const [devices, setDevices] = useState<{ inputs: AudioDevice[]; outputs: AudioDevice[] }>({
    inputs: [
      { id: 'default-in', name: 'Default Microphone (Realtek HD)', type: 'input', isDefault: true },
      { id: 'mic-usb', name: 'Elgato Wave:3 Condenser Mic', type: 'input' },
      { id: 'mic-studio', name: 'Focusrite Scarlett Solo', type: 'input' },
    ],
    outputs: [
      { id: 'default-out', name: 'VB-Audio Virtual Cable (AI Voice)', type: 'output', isDefault: true },
      { id: 'speakers-usb', name: 'Headphones (Realtek Audio)', type: 'output' },
      { id: 'discord-out', name: 'Discord Virtual Audio Device', type: 'output' },
    ],
  });

  const [selectedInputDevice, setSelectedInputDevice] = useState<string>('default-in');
  const [selectedOutputDevice, setSelectedOutputDevice] = useState<string>('default-out');

  const [audioLevels, setAudioLevels] = useState<AudioLevels>({
    inputLevel: 0,
    outputLevel: 0,
    inputPeak: 0,
    outputPeak: 0,
  });

  const [latency, setLatency] = useState<LatencyInfo>({
    totalMs: 42,
    captureMs: 8,
    inferenceMs: 26,
    bufferMs: 8,
  });

  const [error, setError] = useState<string | null>(null);

  const isElectronAvailable = typeof window !== 'undefined' && Boolean(window.electronAPI);

  // Poll for audio levels and engine status
  useEffect(() => {
    const interval = setInterval(async () => {
      if (isElectronAvailable) {
        try {
          const levels = await window.electronAPI.getAudioLevels();
          if (levels) {
            setAudioLevels({
              inputLevel: levels.inputLevel || 0,
              outputLevel: isConverting ? levels.outputLevel || 0 : 0,
              inputPeak: levels.inputPeak || 0,
              outputPeak: isConverting ? levels.outputPeak || 0 : 0,
              isClippingInput: (levels.inputPeak || 0) > 95,
              isClippingOutput: isConverting && (levels.outputPeak || 0) > 95,
            });
          }

          const status: EngineStatus = await window.electronAPI.getEngineStatus();
          if (status) {
            setIsConnected(status.isConnected ?? true);
            if (typeof status.isPipelineRunning === 'boolean') {
              setIsPipelineRunning(status.isPipelineRunning);
            }
          }
        } catch (err) {
          console.warn('Error fetching audio levels via IPC:', err);
        }
      } else {
        // Fallback simulation mode for browser preview
        const time = Date.now() / 250;
        const activeMultiplier = isConverting ? 1 : 0.05;
        const inLvl = isPipelineRunning ? Math.round(Math.max(0, 35 + Math.sin(time * 1.8) * 30 + Math.cos(time * 3) * 15)) : 0;
        const outLvl = isPipelineRunning ? Math.round(Math.max(0, (inLvl * 1.1 + Math.sin(time * 2.5) * 10) * activeMultiplier)) : 0;

        setAudioLevels({
          inputLevel: inLvl,
          outputLevel: Math.min(100, outLvl),
          inputPeak: Math.min(100, inLvl + 6),
          outputPeak: Math.min(100, outLvl + 8),
          isClippingInput: inLvl > 92,
          isClippingOutput: outLvl > 92,
        });

        // Simulate micro latency variations
        const activePreset = presets.find((p) => p.id === activePresetId);
        const baseLatency = activePreset ? activePreset.latencyMs : 42;
        const jitter = Math.floor(Math.sin(time) * 4);
        setLatency({
          totalMs: Math.max(15, baseLatency + jitter),
          captureMs: 8,
          inferenceMs: Math.max(10, baseLatency - 16 + jitter),
          bufferMs: 8,
        });
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isConverting, isPipelineRunning, activePresetId, presets, isElectronAvailable]);

  // Load initial devices and presets on mount
  useEffect(() => {
    async function loadInitialData() {
      if (isElectronAvailable) {
        try {
          const fetchedDevices = await window.electronAPI.getAudioDevices();
          if (fetchedDevices && fetchedDevices.inputs && fetchedDevices.outputs) {
            setDevices({
              inputs: fetchedDevices.inputs,
              outputs: fetchedDevices.outputs,
            });
            if (fetchedDevices.inputs.length > 0) setSelectedInputDevice(fetchedDevices.inputs[0].id);
            if (fetchedDevices.outputs.length > 0) setSelectedOutputDevice(fetchedDevices.outputs[0].id);
          }

          const fetchedPresets = await window.electronAPI.getVoicePresets();
          if (fetchedPresets && fetchedPresets.length > 0) {
            setPresets(fetchedPresets);
          }
        } catch (err) {
          console.error('Failed to load initial hardware/preset configs:', err);
        }
      }
    }
    loadInitialData();
  }, [isElectronAvailable]);

  // Actions
  const toggleConversion = useCallback(async (forcedState?: boolean) => {
    const targetState = forcedState !== undefined ? forcedState : !isConverting;
    setIsConverting(targetState);
    if (isElectronAvailable) {
      try {
        await window.electronAPI.toggleVoice(targetState);
      } catch (err) {
        console.error('Failed to toggle voice conversion:', err);
        setError('Failed to communicate with voice engine');
      }
    }
  }, [isConverting, isElectronAvailable]);

  const selectPreset = useCallback(async (presetId: string) => {
    setActivePresetId(presetId);
    const selected = presets.find((p) => p.id === presetId);
    if (selected) {
      setLatency((prev) => ({
        ...prev,
        totalMs: selected.latencyMs,
        inferenceMs: selected.latencyMs - 16,
      }));
    }
    if (isElectronAvailable) {
      try {
        await window.electronAPI.setVoicePreset(presetId);
      } catch (err) {
        console.error('Failed to set voice preset:', err);
      }
    }
  }, [presets, isElectronAvailable]);

  const updateSettings = useCallback(async (newSettings: Partial<AudioSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...newSettings };
      if (isElectronAvailable) {
        window.electronAPI.updateAudioSettings(updated).catch((err) => {
          console.error('Failed to update audio settings:', err);
        });
      }
      return updated;
    });
  }, [isElectronAvailable]);

  const startPipeline = useCallback(async () => {
    setIsPipelineRunning(true);
    if (isElectronAvailable) {
      try {
        await window.electronAPI.startPipeline(selectedInputDevice, selectedOutputDevice);
      } catch (err) {
        console.error('Failed to start pipeline:', err);
      }
    }
  }, [selectedInputDevice, selectedOutputDevice, isElectronAvailable]);

  const stopPipeline = useCallback(async () => {
    setIsPipelineRunning(false);
    setIsConverting(false);
    if (isElectronAvailable) {
      try {
        await window.electronAPI.stopPipeline();
      } catch (err) {
        console.error('Failed to stop pipeline:', err);
      }
    }
  }, [isElectronAvailable]);

  const refreshDevices = useCallback(async () => {
    if (isElectronAvailable) {
      try {
        const fetched = await window.electronAPI.getAudioDevices();
        if (fetched?.inputs && fetched?.outputs) {
          setDevices({ inputs: fetched.inputs, outputs: fetched.outputs });
        }
      } catch (err) {
        console.error('Failed to refresh devices:', err);
      }
    }
  }, [isElectronAvailable]);

  const currentPreset = presets.find((p) => p.id === activePresetId) || presets[0];

  return {
    isConverting,
    isPipelineRunning,
    isConnected,
    currentPreset,
    activePresetId,
    presets,
    settings,
    devices,
    selectedInputDevice,
    selectedOutputDevice,
    audioLevels,
    latency,
    error,
    toggleConversion,
    selectPreset,
    updateSettings,
    setSelectedInputDevice,
    setSelectedOutputDevice,
    startPipeline,
    stopPipeline,
    refreshDevices,
  };
}
