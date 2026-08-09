export interface AudioSettings {
  gain: number; // 0.0 to 2.0 (1.0 = 0dB)
  noiseSuppression: boolean;
  noiseSuppressionMode: 'rnnoise' | 'deepfilternet' | 'speex';
  echoCancellation: boolean;
  compressorEnabled: boolean;
  compressorThreshold: number; // -60 to 0 dB
  compressorRatio: number; // 1 to 20
  deEsserEnabled: boolean;
  reverbAmount: number; // 0.0 to 1.0
  eqLow: number; // -12 to +12 dB
  eqMid: number; // -12 to +12 dB
  eqHigh: number; // -12 to +12 dB
  limiterThreshold: number; // -12 to 0 dB
}

export interface AudioDevice {
  id: string;
  name: string;
  type: 'input' | 'output';
  isDefault?: boolean;
}

export interface VoicePreset {
  id: string;
  name: string;
  description: string;
  category: 'Character' | 'Celebrity' | 'Custom' | 'Robotic' | 'Anime' | 'Fantasy';
  pitchShift: number; // semi-tones -12 to +12
  gender: 'Male' | 'Female' | 'Neutral' | 'Alien';
  latencyMs: number;
  sampleRate: number;
  icon?: string;
  isCustom?: boolean;
}

export interface AudioLevels {
  inputLevel: number; // 0 to 100
  outputLevel: number; // 0 to 100
  inputPeak: number; // 0 to 100
  outputPeak: number; // 0 to 100
  isClippingInput?: boolean;
  isClippingOutput?: boolean;
}

export interface LatencyInfo {
  totalMs: number;
  captureMs: number;
  inferenceMs: number;
  bufferMs: number;
}

export interface EngineStatus {
  isConnected: boolean;
  isPipelineRunning: boolean;
  isConverting: boolean;
  activePresetId: string;
  latencyMs: number;
  inputDeviceId: string;
  outputDeviceId: string;
  cpuUsage: number;
  gpuUsage: number;
}
