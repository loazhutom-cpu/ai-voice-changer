import { contextBridge, ipcRenderer } from 'electron';

export interface IElectronAPI {
  toggleVoice: (active: boolean) => Promise<{ success: boolean; isConverting: boolean; mocked?: boolean }>;
  setVoicePreset: (presetId: string) => Promise<{ success: boolean; presetId: string; mocked?: boolean }>;
  updateAudioSettings: (settings: any) => Promise<{ success: boolean; mocked?: boolean }>;
  getAudioDevices: () => Promise<{ inputs: any[]; outputs: any[]; mocked?: boolean }>;
  startPipeline: (inputDeviceId?: string, outputDeviceId?: string) => Promise<{ success: boolean; mocked?: boolean }>;
  stopPipeline: () => Promise<{ success: boolean; mocked?: boolean }>;
  getEngineStatus: () => Promise<any>;
  getAudioLevels: () => Promise<{ inputLevel: number; outputLevel: number; inputPeak: number; outputPeak: number; mocked?: boolean }>;
  getVoicePresets: () => Promise<any[]>;
}

const api: IElectronAPI = {
  toggleVoice: (active: boolean) => ipcRenderer.invoke('voice:toggle', active),
  setVoicePreset: (presetId: string) => ipcRenderer.invoke('voice:set-preset', presetId),
  updateAudioSettings: (settings: any) => ipcRenderer.invoke('audio:update-settings', settings),
  getAudioDevices: () => ipcRenderer.invoke('audio:get-devices'),
  startPipeline: (inputDeviceId?: string, outputDeviceId?: string) =>
    ipcRenderer.invoke('pipeline:start', inputDeviceId, outputDeviceId),
  stopPipeline: () => ipcRenderer.invoke('pipeline:stop'),
  getEngineStatus: () => ipcRenderer.invoke('pipeline:get-status'),
  getAudioLevels: () => ipcRenderer.invoke('audio:get-levels'),
  getVoicePresets: () => ipcRenderer.invoke('voice:get-presets'),
};

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electronAPI', api);
  } catch (error) {
    console.error('Failed to expose electronAPI on contextBridge:', error);
  }
} else {
  // @ts-ignore (isolatedModules declaration)
  window.electronAPI = api;
}

declare global {
  interface Window {
    electronAPI: IElectronAPI;
  }
}
