import { app, BrowserWindow, ipcMain, shell } from 'electron';
import { join } from 'path';
import { electronApp, optimizer, is } from '@electron-toolkit/utils';
import http from 'http';

const BACKEND_URL = 'http://localhost:7860';

let mainWindow: BrowserWindow | null = null;

// Helper to send HTTP requests to Python FastAPI backend (localhost:7860)
async function requestBackend<T>(endpoint: string, method = 'GET', body?: any): Promise<T> {
  const url = `${BACKEND_URL}${endpoint}`;
  
  return new Promise((resolve, reject) => {
    try {
      const parsedUrl = new URL(url);
      const postData = body ? JSON.stringify(body) : '';
      
      const options: http.RequestOptions = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port || 7860,
        path: parsedUrl.pathname + parsedUrl.search,
        method: method,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData),
        },
        timeout: 2000,
      };

      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(data ? JSON.parse(data) : ({} as T));
            } catch (err) {
              resolve(data as unknown as T);
            }
          } else {
            reject(new Error(`Backend responded with status ${res.statusCode}: ${data}`));
          }
        });
      });

      req.on('error', (err) => {
        reject(err);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Backend connection timeout'));
      });

      if (postData) {
        req.write(postData);
      }
      req.end();
    } catch (err) {
      reject(err);
    }
  });
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 960,
    minHeight: 680,
    show: false,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0f172a',
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.on('ready-to-show', () => {
    if (mainWindow) {
      mainWindow.show();
    }
  });

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url);
    return { action: 'deny' };
  });

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL']);
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'));
  }
}

// Set up IPC Handlers
function setupIpcHandlers(): void {
  // Toggle voice conversion
  ipcMain.handle('voice:toggle', async (_event, active: boolean) => {
    try {
      const response = await requestBackend<{ success: boolean; isConverting: boolean }>(
        '/api/voice/toggle',
        'POST',
        { active }
      );
      return response;
    } catch (error) {
      console.warn('[IPC] Voice toggle fallback mode:', error);
      return { success: true, isConverting: active, mocked: true };
    }
  });

  // Switch voice preset
  ipcMain.handle('voice:set-preset', async (_event, presetId: string) => {
    try {
      const response = await requestBackend<{ success: boolean; presetId: string }>(
        '/api/voice/preset',
        'POST',
        { presetId }
      );
      return response;
    } catch (error) {
      console.warn('[IPC] Voice preset fallback mode:', error);
      return { success: true, presetId, mocked: true };
    }
  });

  // Adjust audio settings (gain, noise suppression, EQ, etc.)
  ipcMain.handle('audio:update-settings', async (_event, settings: any) => {
    try {
      const response = await requestBackend<{ success: boolean }>(
        '/api/audio/settings',
        'POST',
        settings
      );
      return response;
    } catch (error) {
      console.warn('[IPC] Update audio settings fallback mode:', error);
      return { success: true, mocked: true };
    }
  });

  // Get audio device list
  ipcMain.handle('audio:get-devices', async () => {
    try {
      const devices = await requestBackend<{ inputs: any[]; outputs: any[] }>('/api/audio/devices');
      return devices;
    } catch (error) {
      console.warn('[IPC] Get devices fallback mode:', error);
      return {
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
        mocked: true,
      };
    }
  });

  // Start audio processing pipeline
  ipcMain.handle('pipeline:start', async (_event, inputDeviceId?: string, outputDeviceId?: string) => {
    try {
      const response = await requestBackend<{ success: boolean }>('/api/pipeline/start', 'POST', {
        inputDeviceId,
        outputDeviceId,
      });
      return response;
    } catch (error) {
      console.warn('[IPC] Start pipeline fallback mode:', error);
      return { success: true, mocked: true };
    }
  });

  // Stop audio processing pipeline
  ipcMain.handle('pipeline:stop', async () => {
    try {
      const response = await requestBackend<{ success: boolean }>('/api/pipeline/stop', 'POST');
      return response;
    } catch (error) {
      console.warn('[IPC] Stop pipeline fallback mode:', error);
      return { success: true, mocked: true };
    }
  });

  // Get Engine Status
  ipcMain.handle('pipeline:get-status', async () => {
    try {
      const status = await requestBackend<any>('/api/pipeline/status');
      return { ...status, isConnected: true };
    } catch (error) {
      return {
        isConnected: false,
        isPipelineRunning: true,
        isConverting: false,
        activePresetId: 'cyber-hero',
        latencyMs: 42,
        inputDeviceId: 'default-in',
        outputDeviceId: 'default-out',
        cpuUsage: 14,
        gpuUsage: 28,
        mocked: true,
      };
    }
  });

  // Get Audio Levels for meters
  ipcMain.handle('audio:get-levels', async () => {
    try {
      const levels = await requestBackend<{
        inputLevel: number;
        outputLevel: number;
        inputPeak: number;
        outputPeak: number;
      }>('/api/audio/levels');
      return levels;
    } catch (error) {
      // Return realistic synthetic meter jitter if backend is offline
      const t = Date.now() / 200;
      const baseIn = 40 + Math.sin(t) * 25 + Math.cos(t * 2.3) * 15;
      const baseOut = 45 + Math.cos(t * 1.5) * 28 + Math.sin(t * 3.1) * 10;
      return {
        inputLevel: Math.max(0, Math.min(100, Math.round(baseIn))),
        outputLevel: Math.max(0, Math.min(100, Math.round(baseOut))),
        inputPeak: Math.max(0, Math.min(100, Math.round(baseIn + 8))),
        outputPeak: Math.max(0, Math.min(100, Math.round(baseOut + 10))),
        mocked: true,
      };
    }
  });

  // Get Available Voice Presets
  ipcMain.handle('voice:get-presets', async () => {
    try {
      const presets = await requestBackend<any[]>('/api/voice/presets');
      return presets;
    } catch (error) {
      return [
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
    }
  });
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.aivoicechanger.app');

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window);
  });

  setupIpcHandlers();
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
