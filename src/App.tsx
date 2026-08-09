import React, { useEffect, useRef } from 'react';
import {
  Mic,
  Volume2,
  Power,
  Sparkles,
  Settings,
  Activity,
  Cpu,
  Layers,
  RefreshCw,
  Zap,
  Sliders,
  ShieldCheck,
  Radio,
} from 'lucide-react';

import { useVoiceEngine } from './hooks/useVoiceEngine';
import { AudioMeter } from './components/AudioMeter';
import { VoicePresetSelector } from './components/VoicePresetSelector';
import { LatencyDisplay } from './components/LatencyDisplay';
import { SettingsPanel } from './components/SettingsPanel';

export function App() {
  const {
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
    toggleConversion,
    selectPreset,
    updateSettings,
    setSelectedInputDevice,
    setSelectedOutputDevice,
    startPipeline,
    stopPipeline,
    refreshDevices,
  } = useVoiceEngine();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Canvas Waveform visualizer placeholder animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let phase = 0;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const width = canvas.width;
      const height = canvas.height;
      const centerY = height / 2;

      // Draw background grid lines
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      if (isPipelineRunning) {
        phase += 0.08;
        const amplitude = isConverting ? (audioLevels.outputLevel / 100) * (height / 2.5) : (audioLevels.inputLevel / 100) * (height / 3.5);

        // Draw primary audio waveform wave
        ctx.beginPath();
        ctx.lineWidth = isConverting ? 2.5 : 1.5;
        ctx.strokeStyle = isConverting ? '#818cf8' : '#38bdf8';

        for (let x = 0; x < width; x += 3) {
          const sinValue =
            Math.sin(x * 0.03 + phase) * Math.cos(x * 0.01 + phase * 0.5) * (amplitude + 2);
          const y = centerY + sinValue;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw secondary harmonic glow wave when converting
        if (isConverting) {
          ctx.beginPath();
          ctx.lineWidth = 1;
          ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)';
          for (let x = 0; x < width; x += 4) {
            const sinValue = Math.sin(x * 0.05 - phase * 1.2) * (amplitude * 0.7);
            const y = centerY + sinValue;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      } else {
        // Flat line when engine is stopped
        ctx.beginPath();
        ctx.strokeStyle = '#334155';
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPipelineRunning, isConverting, audioLevels]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col select-none overflow-x-hidden">
      {/* Top Header Bar */}
      <header className="bg-slate-900/90 border-b border-slate-800 px-6 py-3 flex items-center justify-between sticky top-0 z-40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-indigo-400 animate-pulse" />
            </div>
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-wide flex items-center gap-2">
              AI Voice Changer
              <span className="text-[10px] bg-indigo-950 text-indigo-300 font-mono px-2 py-0.5 rounded border border-indigo-800/60 uppercase">
                Studio Edition
              </span>
            </h1>
            <p className="text-xs text-slate-400">Real-time Neural Latency Voice Conversion Engine</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Engine Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950 border border-slate-800 text-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                !isConnected
                  ? 'bg-rose-500'
                  : isConverting
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'
                  : 'bg-amber-400'
              }`}
            />
            <span className="font-medium text-slate-300">
              {!isConnected ? 'Engine Disconnected' : isConverting ? 'AI Active' : 'Pass-Through (Bypassed)'}
            </span>
          </div>

          {/* Master Engine Toggle */}
          <button
            onClick={() => (isPipelineRunning ? stopPipeline() : startPipeline())}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              isPipelineRunning
                ? 'bg-rose-950/60 border-rose-800/60 text-rose-300 hover:bg-rose-900'
                : 'bg-emerald-950/60 border-emerald-800/60 text-emerald-300 hover:bg-emerald-900'
            }`}
          >
            <Power className="w-3.5 h-3.5" />
            {isPipelineRunning ? 'Stop Engine' : 'Start Engine'}
          </button>
        </div>
      </header>

      {/* Main App Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">
        {/* Device Configuration Bar */}
        <div className="bg-slate-900/60 backdrop-blur-md rounded-2xl p-4 border border-slate-800/80 shadow-xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 items-center">
          {/* Input Device */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Mic className="w-3.5 h-3.5 text-cyan-400" />
              Microphone Input Device
            </label>
            <select
              value={selectedInputDevice}
              onChange={(e) => setSelectedInputDevice(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl p-2.5 focus:border-indigo-500 focus:outline-none"
            >
              {devices.inputs.map((dev) => (
                <option key={dev.id} value={dev.id}>
                  {dev.name} {dev.isDefault ? '(Default)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Output Device */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Volume2 className="w-3.5 h-3.5 text-indigo-400" />
              Virtual Cable Output Device
            </label>
            <select
              value={selectedOutputDevice}
              onChange={(e) => setSelectedOutputDevice(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl p-2.5 focus:border-indigo-500 focus:outline-none"
            >
              {devices.outputs.map((dev) => (
                <option key={dev.id} value={dev.id}>
                  {dev.name} {dev.isDefault ? '(Virtual Output)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Hardware Refresh Button */}
          <div className="flex items-end justify-between md:justify-end gap-2 pt-2 md:pt-0">
            <button
              onClick={refreshDevices}
              className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-400" />
              Refresh Devices
            </button>
          </div>
        </div>

        {/* Central Voice Conversion Big Toggle Hero Section */}
        <div className="relative bg-gradient-to-b from-slate-900/90 to-slate-950 rounded-2xl p-6 border border-slate-800/80 shadow-2xl overflow-hidden flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2 text-center md:text-left z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-800/50 text-indigo-300 text-xs font-medium">
              <Radio className="w-3.5 h-3.5 animate-pulse text-indigo-400" />
              <span>Model Loaded: {currentPreset.name}</span>
            </div>
            <h2 className="text-2xl font-extrabold text-white tracking-tight">
              {isConverting ? 'Real-Time Voice Conversion Active' : 'AI Voice Conversion Bypassed'}
            </h2>
            <p className="text-xs text-slate-400 max-w-lg">
              Toggle to start routing transformed microphone audio into Discord, OBS, Zoom, or games via virtual audio device.
            </p>
          </div>

          {/* Central Animated Toggle Button */}
          <div className="z-10 flex flex-col items-center gap-2">
            <button
              onClick={() => toggleConversion()}
              className={`relative group w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 transform active:scale-95 ${
                isConverting
                  ? 'bg-gradient-to-tr from-indigo-600 to-pink-600 shadow-[0_0_40px_rgba(99,102,241,0.5)] ring-4 ring-indigo-400/30'
                  : 'bg-slate-800 hover:bg-slate-700 shadow-lg border border-slate-700'
              }`}
            >
              <Power
                className={`w-10 h-10 transition-colors ${
                  isConverting ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'
                }`}
              />
            </button>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              {isConverting ? 'Click to Bypass' : 'Click to Activate AI Voice'}
            </span>
          </div>

          {/* Waveform Canvas Visualizer Background Box */}
          <div className="w-full md:w-72 h-20 bg-slate-950/80 rounded-xl border border-slate-800/80 p-2 relative overflow-hidden flex items-center">
            <canvas ref={canvasRef} width={280} height={70} className="w-full h-full" />
            <div className="absolute bottom-1 right-2 text-[9px] text-slate-500 font-mono">
              WAVEFORM MONITOR
            </div>
          </div>
        </div>

        {/* Quick Parameters Toolbar */}
        <div className="bg-slate-900/40 rounded-xl p-3 border border-slate-800/60 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span className="text-slate-300 font-medium">Quick Gain:</span>
              <input
                type="range"
                min="0.0"
                max="2.0"
                step="0.05"
                value={settings.gain}
                onChange={(e) => updateSettings({ gain: parseFloat(e.target.value) })}
                className="accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg w-28"
              />
              <span className="font-mono text-indigo-300 text-xs">{(settings.gain * 100).toFixed(0)}%</span>
            </div>

            <div className="h-4 w-px bg-slate-800" />

            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span className="text-slate-300 font-medium">Noise Suppression:</span>
              <button
                onClick={() => updateSettings({ noiseSuppression: !settings.noiseSuppression })}
                className={`px-2.5 py-0.5 rounded text-[11px] font-semibold border transition-colors ${
                  settings.noiseSuppression
                    ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                {settings.noiseSuppression ? 'Enabled' : 'Off'}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3 text-slate-400 font-mono text-[11px]">
            <span>Latency: {latency.totalMs}ms</span>
            <span>•</span>
            <span>Pitch: {currentPreset.pitchShift > 0 ? `+${currentPreset.pitchShift}` : currentPreset.pitchShift}st</span>
          </div>
        </div>

        {/* Main Grid Components */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Preset Selector + Audio Meter */}
          <div className="space-y-6">
            <VoicePresetSelector
              presets={presets}
              activePresetId={activePresetId}
              onSelectPreset={selectPreset}
              isConverting={isConverting}
            />

            <AudioMeter levels={audioLevels} isConverting={isConverting} gain={settings.gain} />
          </div>

          {/* Right Column: Latency Display + Audio Settings Panel */}
          <div className="space-y-6">
            <LatencyDisplay latency={latency} isConverting={isConverting} />

            <SettingsPanel settings={settings} onUpdateSettings={updateSettings} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-900/80 border-t border-slate-800/80 px-6 py-2.5 text-xs text-slate-500 flex flex-col sm:flex-row justify-between items-center gap-2 mt-auto">
        <div className="flex items-center gap-3">
          <span>AI Voice Changer Desktop Frontend Scaffold v1.0.0</span>
          <span>•</span>
          <span>FastAPI Port: 7860</span>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>Local Engine Backend Connected</span>
        </div>
      </footer>
    </div>
  );
}
