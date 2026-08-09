import React, { useState } from 'react';
import { Sliders, Volume2, ShieldCheck, Disc, Waves, Activity, RotateCcw, ChevronDown, ChevronUp } from 'lucide-react';
import { AudioSettings } from '../types';

interface SettingsPanelProps {
  settings: AudioSettings;
  onUpdateSettings: (newSettings: Partial<AudioSettings>) => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ settings, onUpdateSettings }) => {
  const [activeTab, setActiveTab] = useState<'dsp' | 'eq' | 'dynamics'>('dsp');
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  const resetDefaults = () => {
    onUpdateSettings({
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
    });
  };

  return (
    <div className="bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-800/80 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Audio Processing & FX Controls
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={resetDefaults}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/60 hover:bg-slate-800 px-2.5 py-1 rounded transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            Reset Defaults
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 text-slate-400 hover:text-slate-200 rounded hover:bg-slate-800"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 space-y-5">
          {/* Tabs */}
          <div className="flex border-b border-slate-800 gap-4 text-xs font-medium">
            <button
              onClick={() => setActiveTab('dsp')}
              className={`pb-2 transition-colors border-b-2 ${
                activeTab === 'dsp'
                  ? 'border-indigo-500 text-indigo-400 font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              Noise & Clean DSP
            </button>
            <button
              onClick={() => setActiveTab('dynamics')}
              className={`pb-2 transition-colors border-b-2 ${
                activeTab === 'dynamics'
                  ? 'border-indigo-500 text-indigo-400 font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              Dynamics & Effects
            </button>
            <button
              onClick={() => setActiveTab('eq')}
              className={`pb-2 transition-colors border-b-2 ${
                activeTab === 'eq'
                  ? 'border-indigo-500 text-indigo-400 font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              3-Band Equalizer
            </button>
          </div>

          {/* Tab 1: DSP & Noise Filters */}
          {activeTab === 'dsp' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Gain Control */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Volume2 className="w-3.5 h-3.5 text-indigo-400" />
                    Input Gain Booster
                  </label>
                  <span className="font-mono text-indigo-300">{(settings.gain * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.05"
                  value={settings.gain}
                  onChange={(e) => onUpdateSettings({ gain: parseFloat(e.target.value) })}
                  className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Mute (0%)</span>
                  <span>100% (Unity)</span>
                  <span>200% (+6dB)</span>
                </div>
              </div>

              {/* Noise Suppression */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    AI Noise Suppression
                  </label>
                  <input
                    type="checkbox"
                    checked={settings.noiseSuppression}
                    onChange={(e) => onUpdateSettings({ noiseSuppression: e.target.checked })}
                    className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
                  />
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-[11px] text-slate-400">Mode:</span>
                  <select
                    disabled={!settings.noiseSuppression}
                    value={settings.noiseSuppressionMode}
                    onChange={(e) =>
                      onUpdateSettings({
                        noiseSuppressionMode: e.target.value as any,
                      })
                    }
                    className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1 w-full disabled:opacity-50"
                  >
                    <option value="rnnoise">RNNoise (Ultra Low Latency)</option>
                    <option value="deepfilternet">DeepFilterNet (High Quality AI)</option>
                    <option value="speex">Speex DSP (Classic DSP)</option>
                  </select>
                </div>
              </div>

              {/* Echo Cancellation */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Waves className="w-3.5 h-3.5 text-cyan-400" />
                    Acoustic Echo Cancellation
                  </label>
                  <input
                    type="checkbox"
                    checked={settings.echoCancellation}
                    onChange={(e) => onUpdateSettings({ echoCancellation: e.target.checked })}
                    className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
                  />
                </div>
                <p className="text-[11px] text-slate-400">
                  Prevents speaker feedback loop when using open studio monitors or high sensitivity mics.
                </p>
              </div>

              {/* De-Esser */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-violet-400" />
                    Sibilance De-Esser
                  </label>
                  <input
                    type="checkbox"
                    checked={settings.deEsserEnabled}
                    onChange={(e) => onUpdateSettings({ deEsserEnabled: e.target.checked })}
                    className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
                  />
                </div>
                <p className="text-[11px] text-slate-400">
                  Softens sharp harsh 'S' and 'T' consonants in high pitch converted audio.
                </p>
              </div>
            </div>
          )}

          {/* Tab 2: Dynamics & Reverb */}
          {activeTab === 'dynamics' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Compressor */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Disc className="w-3.5 h-3.5 text-amber-400" />
                    Dynamic Compressor
                  </label>
                  <input
                    type="checkbox"
                    checked={settings.compressorEnabled}
                    onChange={(e) => onUpdateSettings({ compressorEnabled: e.target.checked })}
                    className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
                  />
                </div>
                <div className="space-y-2 pt-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-400">Threshold:</span>
                    <span className="font-mono text-slate-200">{settings.compressorThreshold} dB</span>
                  </div>
                  <input
                    type="range"
                    min="-40"
                    max="0"
                    disabled={!settings.compressorEnabled}
                    value={settings.compressorThreshold}
                    onChange={(e) => onUpdateSettings({ compressorThreshold: parseInt(e.target.value) })}
                    className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg disabled:opacity-50"
                  />
                </div>
              </div>

              {/* Reverb */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Waves className="w-3.5 h-3.5 text-indigo-400" />
                    Room Reverb Amount
                  </label>
                  <span className="font-mono text-indigo-300">{(settings.reverbAmount * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.8"
                  step="0.02"
                  value={settings.reverbAmount}
                  onChange={(e) => onUpdateSettings({ reverbAmount: parseFloat(e.target.value) })}
                  className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
                <p className="text-[11px] text-slate-400">
                  Adds subtle room spatialization to make converted voice sound more natural.
                </p>
              </div>

              {/* Peak Limiter */}
              <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2 md:col-span-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-rose-400" />
                    Safety Peak Limiter Threshold
                  </label>
                  <span className="font-mono text-rose-300">{settings.limiterThreshold} dB</span>
                </div>
                <input
                  type="range"
                  min="-12"
                  max="0"
                  step="0.5"
                  value={settings.limiterThreshold}
                  onChange={(e) => onUpdateSettings({ limiterThreshold: parseFloat(e.target.value) })}
                  className="w-full accent-rose-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
            </div>
          )}

          {/* Tab 3: Equalizer */}
          {activeTab === 'eq' && (
            <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800 space-y-4">
              <div className="text-xs font-semibold text-slate-300">3-Band Parametric Equalizer</div>
              <div className="grid grid-cols-3 gap-6">
                {/* Low */}
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium">Bass (Low)</span>
                  <input
                    type="range"
                    min="-12"
                    max="12"
                    step="1"
                    value={settings.eqLow}
                    onChange={(e) => onUpdateSettings({ eqLow: parseInt(e.target.value) })}
                    className="h-24 accent-indigo-500 cursor-pointer [writing-mode:vertical-lr] [direction:rtl]"
                  />
                  <span className="text-xs font-mono text-indigo-300">
                    {settings.eqLow > 0 ? `+${settings.eqLow}` : settings.eqLow} dB
                  </span>
                </div>

                {/* Mid */}
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium">Presence (Mid)</span>
                  <input
                    type="range"
                    min="-12"
                    max="12"
                    step="1"
                    value={settings.eqMid}
                    onChange={(e) => onUpdateSettings({ eqMid: parseInt(e.target.value) })}
                    className="h-24 accent-indigo-500 cursor-pointer [writing-mode:vertical-lr] [direction:rtl]"
                  />
                  <span className="text-xs font-mono text-indigo-300">
                    {settings.eqMid > 0 ? `+${settings.eqMid}` : settings.eqMid} dB
                  </span>
                </div>

                {/* High */}
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium">Air (Treble)</span>
                  <input
                    type="range"
                    min="-12"
                    max="12"
                    step="1"
                    value={settings.eqHigh}
                    onChange={(e) => onUpdateSettings({ eqHigh: parseInt(e.target.value) })}
                    className="h-24 accent-indigo-500 cursor-pointer [writing-mode:vertical-lr] [direction:rtl]"
                  />
                  <span className="text-xs font-mono text-indigo-300">
                    {settings.eqHigh > 0 ? `+${settings.eqHigh}` : settings.eqHigh} dB
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
