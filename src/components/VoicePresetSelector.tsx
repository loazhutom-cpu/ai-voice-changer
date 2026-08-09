import React, { useState } from 'react';
import { Sparkles, ChevronDown, UserCheck, Shield, Upload, RefreshCw, Zap } from 'lucide-react';
import { VoicePreset } from '../types';

interface VoicePresetSelectorProps {
  presets: VoicePreset[];
  activePresetId: string;
  onSelectPreset: (presetId: string) => void;
  onRefreshPresets?: () => void;
  isConverting: boolean;
}

export const VoicePresetSelector: React.FC<VoicePresetSelectorProps> = ({
  presets,
  activePresetId,
  onSelectPreset,
  onRefreshPresets,
  isConverting,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', 'Character', 'Anime', 'Celebrity', 'Robotic', 'Fantasy', 'Custom'];

  const filteredPresets = presets.filter((p) => {
    if (selectedCategory === 'All') return true;
    return p.category === selectedCategory;
  });

  const activePreset = presets.find((p) => p.id === activePresetId) || presets[0];

  return (
    <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-800/80 shadow-lg space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            AI Voice Model Preset
          </h3>
        </div>
        {onRefreshPresets && (
          <button
            onClick={onRefreshPresets}
            title="Refresh Presets"
            className="p-1 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Main Preset Selector Dropdown Button */}
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`w-full flex items-center justify-between p-3.5 bg-slate-950/80 hover:bg-slate-900 rounded-xl border ${
            isConverting
              ? 'border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.15)]'
              : 'border-slate-800'
          } transition-all text-left`}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-700 flex items-center justify-center text-white font-bold text-lg shadow-inner">
              {activePreset?.name.charAt(0) || 'V'}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-100 text-sm">{activePreset?.name}</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                  {activePreset?.category}
                </span>
                {activePreset?.isCustom && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/50">
                    Custom Model
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 truncate max-w-[280px] sm:max-w-md">
                {activePreset?.description}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="text-right hidden sm:block">
              <div className="text-[11px] text-slate-400 font-mono flex items-center gap-1 justify-end">
                <Zap className="w-3 h-3 text-amber-400" />
                {activePreset?.latencyMs} ms
              </div>
              <div className="text-[10px] text-slate-500">
                Shift: {activePreset?.pitchShift > 0 ? `+${activePreset.pitchShift}` : activePreset?.pitchShift} st
              </div>
            </div>
            <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </div>
        </button>

        {/* Dropdown Menu Overlay */}
        {isOpen && (
          <div className="absolute z-50 left-0 right-0 mt-2 bg-slate-950 border border-slate-800 rounded-xl shadow-2xl p-3 space-y-3">
            {/* Category Filter Pills */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`text-xs px-2.5 py-1 rounded-md transition-colors whitespace-nowrap ${
                    selectedCategory === cat
                      ? 'bg-indigo-600 text-white font-medium'
                      : 'bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Presets List */}
            <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
              {filteredPresets.map((preset) => {
                const isSelected = preset.id === activePresetId;
                return (
                  <button
                    key={preset.id}
                    onClick={() => {
                      onSelectPreset(preset.id);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left p-2.5 rounded-lg border transition-all flex items-center justify-between ${
                      isSelected
                        ? 'bg-indigo-950/60 border-indigo-500/60 text-slate-100'
                        : 'bg-slate-900/50 border-slate-800/60 text-slate-300 hover:bg-slate-800/70 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <div
                        className={`w-8 h-8 rounded flex items-center justify-center font-bold text-xs ${
                          isSelected
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {preset.name.charAt(0)}
                      </div>
                      <div>
                        <div className="text-xs font-semibold flex items-center gap-1.5">
                          {preset.name}
                          {preset.gender && (
                            <span className="text-[10px] text-slate-500 font-normal">
                              ({preset.gender})
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-400 line-clamp-1">
                          {preset.description}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                        {preset.latencyMs}ms
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Custom Model Import Footer */}
            <div className="pt-2 border-t border-slate-800/80 flex justify-between items-center text-xs">
              <span className="text-slate-400">Want custom voice weights?</span>
              <button
                onClick={() => alert('Select .pth or .index RVC model file')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
              >
                <Upload className="w-3.5 h-3.5 text-indigo-400" />
                Import Model (.pth)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
