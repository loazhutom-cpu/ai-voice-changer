import React from 'react';
import { Mic, Volume2, AlertCircle } from 'lucide-react';
import { AudioLevels } from '../types';

interface AudioMeterProps {
  levels: AudioLevels;
  isConverting: boolean;
  gain: number;
}

export const AudioMeter: React.FC<AudioMeterProps> = ({ levels, isConverting, gain }) => {
  const { inputLevel, outputLevel, inputPeak, outputPeak, isClippingInput, isClippingOutput } = levels;

  // Render segments for LED-style meter bar
  const renderMeterBars = (level: number, peak: number, isClipping?: boolean) => {
    const totalSegments = 24;
    const activeSegments = Math.round((level / 100) * totalSegments);
    const peakSegmentIndex = Math.min(totalSegments - 1, Math.round((peak / 100) * totalSegments));

    return (
      <div className="relative flex items-center gap-0.5 h-6 w-full bg-slate-900/90 rounded p-1 border border-slate-800 overflow-hidden">
        {Array.from({ length: totalSegments }).map((_, idx) => {
          const isGreen = idx < 15;
          const isYellow = idx >= 15 && idx < 20;
          const isRed = idx >= 20;

          const isActive = idx < activeSegments;
          const isPeak = idx === peakSegmentIndex && peak > 5;

          let colorClass = 'bg-slate-800';
          if (isActive) {
            if (isGreen) colorClass = 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]';
            else if (isYellow) colorClass = 'bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.6)]';
            else colorClass = 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]';
          } else if (isPeak) {
            colorClass = isRed
              ? 'bg-rose-400 animate-pulse'
              : isYellow
              ? 'bg-amber-300'
              : 'bg-emerald-300';
          }

          return (
            <div
              key={idx}
              className={`h-full flex-1 rounded-sm transition-all duration-75 ${colorClass}`}
            />
          );
        })}

        {isClipping && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 bg-rose-950/80 text-rose-300 text-[10px] font-bold px-1.5 py-0.5 rounded border border-rose-500/50 animate-bounce">
            <AlertCircle className="w-3 h-3 text-rose-400" />
            CLIP
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-800/80 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          Real-Time Audio Levels
        </h3>
        <span className="text-xs text-slate-400 font-mono">Gain: {(gain * 100).toFixed(0)}%</span>
      </div>

      {/* Input Channel */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs text-slate-300">
          <div className="flex items-center gap-1.5 font-medium">
            <Mic className="w-4 h-4 text-cyan-400" />
            <span>Microphone Input</span>
          </div>
          <span className="font-mono text-slate-400">{inputLevel} dBFS</span>
        </div>
        {renderMeterBars(inputLevel, inputPeak, isClippingInput)}
      </div>

      {/* Output Channel */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs text-slate-300">
          <div className="flex items-center gap-1.5 font-medium">
            <Volume2 className={`w-4 h-4 ${isConverting ? 'text-indigo-400' : 'text-slate-500'}`} />
            <span>Processed AI Output</span>
            {!isConverting && <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded ml-1">Bypassed</span>}
          </div>
          <span className="font-mono text-slate-400">{isConverting ? outputLevel : 0} dBFS</span>
        </div>
        {renderMeterBars(isConverting ? outputLevel : 0, isConverting ? outputPeak : 0, isClippingOutput)}
      </div>

      {/* Scale markers */}
      <div className="flex justify-between text-[10px] text-slate-500 font-mono px-1">
        <span>-60dB</span>
        <span>-48dB</span>
        <span>-36dB</span>
        <span>-24dB</span>
        <span>-12dB</span>
        <span>-6dB</span>
        <span className="text-rose-400 font-bold">0dB</span>
      </div>
    </div>
  );
};
