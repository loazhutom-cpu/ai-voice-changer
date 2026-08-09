import React from 'react';
import { Activity, Clock, Cpu, Gauge } from 'lucide-react';
import { LatencyInfo } from '../types';

interface LatencyDisplayProps {
  latency: LatencyInfo;
  isConverting: boolean;
}

export const LatencyDisplay: React.FC<LatencyDisplayProps> = ({ latency, isConverting }) => {
  const { totalMs, captureMs, inferenceMs, bufferMs } = latency;

  // Determine color and status message based on total latency
  let statusColor = 'text-emerald-400 bg-emerald-950/80 border-emerald-500/40';
  let badgeColor = 'bg-emerald-500';
  let statusText = 'Ultra Low (Real-time Ready)';
  let barColor = 'bg-emerald-500';

  if (totalMs >= 80 && totalMs <= 150) {
    statusColor = 'text-amber-400 bg-amber-950/80 border-amber-500/40';
    badgeColor = 'bg-amber-400';
    statusText = 'Moderate (Minor Buffer Delay)';
    barColor = 'bg-amber-400';
  } else if (totalMs > 150) {
    statusColor = 'text-rose-400 bg-rose-950/80 border-rose-500/40';
    badgeColor = 'bg-rose-500';
    statusText = 'High Latency (Noticeable Delay)';
    barColor = 'bg-rose-500';
  }

  // Calculate percentage of max acceptable threshold (200ms)
  const percentGauge = Math.min(100, Math.round((totalMs / 200) * 100));

  return (
    <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-800/80 shadow-lg space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Pipeline Latency
          </h3>
        </div>
        <div className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${statusColor}`}>
          <span className={`w-2 h-2 rounded-full ${badgeColor} animate-pulse`} />
          <span>{statusText}</span>
        </div>
      </div>

      {/* Main Big Latency Readout */}
      <div className="flex items-baseline justify-between pt-1">
        <div className="flex items-baseline gap-1">
          <span className="text-3xl font-bold font-mono text-slate-100">{totalMs}</span>
          <span className="text-sm text-slate-400 font-semibold">ms</span>
        </div>
        <div className="text-right text-xs text-slate-400">
          Target: <span className="text-slate-200 font-mono">&lt; 80 ms</span>
        </div>
      </div>

      {/* Visual gauge bar */}
      <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800 relative">
        <div
          className={`h-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percentGauge}%` }}
        />
      </div>

      {/* Breakdown metrics */}
      <div className="grid grid-cols-3 gap-2 pt-1 text-center">
        <div className="bg-slate-950/60 rounded-lg p-2 border border-slate-800/80">
          <div className="text-[10px] text-slate-400 uppercase">Capture</div>
          <div className="text-xs font-mono font-semibold text-slate-200">{captureMs} ms</div>
        </div>
        <div className="bg-slate-950/60 rounded-lg p-2 border border-slate-800/80">
          <div className="text-[10px] text-slate-400 uppercase">AI Inference</div>
          <div className="text-xs font-mono font-semibold text-indigo-300">{inferenceMs} ms</div>
        </div>
        <div className="bg-slate-950/60 rounded-lg p-2 border border-slate-800/80">
          <div className="text-[10px] text-slate-400 uppercase">Buffer Sync</div>
          <div className="text-xs font-mono font-semibold text-slate-200">{bufferMs} ms</div>
        </div>
      </div>
    </div>
  );
};
