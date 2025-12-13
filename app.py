import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Zap, ShieldAlert, BarChart3, Database, RefreshCw, AlertTriangle, ArrowDown, TrendingDown } from 'lucide-react';
import { clsx } from 'clsx';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

import { CORE_INDICATORS, SHORT_TERM_INDICATORS, LONG_TERM_INDICATORS } from './constants';
import { fetchFredSeries, fetchFmpQuote, transformData } from './services/dataService';
import { IndicatorConfig, IndicatorData } from './types';

// --- Helper Components ---

const StatusCard = ({ title, count, max, type, isActive }: { title: string, count: number, max: number, type: 'kill' | 'dark', isActive: boolean }) => {
  const ratio = count / max;
  const isRed = ratio > 0.6; // Threshold for visual alarm
  return (
    <div className={clsx(
      "p-6 rounded-xl border transition-all duration-300 relative overflow-hidden",
      isActive ? "ring-2 ring-blue-500 bg-slate-800" : "bg-slate-900",
      isRed ? "border-red-500/50 shadow-[0_0_30px_rgba(220,38,38,0.2)]" : "border-slate-700"
    )}>
      {isRed && <div className="absolute inset-0 bg-red-500/5 animate-pulse pointer-events-none" />}
      <div className="flex justify-between items-start mb-2 relative z-10">
        <h3 className="text-slate-400 text-sm font-semibold uppercase tracking-wider">{title}</h3>
        {type === 'kill' ? <Zap className={isRed ? "text-red-500" : "text-slate-500"} size={20} /> : <ShieldAlert className={isRed ? "text-red-900" : "text-slate-500"} size={20} />}
      </div>
      <div className="flex items-baseline gap-2 relative z-10">
        <span className={clsx("text-4xl font-bold", isRed ? "text-red-500" : "text-slate-200")}>{count}</span>
        <span className="text-slate-500 font-medium">/ {max} Signals</span>
      </div>
      <div className="mt-4 h-2 w-full bg-slate-800 rounded-full overflow-hidden relative z-10">
        <div 
          className={clsx("h-full transition-all duration-1000", isRed ? "bg-red-600" : "bg-blue-500")} 
          style={{ width: `${Math.min((count / max) * 100, 100)}%` }}
        />
      </div>
    </div>
  );
};

interface IndicatorRowProps {
  config: IndicatorConfig;
  data?: IndicatorData;
  isTriggered?: boolean;
}

const IndicatorRow: React.FC<IndicatorRowProps> = ({ config, data, isTriggered }) => {
  // data.history is [Oldest...Newest]
  const chartData = data?.history.map((val, i) => ({ i, val })) || [];
  
  const statusColor = data?.status === 'error' ? 'text-yellow-500' : isTriggered ? 'text-red-500' : 'text-slate-200';
  const rowBg = isTriggered ? 'bg-red-900/10' : 'hover:bg-slate-800/50';

  return (
    <div className={clsx("grid grid-cols-12 gap-4 p-4 border-b border-slate-800 transition-colors items-center", rowBg)}>
      <div className="col-span-3">
        <h4 className="font-bold text-slate-200 text-sm">{config.name}</h4>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 uppercase">{config.source}</span>
          <span className="text-[10px] text-slate-500 truncate">{config.id || 'Man'}</span>
        </div>
      </div>
      
      <div className="col-span-2">
        <div className={clsx("text-lg font-mono font-bold flex items-center gap-2", statusColor)}>
          {data?.value !== undefined && data.value !== null ? 
            <span>
                {data.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                <span className="text-xs ml-1 text-slate-500">{config.unit}</span>
            </span>
            : <span className="animate-pulse">Loading...</span>}
            {isTriggered && <AlertTriangle size={14} className="text-red-500" />}
        </div>
        <div className="text-xs text-slate-500">Target: {config.rule}</div>
      </div>

      <div className="col-span-2 h-10 w-full opacity-60">
           {chartData.length > 1 && (
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={chartData}>
                 <Line type="monotone" dataKey="val" stroke={isTriggered ? "#ef4444" : "#3b82f6"} strokeWidth={2} dot={false} isAnimationActive={false} />
               </LineChart>
             </ResponsiveContainer>
           )}
      </div>

      <div className="col-span-5 pl-4 border-l border-slate-800 flex items-center">
        <p className="text-xs text-slate-400 italic">"{config.why}"</p>
      </div>
    </div>
  );
};

// --- Main Application ---

export default function App() {
  const [activeTab, setActiveTab] = useState<'SHORT' | 'LONG' | 'CORE'>('SHORT');
  const [dataMap, setDataMap] = useState<Record<string, IndicatorData>>({});
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [lastRefreshed, setLastRefreshed] = useState<string>("");
  const [spxDrawdown, setSpxDrawdown] = useState<number>(-4.0); // Default/fallback

  // --- Manual/Mock Inputs State ---
  const [manualState, setManualState] = useState({
    margin_falling: true, 
    put_call: 0.55,
    aaii_bulls: 65,
    insider_ratio: 8,
    sp500_above_200dma: 20,
    gold_ath: true,
    gpr_index: 310,
    usd_reserve_share_change: -3,
    reset_event: false,
    pe_sp500: 32,
    real_assets_basket: 160
  });

  // --- Data Fetching Engine ---
  useEffect(() => {
    const fetchAll = async () => {
      const allIndicators = [...SHORT_TERM_INDICATORS, ...LONG_TERM_INDICATORS, ...CORE_INDICATORS];
      const newMap: Record<string, IndicatorData> = { ...dataMap };
      
      const requiredFredIds = ['FEDFUNDS', 'CPIAUCSL', 'T10Y2Y', 'DGS30', 'M2SL', 'BAMLH0A0HYM2', 'VIXCLS', 'SP500'];

      // 1. Fetch FRED Data
      const uniqueFredIds = Array.from(new Set(allIndicators.filter(i => i.source === 'FRED' || i.source === 'CALC').map(i => i.id).concat(requiredFredIds))).filter(Boolean) as string[];
      
      await Promise.all(uniqueFredIds.map(async (id) => {
          try {
              const raw = await fetchFredSeries(id);
              newMap[id] = raw;
          } catch (e) { console.error(`Failed ${id}`, e); }
      }));

      // 2. Fetch FMP Data
      const fmpIds = allIndicators.filter(i => i.source === 'FMP' && i.id).map(i => i.id!) || [];
      await Promise.all(fmpIds.map(async (id) => {
          try {
             const raw = await fetchFmpQuote(id);
             newMap[id] = raw;
          } catch (e) { console.error(e); }
      }));

      // 3. Process Logic for Transformations
      allIndicators.forEach(ind => {
        if (ind.source === 'FRED' && ind.id && newMap[ind.id]) {
            newMap[ind.id] = transformData(newMap[ind.id], ind.transform);
        }
      });

      // 4. Manual/Calc Injection & S&P Drawdown Logic
      const spx = newMap['SP500'];
      if (spx && spx.history.length > 0) {
          // Calculate All-Time High (or 52-week high from fetched set)
          const currentPrice = spx.value || 0;
          const maxPrice = Math.max(...spx.history);
          if (maxPrice > 0) {
              const dd = ((currentPrice - maxPrice) / maxPrice) * 100;
              setSpxDrawdown(dd);
          }
      }

      newMap['margin_debt'] = { value: manualState.margin_falling ? 3.6 : 3.0, prevValue: 3.5, history: [3.5, 3.6], lastUpdated: 'Manual', status: 'success' };
      
      const ff = newMap['FEDFUNDS']?.value;
      const cpiRaw = await fetchFredSeries('CPIAUCSL');
      const cpiYoy = transformData(cpiRaw, 'yoy').value;

      if (ff !== undefined && ff !== null && cpiYoy !== undefined && cpiYoy !== null) {
          const realFF = ff - cpiYoy;
          newMap['real_ff'] = { value: realFF, prevValue: 0, history: [realFF], lastUpdated: 'Calc', status: 'success' };
      }

      const y30 = newMap['DGS30']?.value;
      if (y30 !== undefined && cpiYoy !== undefined) {
          newMap['real_30y'] = { value: y30 - cpiYoy, prevValue: 0, history: [y30 - cpiYoy], lastUpdated: 'Calc', status: 'success' };
      }

      // Manual Projections
      newMap['put_call'] = { value: manualState.put_call, prevValue: 0, history: [0.6, 0.55], lastUpdated: 'Manual', status: 'success' };
      newMap['aaii_bulls'] = { value: manualState.aaii_bulls, prevValue: 0, history: [50, 65], lastUpdated: 'Manual', status: 'success' };
      newMap['pe_sp500'] = { value: manualState.pe_sp500, prevValue: 0, history: [30, 32], lastUpdated: 'Manual', status: 'success' };
      newMap['insider_ratio'] = { value: manualState.insider_ratio, prevValue: 0, history: [12, 8], lastUpdated: 'Manual', status: 'success' };
      newMap['sp500_above_200dma'] = { value: manualState.sp500_above_200dma, prevValue: 0, history: [30, 20], lastUpdated: 'Manual', status: 'success' };
      newMap['gold_ath'] = { value: manualState.gold_ath ? 1 : 0, prevValue: 0, history: [], lastUpdated: 'Manual', status: 'success' };
      newMap['gpr_index'] = { value: manualState.gpr_index, prevValue: 0, history: [200, 310], lastUpdated: 'Manual', status: 'success' };
      newMap['usd_reserve_share'] = { value: manualState.usd_reserve_share_change, prevValue: 0, history: [], lastUpdated: 'Manual', status: 'success' }; 
      newMap['real_assets_basket'] = { value: manualState.real_assets_basket, prevValue: 0, history: [140, 160], lastUpdated: 'Manual', status: 'success' };
      newMap['official_reset'] = { value: manualState.reset_event ? 1 : 0, prevValue: 0, history: [], lastUpdated: 'Manual', status: 'success' };
      newMap['usd_gold_power'] = { value: 0.09, prevValue: 0.11, history: [0.11, 0.09], lastUpdated: 'Calc', status: 'success' }; 

      setDataMap(newMap);
      setLastRefreshed(new Date().toLocaleTimeString());
    };

    fetchAll();
    const interval = setInterval(fetchAll, 60000 * 5); // 5 mins
    return () => clearInterval(interval);
  }, [refreshTrigger, manualState]);


  // --- Logic Engine: Evaluate Signals ---
  const evaluatedSignals = useMemo(() => {
    const checkSignal = (ind: IndicatorConfig): boolean => {
        const data = dataMap[ind.id || ''];
        if (!data || data.value === null) return false;

        const val = data.value;
        const rule = ind.rule.toLowerCase();

        try {
            if (rule.includes('falling') && ind.id === 'margin_debt') return val >= 3.5; 
            if (rule.includes('all majors')) return val === 1;
            if (rule.includes('yes')) return val === 1;
            
            const numThreshold = parseFloat(rule.match(/-?[\d\.]+/)?.[0] || '0');
            
            if (rule.startsWith('>=')) return val >= numThreshold;
            if (rule.startsWith('>')) return val > numThreshold;
            if (rule.startsWith('<=')) return val <= numThreshold;
            if (rule.startsWith('<')) return val < numThreshold;
            
            return false;
        } catch (e) { return false; }
    };

    let killScore = 0;
    const shortTermResults = SHORT_TERM_INDICATORS.map(ind => {
        const triggered = checkSignal(ind);
        if (triggered) killScore++;
        return { ...ind, triggered };
    });

    let darkScore = 0;
    const longTermResults = LONG_TERM_INDICATORS.map(ind => {
        const triggered = checkSignal(ind);
        if (triggered) darkScore++;
        return { ...ind, triggered };
    });

    const coreResults = CORE_INDICATORS.map(ind => {
        return { ...ind, triggered: false };
    });

    return { killScore, darkScore, shortTermResults, longTermResults, coreResults };
  }, [dataMap]);

  const activeList = useMemo(() => {
    if (activeTab === 'SHORT') return evaluatedSignals.shortTermResults;
    if (activeTab === 'LONG') return evaluatedSignals.longTermResults;
    return evaluatedSignals.coreResults;
  }, [activeTab, evaluatedSignals]);

  // Alert Logic
  const showKillAlert = evaluatedSignals.killScore >= 7 && spxDrawdown >= -8.0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-lg border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-900/20">
              <Activity className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent tracking-tight">ECON MIRROR</h1>
              <p className="text-[10px] text-slate-500 font-mono tracking-[0.2em] uppercase">Macro Cycle Dashboard</p>
            </div>
          </div>
          <div className="flex gap-4 text-xs font-mono items-center">
            <div className="px-3 py-1.5 rounded-full bg-slate-900 border border-slate-700 flex items-center gap-2 shadow-inner">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              LIVE
            </div>
            <div className="text-slate-600 hidden sm:block">Updated: {lastRefreshed}</div>
            <button onClick={() => setRefreshTrigger(p => p+1)} className="p-2 hover:bg-slate-800 rounded-full text-blue-400">
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        
        {/* Status Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
           {/* Regime Box */}
           <div className="md:col-span-3 bg-gradient-to-r from-slate-900 to-slate-950 border border-slate-800 rounded-xl p-6 relative overflow-hidden group">
              <div className="absolute -right-10 -top-10 opacity-5 group-hover:opacity-10 transition-opacity">
                <Database size={150} />
              </div>
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-2">
                    <span className="bg-slate-800 text-slate-300 text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">Status</span>
                    <h2 className="text-lg font-bold text-white">Current Regime Analysis</h2>
                </div>
                <p className="text-slate-400 max-w-3xl text-sm leading-relaxed">
                  You are navigating a <strong className="text-slate-200">Late-stage melt-up</strong> inside a <strong className="text-slate-200">late-stage debt super-cycle</strong>.
                  Protocol: Ride stocks with 20–30% cash + 30–40% gold/BTC permanent.
                </p>
              </div>
           </div>
           
           <StatusCard title="Short-Term Kill Score" count={evaluatedSignals.killScore} max={10} type="kill" isActive={activeTab === 'SHORT'} />
           <StatusCard title="Long-Term Dark Score" count={evaluatedSignals.darkScore} max={11} type="dark" isActive={activeTab === 'LONG'} />
           
           <div className="p-6 rounded-xl border border-slate-800 bg-slate-900 flex flex-col justify-center items-center relative overflow-hidden">
              <div className="text-slate-500 text-xs font-bold uppercase mb-3 tracking-widest">S&P 500 Drawdown</div>
              <div className={clsx("text-5xl font-bold tracking-tighter", spxDrawdown < -20 ? "text-red-500" : "text-green-400")}>
                {spxDrawdown.toFixed(2)}%
              </div>
              <div className="text-xs text-slate-600 mt-2">from 52-Week High</div>
              <div className={clsx("absolute inset-0 border-t-4", spxDrawdown < -20 ? "border-red-500/20" : "border-green-500/20")}></div>
           </div>
        </div>

        {/* --- CRITICAL ALERT BANNER --- */}
        {showKillAlert && activeTab === 'SHORT' && (
             <div className="mb-8 bg-red-900 text-white p-8 rounded-2xl border-4 border-red-600 shadow-[0_0_50px_rgba(220,38,38,0.4)] animate-pulse flex flex-col items-center text-center">
                 <div className="flex items-center gap-4 mb-4">
                     <ShieldAlert size={48} className="text-white"/>
                     <h2 className="text-4xl font-black tracking-tighter">SELL 80-90% STOCKS IMMEDIATELY</h2>
                     <ShieldAlert size={48} className="text-white"/>
                 </div>
                 <p className="text-xl font-bold text-red-200">KILL SCORE ≥ 7 AND S&P WITHIN -8% OF ATH</p>
                 <div className="mt-4 px-6 py-2 bg-white text-red-900 font-bold rounded-full uppercase tracking-widest text-sm">Execute Protocol A</div>
             </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800 mb-6 gap-8">
          <button onClick={() => setActiveTab('SHORT')} className={clsx("pb-4 font-bold text-sm transition-all border-b-2 flex items-center gap-2", activeTab === 'SHORT' ? "border-blue-500 text-blue-400" : "border-transparent text-slate-500 hover:text-slate-300")}>
            <Zap size={16}/> SHORT-TERM KILL
          </button>
          <button onClick={() => setActiveTab('LONG')} className={clsx("pb-4 font-bold text-sm transition-all border-b-2 flex items-center gap-2", activeTab === 'LONG' ? "border-red-500 text-red-400" : "border-transparent text-slate-500 hover:text-slate-300")}>
             <ShieldAlert size={16}/> LONG-TERM CYCLE
          </button>
          <button onClick={() => setActiveTab('CORE')} className={clsx("pb-4 font-bold text-sm transition-all border-b-2 flex items-center gap-2", activeTab === 'CORE' ? "border-green-500 text-green-400" : "border-transparent text-slate-500 hover:text-slate-300")}>
             <BarChart3 size={16}/> CORE 50+
          </button>
        </div>

        {/* Content Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
          <div className="grid grid-cols-12 gap-4 p-4 border-b border-slate-800 bg-slate-950/50 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            <div className="col-span-3">Indicator Name</div>
            <div className="col-span-2">Current / Trigger</div>
            <div className="col-span-2">Trend (Latest)</div>
            <div className="col-span-5 pl-4 border-l border-slate-800">Strategic Importance</div>
          </div>
          <div className="divide-y divide-slate-800/50">
            {activeList.map((ind, idx) => (
              <IndicatorRow key={idx} config={ind} data={dataMap[ind.id || '']} isTriggered={ind.triggered} />
            ))}
          </div>
        </div>

        {/* Action Box (Standard) */}
        {!showKillAlert && (
            <div className="mt-8 p-6 bg-slate-900/50 border border-slate-800 rounded-lg text-sm text-slate-400 flex items-start gap-4">
            <div className="p-2 bg-slate-800 rounded text-slate-300"><Activity size={20}/></div>
            <div>
                {activeTab === 'SHORT' && (
                    <div><h3 className="text-blue-400 font-bold mb-1 uppercase tracking-wider">Monitor Status</h3><p>Waiting for 7+ Kill Signals combined with Market Strength.</p></div>
                )}
                {activeTab === 'LONG' && (
                    <div><h3 className="text-red-400 font-bold mb-1 uppercase tracking-wider">Point of No Return Protocol</h3><p>If <strong className="text-white">DARK SCORE ≥ 8/11</strong> AND <strong className="text-white">2 "No-Return" Triggers</strong> → <span className="text-red-400 font-bold bg-red-900/20 px-1 rounded">EXIT SYSTEM.</span></p></div>
                )}
                {activeTab === 'CORE' && (
                    <div><h3 className="text-green-400 font-bold mb-1 uppercase tracking-wider">Core Monitor</h3><p>Monitor these 50+ signals for divergence.</p></div>
                )}
            </div>
            </div>
        )}
      </main>
    </div>
  );
}