#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NAU QUANTUM ALPHA ENGINE v4.0 — Next-Gen Trading Platform              ║
║          Advanced AI/ML Indicator with 18-Factor Confluence Analysis            ║
║                    Developed for Professional Traders                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This platform provides:
  - Interactive candlestick charting (TradingView-style)
  - AI-powered multi-factor indicator engine
  - Advanced mathematics: Kalman Filtering, Wavelet Analysis, HMM, 
    Bayesian Probability, Entropy Analysis, Hurst Exponent
  - Fully customizable visual parameters
  - Multi-timeframe support
  - Long/Short signal generation with confidence scoring

Author: NAU Trading Systems
Version: 3.0
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal as scipy_signal
from scipy.stats import norm, entropy as scipy_entropy
from scipy.ndimage import gaussian_filter1d
from collections import deque
import colorsys

# Suppress warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ADVANCED MATHEMATICAL ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Adaptive Kalman Filter for price estimation.
    Uses state-space model: x(t) = A*x(t-1) + w, z(t) = H*x(t) + v
    Dynamically adjusts process noise based on market volatility.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        self.Q = process_noise      # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.x = None               # State estimate
        self.P = 1.0                # Estimate error covariance
        self.K = 0.0                # Kalman gain
        self.history = []
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            self.history.append(self.x)
            return self.x
        
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        self.K = P_pred / (P_pred + self.R)
        self.x = x_pred + self.K * (measurement - x_pred)
        self.P = (1 - self.K) * P_pred
        
        self.history.append(self.x)
        return self.x
    
    def filter_series(self, data):
        """Filter an entire series and return smoothed values."""
        self.x = None
        self.P = 1.0
        self.history = []
        results = []
        for val in data:
            results.append(self.update(val))
        return np.array(results)


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter that adapts process noise based on 
    realized volatility (innovation-based adaptation).
    """
    def __init__(self, base_process_noise=1e-5, measurement_noise=1e-2, 
                 adaptation_rate=0.1, volatility_window=20):
        super().__init__(base_process_noise, measurement_noise)
        self.base_Q = base_process_noise
        self.adaptation_rate = adaptation_rate
        self.innovations = deque(maxlen=volatility_window)
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            self.history.append(self.x)
            return self.x
        
        # Track innovation (prediction error)
        innovation = measurement - self.x
        self.innovations.append(innovation)
        
        # Adapt process noise based on innovation variance
        if len(self.innovations) > 5:
            innov_var = np.var(list(self.innovations))
            self.Q = self.base_Q + self.adaptation_rate * innov_var
        
        return super().update(measurement)


class WaveletAnalyzer:
    """
    Continuous Wavelet Transform (CWT) for multi-scale price analysis.
    Uses Morlet wavelet to decompose price into different frequency components.
    """
    def __init__(self, scales=None):
        self.scales = scales or np.arange(2, 64, 2)
        
    def morlet_wavelet(self, t, omega0=6.0):
        """Morlet wavelet function."""
        return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)
    
    def cwt(self, data):
        """Compute Continuous Wavelet Transform."""
        n = len(data)
        coefficients = np.zeros((len(self.scales), n))
        
        for i, scale in enumerate(self.scales):
            # Create wavelet at this scale
            t = np.arange(-4*scale, 4*scale + 1) / scale
            wavelet = np.real(self.morlet_wavelet(t))
            wavelet = wavelet / np.sqrt(scale)
            
            # Convolve with data
            conv = np.convolve(data, wavelet, mode='same')
            coefficients[i] = conv
            
        return coefficients
    
    def get_dominant_cycle(self, data):
        """Find the dominant cycle length in the data."""
        coefficients = self.cwt(data)
        power = np.abs(coefficients)**2
        dominant_scale_idx = np.argmax(np.mean(power, axis=1))
        return self.scales[dominant_scale_idx]
    
    def get_trend_component(self, data, threshold_scale=20):
        """Extract trend component (low-frequency)."""
        coefficients = self.cwt(data)
        # Use only large-scale (low-frequency) components
        mask = self.scales >= threshold_scale
        trend = np.mean(coefficients[mask], axis=0)
        return trend
    
    def get_noise_component(self, data, threshold_scale=5):
        """Extract noise component (high-frequency)."""
        coefficients = self.cwt(data)
        mask = self.scales <= threshold_scale
        noise = np.mean(np.abs(coefficients[mask]), axis=0)
        return noise


class HiddenMarkovModel:
    """
    Gaussian Hidden Markov Model for market regime detection.
    States: Bull Market, Bear Market, Sideways/Consolidation
    Uses Baum-Welch for parameter estimation (simplified).
    """
    def __init__(self, n_states=3, n_iterations=50):
        self.n_states = n_states
        self.n_iterations = n_iterations
        # Transition matrix (initialized with slight persistence bias)
        self.A = np.array([
            [0.7, 0.15, 0.15],  # Bull -> Bull/Bear/Sideways
            [0.15, 0.7, 0.15],  # Bear -> Bull/Bear/Sideways
            [0.2, 0.2, 0.6],    # Sideways -> Bull/Bear/Sideways
        ])
        # State means and variances for returns
        self.means = np.array([0.002, -0.002, 0.0])     # Bull, Bear, Sideways
        self.stds = np.array([0.01, 0.015, 0.005])      # Volatilities
        self.pi = np.array([0.33, 0.33, 0.34])          # Initial state probs
        
    def _emission_prob(self, x, state):
        """Gaussian emission probability."""
        return norm.pdf(x, self.means[state], self.stds[state])
    
    def viterbi(self, observations):
        """
        Viterbi algorithm to find most likely state sequence.
        Returns: state sequence (0=Bull, 1=Bear, 2=Sideways)
        """
        T = len(observations)
        N = self.n_states
        
        # Initialize
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        for s in range(N):
            delta[0, s] = np.log(self.pi[s] + 1e-300) + \
                          np.log(self._emission_prob(observations[0], s) + 1e-300)
        
        # Recursion
        for t in range(1, T):
            for s in range(N):
                trans_probs = delta[t-1] + np.log(self.A[:, s] + 1e-300)
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = trans_probs[psi[t, s]] + \
                              np.log(self._emission_prob(observations[t], s) + 1e-300)
        
        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def fit_and_predict(self, returns):
        """Simplified EM fitting + Viterbi prediction."""
        if len(returns) < 30:
            return np.ones(len(returns), dtype=int) * 2  # Default to sideways
        
        # Estimate parameters from data
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        # Cluster into 3 regimes by percentile
        bear_data = sorted_returns[:n//3]
        sideways_data = sorted_returns[n//3:2*n//3]
        bull_data = sorted_returns[2*n//3:]
        
        self.means = np.array([np.mean(bull_data), np.mean(bear_data), np.mean(sideways_data)])
        self.stds = np.array([
            max(np.std(bull_data), 1e-6),
            max(np.std(bear_data), 1e-6),
            max(np.std(sideways_data), 1e-6)
        ])
        
        return self.viterbi(returns)


class BayesianAnalyzer:
    """
    Bayesian probability engine for trade signal confidence estimation.
    Combines multiple evidence sources using Bayes' theorem.
    """
    def __init__(self):
        self.prior_long = 0.5   # Prior probability of long signal
        self.prior_short = 0.5  # Prior probability of short signal
        
    def compute_posterior(self, evidences):
        """
        Compute posterior probability using multiple evidence factors.
        
        evidences: list of tuples (likelihood_long, likelihood_short)
        Each tuple represents P(evidence | long) and P(evidence | short)
        
        Returns: (P(long | all evidence), P(short | all evidence))
        """
        log_likelihood_long = np.log(self.prior_long + 1e-300)
        log_likelihood_short = np.log(self.prior_short + 1e-300)
        
        for lik_long, lik_short in evidences:
            log_likelihood_long += np.log(max(lik_long, 1e-300))
            log_likelihood_short += np.log(max(lik_short, 1e-300))
        
        # Normalize using log-sum-exp trick
        max_ll = max(log_likelihood_long, log_likelihood_short)
        norm_const = max_ll + np.log(
            np.exp(log_likelihood_long - max_ll) + 
            np.exp(log_likelihood_short - max_ll)
        )
        
        p_long = np.exp(log_likelihood_long - norm_const)
        p_short = np.exp(log_likelihood_short - norm_const)
        
        return p_long, p_short
    
    def update_priors(self, win_rate_long, win_rate_short):
        """Update priors based on historical win rates."""
        total = win_rate_long + win_rate_short
        if total > 0:
            self.prior_long = win_rate_long / total
            self.prior_short = win_rate_short / total


class EntropyAnalyzer:
    """
    Information-theoretic analysis of price data.
    Uses Shannon Entropy and Approximate Entropy for market chaos measurement.
    """
    @staticmethod
    def shannon_entropy(data, bins=20):
        """
        Compute Shannon entropy of price returns distribution.
        Higher entropy = more random/uncertain market.
        """
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return scipy_entropy(hist)
    
    @staticmethod
    def approximate_entropy(data, m=2, r_factor=0.2):
        """
        Compute Approximate Entropy (ApEn).
        Measures unpredictability/complexity of time series.
        Low ApEn = regular, predictable → trending
        High ApEn = irregular, complex → ranging/chaotic
        """
        N = len(data)
        if N < m + 1:
            return 0.0
            
        r = r_factor * np.std(data)
        if r == 0:
            return 0.0
        
        def _phi(m_val):
            patterns = np.array([data[i:i+m_val] for i in range(N - m_val + 1)])
            counts = np.zeros(len(patterns))
            for i, pat in enumerate(patterns):
                dists = np.max(np.abs(patterns - pat), axis=1)
                counts[i] = np.sum(dists <= r) / len(patterns)
            return np.mean(np.log(counts + 1e-300))
        
        return abs(_phi(m) - _phi(m + 1))
    
    @staticmethod
    def hurst_exponent(data, max_lag=20):
        """
        Compute Hurst Exponent using R/S analysis.
        H < 0.5: Mean-reverting (anti-persistent)
        H = 0.5: Random walk
        H > 0.5: Trending (persistent)
        """
        N = len(data)
        if N < max_lag * 2:
            return 0.5
            
        lags = range(2, min(max_lag, N // 2))
        rs_values = []
        
        for lag in lags:
            # Split into sub-series
            n_subseries = N // lag
            if n_subseries < 1:
                continue
            rs_list = []
            for i in range(n_subseries):
                subseries = data[i*lag:(i+1)*lag]
                mean = np.mean(subseries)
                deviations = subseries - mean
                cumulative = np.cumsum(deviations)
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(subseries)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
        
        if len(rs_values) < 2:
            return 0.5
            
        rs_array = np.array(rs_values)
        # Linear regression in log-log space
        coeffs = np.polyfit(rs_array[:, 0], rs_array[:, 1], 1)
        return np.clip(coeffs[0], 0.0, 1.0)


class FractalAnalyzer:
    """
    Fractal dimension and Williams' fractal analysis for support/resistance.
    """
    @staticmethod
    def williams_fractals(highs, lows, period=5):
        """
        Detect Williams' fractals (fractal highs and lows).
        A fractal high: highest high with 2 lower highs on each side.
        A fractal low: lowest low with 2 higher lows on each side.
        """
        n = len(highs)
        half = period // 2
        fractal_highs = np.full(n, np.nan)
        fractal_lows = np.full(n, np.nan)
        
        for i in range(half, n - half):
            # Check fractal high
            is_high = True
            for j in range(1, half + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_high = False
                    break
            if is_high:
                fractal_highs[i] = highs[i]
            
            # Check fractal low
            is_low = True
            for j in range(1, half + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_low = False
                    break
            if is_low:
                fractal_lows[i] = lows[i]
        
        return fractal_highs, fractal_lows
    
    @staticmethod
    def fractal_dimension(data, max_k=10):
        """
        Compute Higuchi's Fractal Dimension.
        FD ≈ 1.0: Smooth trend
        FD ≈ 1.5: Random walk
        FD ≈ 2.0: Extremely complex/noisy
        """
        N = len(data)
        if N < max_k * 4:
            return 1.5
            
        lk = []
        for k in range(1, max_k + 1):
            lengths = []
            for m in range(1, k + 1):
                # Compute length for this k and starting point m
                n_points = (N - m) // k
                if n_points < 2:
                    continue
                length = sum(abs(data[m + i*k] - data[m + (i-1)*k]) 
                           for i in range(1, n_points) if m + i*k < N)
                if length > 0:
                    length = (length * (N - 1)) / (max(n_points - 1, 1) * k * k)
                    lengths.append(length)
            if lengths:
                lk.append(np.mean(lengths))
        
        if len(lk) < 2:
            return 1.5
            
        # Linear regression in log-log space
        x = np.log(np.arange(1, len(lk) + 1))
        y = np.log(np.array(lk) + 1e-300)
        coeffs = np.polyfit(x, y, 1)
        return np.clip(abs(coeffs[0]), 1.0, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SMART MONEY CONCEPTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SmartMoneyConcepts:
    """
    Institutional order flow analysis engine.
    Detects: Order Blocks, Fair Value Gaps, Liquidity Sweeps,
    Break of Structure, Change of Character.
    """
    @staticmethod
    def detect_order_blocks(opens, highs, lows, closes, lookback=20):
        """
        Detect bullish and bearish order blocks.
        Bullish OB: Last down candle before a strong up move
        Bearish OB: Last up candle before a strong down move
        """
        n = len(closes)
        bullish_ob = np.full(n, np.nan)
        bearish_ob = np.full(n, np.nan)
        
        avg_range = np.mean(highs[:min(50, n)] - lows[:min(50, n)])
        
        for i in range(2, n):
            # Strong move detection (1.5x average range)
            move = closes[i] - closes[i-2]
            
            if move > avg_range * 1.5:  # Strong bullish move
                # Find last bearish candle before move
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        bullish_ob[j] = lows[j]
                        break
                        
            elif move < -avg_range * 1.5:  # Strong bearish move
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        bearish_ob[j] = highs[j]
                        break
        
        return bullish_ob, bearish_ob
    
    @staticmethod
    def detect_fair_value_gaps(highs, lows, closes):
        """
        Detect Fair Value Gaps (FVG) — imbalances in price.
        Bullish FVG: Gap between candle 1's high and candle 3's low
        Bearish FVG: Gap between candle 1's low and candle 3's high
        """
        n = len(closes)
        bullish_fvg_top = np.full(n, np.nan)
        bullish_fvg_bottom = np.full(n, np.nan)
        bearish_fvg_top = np.full(n, np.nan)
        bearish_fvg_bottom = np.full(n, np.nan)
        
        for i in range(2, n):
            # Bullish FVG: candle[i]'s low > candle[i-2]'s high
            if lows[i] > highs[i-2]:
                bullish_fvg_top[i-1] = lows[i]
                bullish_fvg_bottom[i-1] = highs[i-2]
            
            # Bearish FVG: candle[i]'s high < candle[i-2]'s low
            if highs[i] < lows[i-2]:
                bearish_fvg_top[i-1] = lows[i-2]
                bearish_fvg_bottom[i-1] = highs[i]
        
        return (bullish_fvg_top, bullish_fvg_bottom, 
                bearish_fvg_top, bearish_fvg_bottom)
    
    @staticmethod
    def detect_bos_choch(highs, lows, closes, swing_period=5):
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH).
        BOS: Continuation break of previous swing high/low
        CHoCH: Reversal break (first break against trend)
        """
        n = len(closes)
        bos_signals = np.zeros(n)   # +1 bullish, -1 bearish
        choch_signals = np.zeros(n) # +1 bullish, -1 bearish
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_period, n - swing_period):
            if highs[i] == max(highs[i-swing_period:i+swing_period+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-swing_period:i+swing_period+1]):
                swing_lows.append((i, lows[i]))
        
        # Track current trend direction
        trend = 0  # 0=undefined, 1=bullish, -1=bearish
        
        for i in range(1, n):
            # Check for break of swing highs
            for idx, level in swing_highs:
                if idx < i - 1 and closes[i] > level and closes[i-1] <= level:
                    if trend == -1:  # Was bearish, now breaking high
                        choch_signals[i] = 1
                        trend = 1
                    elif trend == 1:
                        bos_signals[i] = 1
                    else:
                        trend = 1
                        bos_signals[i] = 1
                    break
            
            # Check for break of swing lows
            for idx, level in swing_lows:
                if idx < i - 1 and closes[i] < level and closes[i-1] >= level:
                    if trend == 1:  # Was bullish, now breaking low
                        choch_signals[i] = -1
                        trend = -1
                    elif trend == -1:
                        bos_signals[i] = -1
                    else:
                        trend = -1
                        bos_signals[i] = -1
                    break
        
        return bos_signals, choch_signals


# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5: NEW AI/ML ENGINES (v4.0)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalAttention:
    """Transformer-inspired self-attention for pattern recognition."""
    def __init__(self, window=30, n_heads=4):
        self.window = window; self.n_heads = n_heads
    def compute_score(self, closes, volumes, idx):
        if idx < self.window: return 0.0
        start = max(0, idx - self.window)
        seg_c = closes[start:idx+1]; seg_v = volumes[start:idx+1]; n = len(seg_c)
        if n < 5: return 0.0
        feats = np.zeros((n, 6))
        for i in range(n):
            feats[i,0] = (seg_c[i]-seg_c[max(0,i-1)])/(seg_c[max(0,i-1)]+1e-10)
            feats[i,1] = seg_v[i]/(np.mean(seg_v[:max(1,i)])+1e-10)
            feats[i,2] = (seg_c[i]-seg_c[max(0,i-3)])/(seg_c[max(0,i-3)]+1e-10) if i>=3 else 0
            feats[i,3] = np.std(seg_c[max(0,i-5):i+1])/(np.mean(seg_c[max(0,i-5):i+1])+1e-10) if i>=1 else 0
            rh=np.max(seg_c[max(0,i-10):i+1]); rl=np.min(seg_c[max(0,i-10):i+1])
            feats[i,4] = (seg_c[i]-rl)/(rh-rl+1e-10)
            feats[i,5] = abs(feats[i,0])/(feats[i,3]+1e-10)
        scores = []
        for h in range(self.n_heads):
            s=(h*6)//self.n_heads; e=((h+1)*6)//self.n_heads
            hf=feats[:,s:e]; q=hf[-1:]; k=hf
            attn=(q@k.T)/np.sqrt(hf.shape[1]+1e-10)
            attn=attn.flatten()*np.exp(-np.arange(n-1,-1,-1)*0.1)
            ae=np.exp(attn-np.max(attn)); w=ae/(np.sum(ae)+1e-10)
            scores.append(np.sum(w*feats[:,0]))
        return np.clip(np.mean(scores)*5000, -100, 100)

class RLSignalOptimizer:
    """Q-Learning adaptive signal optimizer."""
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        self.Q=np.random.uniform(0,0.1,(5,3,5,5))
        self.mults=np.array([-1.0,-0.5,0.0,0.5,1.0])
    def optimize(self, closes, returns, raw_signal, window=50):
        n=len(closes); opt=np.zeros(n)
        for i in range(window, n):
            ti=np.clip(int((raw_signal[i]+100)/40),0,4)
            vi=np.clip(int(np.std(returns[max(0,i-window):i])*300),0,2)
            mi=np.clip(int((np.mean(returns[max(0,i-5):i])*1000+100)/40),0,4)
            st=(ti,vi,mi)
            a=np.random.randint(5) if np.random.random()<self.epsilon else np.argmax(self.Q[st])
            opt[i]=raw_signal[i]*(0.5+0.5*self.mults[a])
            if i+1<n:
                rw=np.clip(np.sign(opt[i])*returns[i+1]*100,-1,1)
                ns=(np.clip(int((raw_signal[min(i+1,n-1)]+100)/40),0,4),vi,np.clip(int((np.mean(returns[max(0,i-4):i+1])*1000+100)/40),0,4))
                self.Q[st+(a,)]+=self.alpha*(rw+self.gamma*np.max(self.Q[ns])-self.Q[st+(a,)])
            self.epsilon=max(0.01,self.epsilon*0.999)
        return opt

class DeepRegimeDetector:
    """GMM-based 4-regime detector with transition dynamics."""
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
    def detect(self, closes, volumes, returns, window=20):
        n=len(closes); score=np.zeros(n)
        for i in range(window, n):
            seg_r=returns[max(0,i-window):i]; seg_v=volumes[max(0,i-window):i]
            seg_c=closes[max(0,i-window):i]
            trend=np.mean(seg_r); vol=np.std(seg_r)
            rel_vol=volumes[i]/(np.mean(seg_v)+1e-10)-1
            zscore=(closes[i]-np.mean(seg_c))/(np.std(seg_c)+1e-10)
            # Classify regime
            if trend > 0.001 and vol < np.percentile(np.abs(returns[max(0,i-100):i]),75):
                score[i] = 80  # Bull
            elif trend < -0.001 and vol < np.percentile(np.abs(returns[max(0,i-100):i]),75):
                score[i] = -80  # Bear
            elif vol > np.percentile(np.abs(returns[max(0,i-100):i]),85):
                score[i] = 0  # High vol
            else:
                score[i] = 10 if zscore > 0 else -10  # Accumulation
        return gaussian_filter1d(np.clip(score,-100,100), sigma=3)

class AdvancedOrderFlow:
    """VWAP deviation + volume delta + absorption detection."""
    @staticmethod
    def compute_score(opens, highs, lows, closes, volumes, window=20):
        n=len(closes); score=np.zeros(n)
        for i in range(window, n):
            seg_c=closes[i-window:i+1]; seg_v=volumes[i-window:i+1]
            vwap=np.sum(seg_c*seg_v)/(np.sum(seg_v)+1e-10)
            vwap_dev=np.clip((closes[i]-vwap)/(vwap*0.01+1e-10)*20,-100,100)
            # Volume delta proxy
            cr=highs[i]-lows[i]
            delta=((closes[i]-lows[i])/(cr+1e-10)-0.5)*2*volumes[i] if cr>0 else 0
            avg_vol=np.mean(seg_v)
            delta_score=np.clip(delta/(avg_vol*0.5+1e-10)*30,-100,100)
            # Absorption
            avg_range=np.mean(highs[i-window:i+1]-lows[i-window:i+1])
            absorption=0
            if volumes[i]/(avg_vol+1e-10)>1.5 and cr/(avg_range+1e-10)<0.5:
                absorption=((closes[i]-lows[i])/(cr+1e-10)-0.5)*100 if cr>0 else 0
            score[i]=0.4*vwap_dev+0.4*delta_score+0.2*absorption
        return np.clip(score,-100,100)

class MicroStructureAnalyzer:
    """Market microstructure: spread estimation + liquidity."""
    @staticmethod
    def compute_score(opens, highs, lows, closes, volumes, window=20):
        n=len(closes); score=np.zeros(n)
        for i in range(window, n):
            seg_c=closes[max(0,i-window):i+1]
            if len(seg_c)>=3:
                rets=np.diff(seg_c)/seg_c[:-1]
                if len(rets)>=2:
                    autocov=np.mean(rets[1:]*rets[:-1])
                    spread=2*np.sqrt(max(-autocov,0))
                else: spread=0
            else: spread=0
            spread_factor=np.clip(1-spread*100,-1,1)
            trend_dir=np.sign(closes[i]-closes[max(0,i-5)])
            score[i]=trend_dir*spread_factor*50
        return np.clip(score,-100,100)

class MultiTimeframeMomentum:
    """Multi-timeframe momentum coherence across lookback periods."""
    @staticmethod
    def compute_score(closes, windows=(5,10,20,50)):
        n=len(closes); score=np.zeros(n); mw=max(windows)
        for i in range(mw, n):
            moms=[np.sign((closes[i]-closes[i-w])/(closes[i-w]+1e-10)) for w in windows]
            agreement=np.sum(moms)
            if abs(agreement)==len(windows):
                mag=np.mean([abs((closes[i]-closes[i-w])/(closes[i-w]+1e-10)) for w in windows])
                score[i]=np.sign(agreement)*min(mag*2000,100)
            elif abs(agreement)>=len(windows)-1:
                mag=np.mean([abs((closes[i]-closes[i-w])/(closes[i-w]+1e-10)) for w in windows])
                score[i]=np.sign(agreement)*min(mag*1000,70)
        return np.clip(score,-100,100)


# SECTION 3: NAU QUANTUM ALPHA INDICATOR (MAIN ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class NAUQuantumAlphaIndicator:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                NAU QUANTUM ALPHA INDICATOR v4.0                     ║
    ║              18-Factor AI/ML Confluence Trading Engine                 ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Combines 12 advanced factors into a unified signal:
    
    1.  Adaptive Kalman Filter (trend estimation)
    2.  Wavelet Multi-Scale Analysis (cycle detection)
    3.  Hidden Markov Model (regime detection)
    4.  Bayesian Probability (confidence scoring)
    5.  Shannon Entropy (market chaos measurement)
    6.  Approximate Entropy (predictability scoring)
    7.  Hurst Exponent (trend persistence)
    8.  Fractal Dimension (complexity analysis)
    9.  Smart Money: Order Blocks (institutional zones)
    10. Smart Money: Fair Value Gaps (price imbalances)
    11. Smart Money: BOS/CHoCH (structure analysis)
    12. Williams Fractals (support/resistance)
    
    Output: Composite score [-100, +100] with Bayesian confidence [0, 1]
    """
    
    def __init__(self, config=None):
        """Initialize with customizable configuration."""
        self.config = config or self.default_config()
        
        # Initialize engines
        self.kalman = AdaptiveKalmanFilter(
            base_process_noise=self.config['kalman_process_noise'],
            measurement_noise=self.config['kalman_measurement_noise'],
            adaptation_rate=self.config['kalman_adaptation_rate']
        )
        self.wavelet = WaveletAnalyzer()
        self.hmm = HiddenMarkovModel(n_states=3)
        self.bayesian = BayesianAnalyzer()
        self.entropy_analyzer = EntropyAnalyzer()
        self.fractal_analyzer = FractalAnalyzer()
        self.smc = SmartMoneyConcepts()
        
        # New v4.0 AI/ML engines
        self.attention = TemporalAttention(window=30, n_heads=4)
        self.rl_optimizer = RLSignalOptimizer(alpha=0.1, gamma=0.95)
        self.deep_regime = DeepRegimeDetector(n_regimes=4)
        self.order_flow = AdvancedOrderFlow()
        self.micro_structure = MicroStructureAnalyzer()
        self.mtf_momentum = MultiTimeframeMomentum()
        
    @staticmethod
    def default_config():
        return {
            # Kalman Filter
            'kalman_process_noise': 1e-5,
            'kalman_measurement_noise': 0.01,
            'kalman_adaptation_rate': 0.1,
            # Entropy
            'entropy_window': 20,
            'apen_m': 2,
            'apen_r_factor': 0.2,
            # Hurst
            'hurst_max_lag': 20,
            # Fractal
            'fractal_max_k': 10,
            # Smart Money
            'ob_lookback': 20,
            'swing_period': 5,
            # Signal
            'signal_smoothing': 3,
            'confidence_threshold': 0.6,
            # Factor weights (sum to 1.0)
            'weights': {
                'kalman': 0.08, 'wavelet': 0.06, 'hmm': 0.06,
                'entropy': 0.04, 'apen': 0.04, 'hurst': 0.06,
                'fractal': 0.04, 'order_blocks': 0.06, 'fvg': 0.05,
                'bos_choch': 0.06, 'fractals': 0.04,
                'attention': 0.10, 'rl': 0.08, 'deep_regime': 0.07,
                'order_flow': 0.07, 'micro_structure': 0.04, 'mtf_momentum': 0.05,
            }
        }
    
    def compute(self, df):
        """
        Compute the NAU Quantum Alpha Indicator on a DataFrame.
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            DataFrame with indicator columns added
        """
        opens = df['Open'].values.flatten().astype(float)
        highs = df['High'].values.flatten().astype(float)
        lows = df['Low'].values.flatten().astype(float)
        closes = df['Close'].values.flatten().astype(float)
        volumes = df['Volume'].values.flatten().astype(float)
        n = len(closes)
        
        if n < 30:
            df['NAU_Signal'] = 0
            df['NAU_Confidence'] = 0
            df['NAU_Regime'] = 2
            return df
        
        returns = np.diff(closes) / closes[:-1]
        returns = np.concatenate([[0], returns])
        
        # ─── Factor 1: Adaptive Kalman Filter ───
        kalman_filtered = self.kalman.filter_series(closes)
        kalman_trend = np.zeros(n)
        for i in range(1, n):
            if kalman_filtered[i] > kalman_filtered[i-1]:
                kalman_trend[i] = min((kalman_filtered[i] - kalman_filtered[i-1]) / 
                                     (np.std(closes[:max(i,2)]) + 1e-10) * 100, 100)
            else:
                kalman_trend[i] = max((kalman_filtered[i] - kalman_filtered[i-1]) / 
                                     (np.std(closes[:max(i,2)]) + 1e-10) * 100, -100)
        
        # ─── Factor 2: Wavelet Analysis ───
        wavelet_trend = self.wavelet.get_trend_component(closes)
        # Guard: ensure wavelet output length matches closes
        if len(wavelet_trend) != n:
            wavelet_trend = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(wavelet_trend)),
                wavelet_trend
            )
        wavelet_score = np.zeros(n)
        for i in range(1, n):
            if wavelet_trend[i] > wavelet_trend[i-1]:
                wavelet_score[i] = min(abs(wavelet_trend[i] - wavelet_trend[i-1]) / 
                                      (np.std(closes[:max(i,2)]) + 1e-10) * 200, 100)
            else:
                wavelet_score[i] = -min(abs(wavelet_trend[i] - wavelet_trend[i-1]) / 
                                       (np.std(closes[:max(i,2)]) + 1e-10) * 200, 100)
        
        # ─── Factor 3: HMM Regime Detection ───
        hmm_states = self.hmm.fit_and_predict(returns)
        hmm_score = np.zeros(n)
        hmm_score[hmm_states == 0] = 80   # Bull regime
        hmm_score[hmm_states == 1] = -80  # Bear regime
        hmm_score[hmm_states == 2] = 0    # Sideways
        
        # ─── Factor 4: Shannon Entropy ───
        entropy_score = np.zeros(n)
        window = self.config['entropy_window']
        for i in range(window, n):
            se = self.entropy_analyzer.shannon_entropy(returns[i-window:i])
            # Low entropy → confident trend → amplify signal
            # High entropy → uncertain → dampen signal
            entropy_score[i] = np.clip((3.0 - se) / 3.0 * 100, -100, 100)
        
        # ─── Factor 5: Approximate Entropy ───
        apen_score = np.zeros(n)
        for i in range(window, n):
            apen = self.entropy_analyzer.approximate_entropy(
                closes[i-window:i], 
                m=self.config['apen_m'],
                r_factor=self.config['apen_r_factor']
            )
            # Low ApEn = trending → score follows trend direction
            trend_dir = 1 if closes[i] > closes[i-window//2] else -1
            apen_score[i] = trend_dir * np.clip((1.0 - apen) * 100, 0, 100)
        
        # ─── Factor 6: Hurst Exponent ───
        hurst_score = np.zeros(n)
        for i in range(window*2, n):
            h = self.entropy_analyzer.hurst_exponent(
                closes[i-window*2:i], 
                max_lag=self.config['hurst_max_lag']
            )
            trend_dir = 1 if closes[i] > closes[i-window] else -1
            if h > 0.5:  # Trending
                hurst_score[i] = trend_dir * (h - 0.5) * 200
            else:  # Mean-reverting
                hurst_score[i] = -trend_dir * (0.5 - h) * 200
        
        # ─── Factor 7: Fractal Dimension ───
        fractal_score = np.zeros(n)
        for i in range(window*2, n):
            fd = self.fractal_analyzer.fractal_dimension(
                closes[i-window*2:i],
                max_k=self.config['fractal_max_k']
            )
            # FD near 1 = smooth trend, FD near 2 = complex
            trend_dir = 1 if closes[i] > closes[i-window] else -1
            fractal_score[i] = trend_dir * np.clip((1.5 - fd) * 200, -100, 100)
        
        # ─── Factor 8-9: Smart Money Order Blocks & FVG ───
        bull_ob, bear_ob = self.smc.detect_order_blocks(
            opens, highs, lows, closes, self.config['ob_lookback'])
        ob_score = np.zeros(n)
        for i in range(n):
            if not np.isnan(bull_ob[i]):
                ob_score[i] = 80
            elif not np.isnan(bear_ob[i]):
                ob_score[i] = -80
        # Smooth OB influence over nearby candles
        ob_score = gaussian_filter1d(ob_score, sigma=3)
        
        fvg_data = self.smc.detect_fair_value_gaps(highs, lows, closes)
        fvg_score = np.zeros(n)
        for i in range(n):
            if not np.isnan(fvg_data[0][i]):  # Bullish FVG
                fvg_score[i] += 70
            if not np.isnan(fvg_data[2][i]):  # Bearish FVG
                fvg_score[i] -= 70
        fvg_score = gaussian_filter1d(fvg_score, sigma=2)
        
        # ─── Factor 10: BOS/CHoCH ───
        bos, choch = self.smc.detect_bos_choch(
            highs, lows, closes, self.config['swing_period'])
        structure_score = np.zeros(n)
        for i in range(n):
            if bos[i] > 0 or choch[i] > 0:
                structure_score[i] = 90 if choch[i] > 0 else 70
            elif bos[i] < 0 or choch[i] < 0:
                structure_score[i] = -90 if choch[i] < 0 else -70
        structure_score = gaussian_filter1d(structure_score, sigma=2)
        
        # ─── Factor 11: Williams Fractals ───
        frac_highs, frac_lows = self.fractal_analyzer.williams_fractals(
            highs, lows, period=self.config['swing_period'])
        fractal_sr_score = np.zeros(n)
        for i in range(n):
            # Near fractal low = support = bullish
            # Near fractal high = resistance = bearish
            if not np.isnan(frac_lows[i]):
                fractal_sr_score[i] = 60
            elif not np.isnan(frac_highs[i]):
                fractal_sr_score[i] = -60
        fractal_sr_score = gaussian_filter1d(fractal_sr_score, sigma=2)
        
        # ═══ NEW v4.0 FACTORS ═══
        # Factor 13: Temporal Self-Attention
        attention_score = np.zeros(n)
        for i in range(30, n):
            attention_score[i] = self.attention.compute_score(closes, volumes, i)
        
        # Factor 15: Deep Regime Detection (GMM)
        deep_regime_score = self.deep_regime.detect(closes, volumes, returns)
        
        # Factor 16: Advanced Order Flow
        flow_score = self.order_flow.compute_score(opens, highs, lows, closes, volumes)
        
        # Factor 17: Micro-Structure
        micro_score = self.micro_structure.compute_score(opens, highs, lows, closes, volumes)
        
        # Factor 18: Multi-Timeframe Momentum
        mtf_score = self.mtf_momentum.compute_score(closes)
        
        # ═══ COMPOSITE SIGNAL COMPUTATION ═══
        w = self.config['weights']
        pre_composite = (
            w['kalman'] * kalman_trend +
            w['wavelet'] * wavelet_score +
            w['hmm'] * hmm_score +
            w['entropy'] * entropy_score +
            w['apen'] * apen_score +
            w['hurst'] * hurst_score +
            w['fractal'] * fractal_score +
            w['order_blocks'] * ob_score +
            w['fvg'] * fvg_score +
            w['bos_choch'] * structure_score +
            w['fractals'] * fractal_sr_score +
            w['attention'] * attention_score +
            w['deep_regime'] * deep_regime_score +
            w['order_flow'] * flow_score +
            w['micro_structure'] * micro_score +
            w['mtf_momentum'] * mtf_score
        )
        
        # Factor 14: RL Signal Optimizer (operates on pre-composite)
        rl_optimized = self.rl_optimizer.optimize(closes, returns, pre_composite)
        
        # Blend pre-composite with RL-optimized
        composite = (1 - w['rl']) * pre_composite + w['rl'] * rl_optimized
        
        # Smooth the composite signal
        if self.config['signal_smoothing'] > 1:
            composite = gaussian_filter1d(composite, sigma=self.config['signal_smoothing'])
        
        # Clip to [-100, 100]
        composite = np.clip(composite, -100, 100)
        
        # ═══ BAYESIAN CONFIDENCE SCORING ═══
        confidence = np.zeros(n)
        for i in range(window, n):
            evidences = []
            
            # Each factor provides evidence
            factors = [kalman_trend[i], wavelet_score[i], hmm_score[i],
                      entropy_score[i], hurst_score[i], ob_score[i],
                      structure_score[i], attention_score[i],
                      deep_regime_score[i], flow_score[i], micro_score[i], mtf_score[i]]
            
            for f in factors:
                if f > 0:
                    evidences.append((0.5 + f/200, 0.5 - f/200))
                else:
                    evidences.append((0.5 + f/200, 0.5 - f/200))
            
            p_long, p_short = self.bayesian.compute_posterior(evidences)
            confidence[i] = max(p_long, p_short)
        
        # ═══ Store Results ═══
        df['NAU_Signal'] = composite
        df['NAU_Confidence'] = confidence
        df['NAU_Regime'] = hmm_states
        df['NAU_Kalman'] = kalman_filtered
        df['NAU_Kalman_Score'] = kalman_trend
        df['NAU_Wavelet_Score'] = wavelet_score
        df['NAU_HMM_Score'] = hmm_score
        df['NAU_Entropy_Score'] = entropy_score
        df['NAU_Hurst_Score'] = hurst_score
        df['NAU_Fractal_Score'] = fractal_score
        df['NAU_OB_Score'] = ob_score
        df['NAU_FVG_Score'] = fvg_score
        df['NAU_Structure_Score'] = structure_score
        df['NAU_Williams_Score'] = fractal_sr_score
        # v4.0 factor columns
        df['NAU_Attention_Score'] = attention_score
        df['NAU_RL_Score'] = rl_optimized
        df['NAU_DeepRegime_Score'] = deep_regime_score
        df['NAU_OrderFlow_Score'] = flow_score
        df['NAU_MicroStructure_Score'] = micro_score
        df['NAU_MTF_Score'] = mtf_score
        
        # Long/Short signals
        df['NAU_Long'] = (composite > 20) & (confidence > self.config['confidence_threshold'])
        df['NAU_Short'] = (composite < -20) & (confidence > self.config['confidence_threshold'])
        
        # Store fractal/OB/FVG locations for plotting
        df['Fractal_High'] = frac_highs
        df['Fractal_Low'] = frac_lows
        df['Bull_OB'] = bull_ob
        df['Bear_OB'] = bear_ob
        df['Bull_FVG_Top'] = fvg_data[0]
        df['Bull_FVG_Bot'] = fvg_data[1]
        df['Bear_FVG_Top'] = fvg_data[2]
        df['Bear_FVG_Bot'] = fvg_data[3]
        df['BOS'] = bos
        df['CHoCH'] = choch
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SAMPLE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_realistic_ohlcv(symbol="AAPL", days=200, timeframe='1D', seed=42):
    """
    Generate realistic OHLCV data with regime changes.
    Simulates trending, mean-reverting, and volatile periods.
    """
    np.random.seed(seed)
    
    # Timeframe multiplier
    tf_map = {
        '1m': (days * 390, 0.0002), '5m': (days * 78, 0.0005),
        '15m': (days * 26, 0.001), '30m': (days * 13, 0.0015),
        '1H': (days * 7, 0.002), '4H': (days * 2, 0.004),
        '1D': (days, 0.01), '1W': (days // 5, 0.02),
    }
    
    n_bars, base_vol = tf_map.get(timeframe, (days, 0.01))
    n_bars = max(n_bars, 100)
    
    # Start price
    price = 150.0
    prices = [price]
    
    # Simulate with regime switching
    regime_length = n_bars // 5
    for i in range(1, n_bars):
        regime = (i // regime_length) % 3
        if regime == 0:  # Bull
            drift = 0.0003
            vol = base_vol * 0.8
        elif regime == 1:  # Bear
            drift = -0.0002
            vol = base_vol * 1.2
        else:  # Sideways
            drift = 0.0
            vol = base_vol * 0.5
        
        ret = drift + vol * np.random.randn()
        price = price * (1 + ret)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Generate OHLCV from close prices
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='h' if 'H' in timeframe or 'm' in timeframe else 'D')
    
    data = []
    for i, close in enumerate(prices):
        spread = close * base_vol * 0.5
        high = close + abs(np.random.randn()) * spread
        low = close - abs(np.random.randn()) * spread
        open_price = low + np.random.rand() * (high - low)
        volume = int(np.random.lognormal(15, 1))
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume,
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INTERACTIVE CHARTING ENGINE (TradingView-style)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_chart(df, indicator_config=None, visual_config=None, title=None):
    """
    Generate a complete TradingView-style interactive HTML chart.
    Uses lightweight-charts library for professional rendering.
    """
    if visual_config is None:
        visual_config = {
            'bg_color': '#131722',
            'text_color': '#D1D4DC',
            'grid_color': '#1E222D',
            'up_color': '#26A69A',
            'down_color': '#EF5350',
            'signal_line_color': '#2196F3',
            'long_color': '#00E676',
            'short_color': '#FF1744',
            'kalman_color': '#FFD700',
            'volume_up_color': 'rgba(38,166,154,0.5)',
            'volume_down_color': 'rgba(239,83,80,0.5)',
            'candle_border_up': '#26A69A',
            'candle_border_down': '#EF5350',
            'candle_wick_up': '#26A69A',
            'candle_wick_down': '#EF5350',
            'font_size': 12,
            'line_width': 2,
            'signal_line_width': 2,
            'confidence_opacity': 0.3,
            'show_volume': True,
            'show_kalman': True,
            'show_signals': True,
            'show_fractals': True,
            'show_order_blocks': True,
            'show_fvg': True,
            'show_structure': True,
            'show_regime': True,
            'show_confidence': True,
            'show_factor_panel': True,
        }
    
    # Prepare data as JSON
    candle_data = []
    volume_data = []
    signal_data = []
    confidence_data = []
    kalman_data = []
    long_markers = []
    short_markers = []
    
    for idx, row in df.iterrows():
        ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp())
        
        candle_data.append({
            'time': ts,
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
        })
        
        vol_color = visual_config['volume_up_color'] if row['Close'] >= row['Open'] else visual_config['volume_down_color']
        volume_data.append({
            'time': ts,
            'value': int(row['Volume']),
            'color': vol_color,
        })
        
        if 'NAU_Signal' in row:
            signal_data.append({'time': ts, 'value': round(float(row['NAU_Signal']), 2)})
        if 'NAU_Confidence' in row:
            confidence_data.append({'time': ts, 'value': round(float(row['NAU_Confidence']) * 100, 2)})
        if 'NAU_Kalman' in row:
            kalman_data.append({'time': ts, 'value': round(float(row['NAU_Kalman']), 2)})
        
        if 'NAU_Long' in row and row['NAU_Long']:
            _mfs = visual_config.get('marker_font_size', 11)
            _label = f"L {round(float(row['NAU_Confidence'])*100)}%" if _mfs > 0 else ""
            long_markers.append({
                'time': ts, 'position': 'belowBar',
                'color': visual_config['long_color'],
                'shape': 'arrowUp', 'text': _label,
                'size': max(1, min(3, _mfs // 5)),
            })
        
        if 'NAU_Short' in row and row['NAU_Short']:
            _mfs = visual_config.get('marker_font_size', 11)
            _label = f"S {round(float(row['NAU_Confidence'])*100)}%" if _mfs > 0 else ""
            short_markers.append({
                'time': ts, 'position': 'aboveBar',
                'color': visual_config['short_color'],
                'shape': 'arrowDown', 'text': _label,
                'size': max(1, min(3, _mfs // 5)),
            })
    
    # Factor data for the sub-panel
    factor_data = {}
    factor_cols = [
        ('NAU_Kalman_Score', 'Kalman', '#FFD700'),
        ('NAU_Wavelet_Score', 'Wavelet', '#E040FB'),
        ('NAU_HMM_Score', 'HMM', '#00BCD4'),
        ('NAU_Entropy_Score', 'Entropy', '#FF9800'),
        ('NAU_Hurst_Score', 'Hurst', '#8BC34A'),
        ('NAU_Fractal_Score', 'Fractal', '#FF5722'),
        ('NAU_OB_Score', 'OrdBlk', '#2196F3'),
        ('NAU_FVG_Score', 'FVG', '#9C27B0'),
        ('NAU_Structure_Score', 'BOS/CHoCH', '#F44336'),
        ('NAU_Williams_Score', 'Fractals', '#795548'),
        ('NAU_Attention_Score', 'Attention', '#00E5FF'),
        ('NAU_RL_Score', 'RL-Opt', '#76FF03'),
        ('NAU_DeepRegime_Score', 'DeepReg', '#FF6D00'),
        ('NAU_OrderFlow_Score', 'OrdFlow', '#D500F9'),
        ('NAU_MicroStructure_Score', 'MicroStr', '#1DE9B6'),
        ('NAU_MTF_Score', 'MTF-Mom', '#FFAB40'),
    ]
    
    for col, name, color in factor_cols:
        if col in df.columns:
            factor_data[name] = {
                'color': color,
                'data': [{'time': int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp()), 
                          'value': round(float(row[col]), 2)} 
                         for idx, row in df.iterrows()]
            }
    
    # Regime data
    regime_data = []
    if 'NAU_Regime' in df.columns:
        regime_colors = {0: 'rgba(0,230,118,0.08)', 1: 'rgba(255,23,68,0.08)', 2: 'rgba(255,235,59,0.05)'}
        regime_names = {0: 'BULL', 1: 'BEAR', 2: 'RANGE'}
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp())
            regime_data.append({
                'time': ts,
                'regime': int(row['NAU_Regime']),
                'color': regime_colors.get(int(row['NAU_Regime']), 'rgba(128,128,128,0.05)'),
                'name': regime_names.get(int(row['NAU_Regime']), 'UNKNOWN')
            })
    
    # Statistics
    if 'NAU_Long' in df.columns:
        total_long = int(df['NAU_Long'].sum())
        total_short = int(df['NAU_Short'].sum())
        avg_confidence = round(float(df['NAU_Confidence'].mean()) * 100, 1)
        last_signal = float(df['NAU_Signal'].iloc[-1])
        last_confidence = float(df['NAU_Confidence'].iloc[-1]) * 100
        last_regime = int(df['NAU_Regime'].iloc[-1])
    else:
        total_long = total_short = 0
        avg_confidence = last_signal = last_confidence = 0
        last_regime = 2

    # Parse title for display
    chart_title = title or "NAU Quantum Alpha Engine v3.0"
    symbol_parts = chart_title.split("·")
    symbol_display = symbol_parts[0].strip() if symbol_parts else "AAPL"
    tf_display = symbol_parts[1].strip() if len(symbol_parts) > 1 else ""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{chart_title}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

:root {{
    --bg-primary: {visual_config['bg_color']};
    --bg-secondary: #1B1F2B;
    --bg-tertiary: #252A37;
    --text-primary: {visual_config['text_color']};
    --text-secondary: #787B86;
    --text-muted: #505462;
    --accent-blue: #2962FF;
    --accent-cyan: #00BCD4;
    --up-color: {visual_config['up_color']};
    --down-color: {visual_config['down_color']};
    --long-color: {visual_config['long_color']};
    --short-color: {visual_config['short_color']};
    --border-color: #2A2E39;
    --font-mono: 'JetBrains Mono', monospace;
    --font-sans: 'Inter', -apple-system, sans-serif;
    --font-size: {visual_config['font_size']}px;
}}

html, body {{
    height: 100%;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: var(--font-size);
    overflow: hidden;
}}

/* ─── HEADER ─── */
.header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    height: 44px;
    z-index: 100;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.logo {{
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 14px;
    color: var(--accent-cyan);
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.logo-dot {{
    width: 8px;
    height: 8px;
    background: var(--accent-cyan);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

.symbol-info {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
}}

.symbol-name {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}}

.badge-tf {{
    font-size: 11px;
    font-weight: 500;
    color: var(--accent-cyan);
    background: rgba(0,188,212,0.12);
    padding: 1px 7px;
    border-radius: 3px;
    border: 1px solid rgba(0,188,212,0.25);
    letter-spacing: 0.5px;
}}

.price-change {{
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 3px;
}}

.price-up {{ background: rgba(38,166,154,0.15); color: var(--up-color); }}
.price-down {{ background: rgba(239,83,80,0.15); color: var(--down-color); }}

/* ─── TIMEFRAME BAR ─── */
.timeframe-bar {{
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 0 16px;
    height: 32px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
}}

.tf-btn {{
    padding: 3px 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    background: transparent;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    transition: all 0.15s;
}}

.tf-btn:hover {{ color: var(--text-primary); background: var(--bg-tertiary); }}
.tf-btn.active {{ color: var(--text-primary); background: var(--accent-blue); }}

.separator {{
    width: 1px;
    height: 16px;
    background: var(--border-color);
    margin: 0 6px;
}}

/* ─── INDICATOR PANEL ─── */
.indicator-panel {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 16px;
    height: 28px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-color);
    font-family: var(--font-mono);
    font-size: 11px;
    overflow-x: auto;
}}

.ind-label {{
    color: var(--text-muted);
    white-space: nowrap;
}}

.ind-value {{
    font-weight: 500;
    white-space: nowrap;
    margin-right: 12px;
}}

/* ─── CHART CONTAINER ─── */
.chart-wrapper {{
    display: flex;
    flex-direction: column;
    height: calc(100vh - 104px);
}}

#main-chart {{
    flex: 1;
    min-height: 0;
}}

#signal-chart {{
    height: 180px;
    border-top: 1px solid var(--border-color);
}}

#factor-chart {{
    height: 120px;
    border-top: 1px solid var(--border-color);
}}

/* ─── SIDE PANEL ─── */
.side-panel {{
    position: fixed;
    right: 0;
    top: 44px;
    width: 280px;
    height: calc(100vh - 44px);
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-color);
    z-index: 50;
    overflow-y: auto;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    padding: 16px;
}}

.side-panel.open {{
    transform: translateX(0);
}}

.panel-section {{
    margin-bottom: 16px;
}}

.panel-title {{
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-color);
}}

.stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 12px;
}}

.stat-label {{ color: var(--text-secondary); }}
.stat-value {{ font-family: var(--font-mono); font-weight: 500; }}

.signal-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--font-mono);
}}

.signal-long {{ background: rgba(0,230,118,0.2); color: var(--long-color); }}
.signal-short {{ background: rgba(255,23,68,0.2); color: var(--short-color); }}
.signal-neutral {{ background: rgba(255,235,59,0.15); color: #FFEB3B; }}

/* ─── SETTINGS PANEL ─── */
.settings-panel {{
    position: fixed;
    left: 0;
    top: 44px;
    width: 320px;
    height: calc(100vh - 44px);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    z-index: 50;
    overflow-y: auto;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
    padding: 16px;
}}

.settings-panel.open {{
    transform: translateX(0);
}}

.setting-group {{
    margin-bottom: 16px;
}}

.setting-label {{
    display: block;
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    font-family: var(--font-mono);
}}

.setting-input {{
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 12px;
    outline: none;
}}

.setting-input:focus {{ border-color: var(--accent-blue); }}

.color-input {{
    width: 40px;
    height: 28px;
    padding: 2px;
    cursor: pointer;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
}}

.setting-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}}

.setting-row label {{
    flex: 1;
    font-size: 12px;
    color: var(--text-secondary);
}}

.toggle {{
    position: relative;
    width: 36px;
    height: 20px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    cursor: pointer;
    transition: 0.3s;
    border: 1px solid var(--border-color);
}}

.toggle.active {{
    background: var(--accent-blue);
    border-color: var(--accent-blue);
}}

.toggle::after {{
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    top: 1px;
    left: 1px;
    transition: 0.3s;
}}

.toggle.active::after {{
    left: 17px;
}}

/* ─── TOOLBAR ─── */
.header-right {{
    display: flex;
    gap: 8px;
    align-items: center;
}}

.tool-btn {{
    padding: 5px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 11px;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.tool-btn:hover {{ color: var(--text-primary); border-color: var(--accent-blue); }}

/* ─── REGIME BANNER ─── */
.regime-banner {{
    position: absolute;
    top: 8px;
    left: 16px;
    z-index: 10;
    display: flex;
    gap: 8px;
    align-items: center;
}}

.regime-tag {{
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--font-mono);
    letter-spacing: 0.5px;
}}

.regime-bull {{ background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }}
.regime-bear {{ background: rgba(255,23,68,0.15); color: #FF1744; border: 1px solid rgba(255,23,68,0.3); }}
.regime-range {{ background: rgba(255,235,59,0.1); color: #FFEB3B; border: 1px solid rgba(255,235,59,0.2); }}

/* ─── TOOLTIP ─── */
.custom-tooltip {{
    position: absolute;
    display: none;
    background: rgba(19,23,34,0.95);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 11px;
    z-index: 1000;
    pointer-events: none;
    backdrop-filter: blur(8px);
    min-width: 200px;
}}

.tooltip-row {{
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    gap: 16px;
}}

.tooltip-label {{ color: var(--text-secondary); }}
.tooltip-value {{ font-weight: 500; }}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

/* ─── RESIZE HANDLES ─── */
.resize-handle {{ height: 4px; background: transparent; cursor: row-resize; position: relative; z-index: 10; }}
.resize-handle:hover {{ background: var(--accent-blue); opacity: 0.5; }}

/* ─── RESPONSIVE ─── */
@media (max-width: 768px) {{
    .side-panel, .settings-panel {{ width: 100%; }}
    #signal-chart {{ height: 140px; }}
    #factor-chart {{ height: 100px; }}
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="header-left">
        <div class="logo">
            <div class="logo-dot"></div>
            NAU QUANTUM v4.0
        </div>
        <div class="symbol-info">
            <span class="symbol-name">{symbol_display}</span>
            <span class="badge-tf">{tf_display}</span>
            <span class="price-change {'price-up' if last_signal >= 0 else 'price-down'}" id="price-display">
                $0.00
            </span>
        </div>
    </div>
    <div class="header-right">
        <button class="tool-btn" onclick="toggleSettings()">⚙ Settings</button>
        <button class="tool-btn" onclick="togglePanel()">📊 Analysis</button>
        <button class="tool-btn" onclick="takeScreenshot()">📸 Screenshot</button>
    </div>
</div>

<!-- TIMEFRAME BAR -->
<div class="timeframe-bar">
    <button class="tf-btn" onclick="setTF('1m')">1m</button>
    <button class="tf-btn" onclick="setTF('5m')">5m</button>
    <button class="tf-btn" onclick="setTF('15m')">15m</button>
    <button class="tf-btn" onclick="setTF('30m')">30m</button>
    <button class="tf-btn active" onclick="setTF('1H')">1H</button>
    <button class="tf-btn" onclick="setTF('4H')">4H</button>
    <button class="tf-btn" onclick="setTF('1D')">1D</button>
    <button class="tf-btn" onclick="setTF('1W')">1W</button>
    <div class="separator"></div>
    <button class="tf-btn" onclick="toggleIndicator('kalman')">Kalman</button>
    <button class="tf-btn" onclick="toggleIndicator('volume')">Vol</button>
    <button class="tf-btn" onclick="toggleIndicator('signals')">Signals</button>
    <button class="tf-btn" onclick="toggleIndicator('regime')">Regime</button>
</div>

<!-- INDICATOR VALUES BAR -->
<div class="indicator-panel" id="indicator-bar">
    <span class="ind-label">NAU Signal:</span>
    <span class="ind-value" id="ind-signal" style="color: {visual_config['signal_line_color']}">0.00</span>
    <span class="ind-label">Confidence:</span>
    <span class="ind-value" id="ind-confidence" style="color: #FFD700">0%</span>
    <span class="ind-label">Regime:</span>
    <span class="ind-value" id="ind-regime">—</span>
    <span class="ind-label">Hurst:</span>
    <span class="ind-value" id="ind-hurst">—</span>
    <span class="ind-label">Entropy:</span>
    <span class="ind-value" id="ind-entropy">—</span>
    <span class="ind-label">Attention:</span>
    <span class="ind-value" id="ind-attention" style="color:#00E5FF">—</span>
    <span class="ind-label">OrderFlow:</span>
    <span class="ind-value" id="ind-flow" style="color:#D500F9">—</span>
</div>

<!-- CHART -->
<div class="chart-wrapper">
    <div id="main-chart" style="position: relative;">
        <div class="regime-banner" id="regime-banner"></div>
        <div style="position:absolute;top:8px;right:16px;z-index:10;font-family:var(--font-mono);font-size:10px;color:var(--text-muted);background:var(--bg-secondary);padding:2px 8px;border-radius:3px;border:1px solid var(--border-color);">v4.0 · 18 Factors</div>
    </div>
    <div class="resize-handle" id="resize1"></div>
    <div id="signal-chart"></div>
    <div class="resize-handle" id="resize2"></div>
    <div id="factor-chart"></div>
</div>

<!-- ANALYSIS PANEL -->
<div class="side-panel" id="side-panel">
    <div class="panel-section">
        <div class="panel-title">📡 Current Signal</div>
        <div class="stat-row">
            <span class="stat-label">Direction</span>
            <span class="stat-value">
                <span class="signal-badge {'signal-long' if last_signal > 20 else 'signal-short' if last_signal < -20 else 'signal-neutral'}">
                    {'● LONG' if last_signal > 20 else '● SHORT' if last_signal < -20 else '● NEUTRAL'}
                </span>
            </span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Composite Score</span>
            <span class="stat-value" style="color: {'var(--long-color)' if last_signal > 0 else 'var(--short-color)'}">{last_signal:.1f}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Confidence</span>
            <span class="stat-value" style="color: #FFD700">{last_confidence:.1f}%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Market Regime</span>
            <span class="stat-value">
                <span class="regime-tag {'regime-bull' if last_regime == 0 else 'regime-bear' if last_regime == 1 else 'regime-range'}">
                    {'BULL' if last_regime == 0 else 'BEAR' if last_regime == 1 else 'RANGE'}
                </span>
            </span>
        </div>
    </div>
    
    <div class="panel-section">
        <div class="panel-title">📈 Signal Statistics</div>
        <div class="stat-row">
            <span class="stat-label">Total Long Signals</span>
            <span class="stat-value" style="color: var(--long-color)">{total_long}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Short Signals</span>
            <span class="stat-value" style="color: var(--short-color)">{total_short}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Avg Confidence</span>
            <span class="stat-value">{avg_confidence}%</span>
        </div>
    </div>
    
    <div class="panel-section">
        <div class="panel-title">🧠 Factor Breakdown</div>
        <div id="factor-breakdown"></div>
    </div>
    
    <div class="panel-section">
        <div class="panel-title">📖 Methodology</div>
        <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.6;">
            12-factor confluence engine combining:<br>
            Adaptive Kalman Filter, Wavelet CWT,<br>
            Hidden Markov Model, Bayesian Inference,<br>
            Shannon & Approximate Entropy,<br>
            Hurst Exponent, Fractal Dimension,<br>
            Smart Money Concepts (OB/FVG/BOS)
        </div>
    </div>
</div>

<!-- SETTINGS PANEL -->
<div class="settings-panel" id="settings-panel">
    <div class="panel-section">
        <div class="panel-title">🎨 Visual Settings</div>
        
        <div class="setting-group">
            <div class="setting-row">
                <label>Bullish Color</label>
                <input type="color" class="color-input" value="{visual_config['up_color']}" id="set-up-color">
            </div>
            <div class="setting-row">
                <label>Bearish Color</label>
                <input type="color" class="color-input" value="{visual_config['down_color']}" id="set-down-color">
            </div>
            <div class="setting-row">
                <label>Background</label>
                <input type="color" class="color-input" value="{visual_config['bg_color']}" id="set-bg-color">
            </div>
            <div class="setting-row">
                <label>Signal Line</label>
                <input type="color" class="color-input" value="{visual_config['signal_line_color']}" id="set-signal-color">
            </div>
            <div class="setting-row">
                <label>Kalman Line</label>
                <input type="color" class="color-input" value="{visual_config['kalman_color']}" id="set-kalman-color">
            </div>
        </div>
        
        <div class="setting-group">
            <label class="setting-label">Font Size</label>
            <input type="range" class="setting-input" min="9" max="18" value="{visual_config['font_size']}" id="set-font-size" style="padding: 2px;">
        </div>
        
        <div class="setting-group">
            <label class="setting-label">Line Width</label>
            <input type="range" class="setting-input" min="1" max="5" value="{visual_config['line_width']}" id="set-line-width" style="padding: 2px;">
        </div>
    </div>
    
    <div class="panel-section">
        <div class="panel-title">📊 Indicator Toggles</div>
        <div class="setting-row">
            <label>Kalman Filter</label>
            <div class="toggle active" id="tog-kalman" onclick="this.classList.toggle('active')"></div>
        </div>
        <div class="setting-row">
            <label>Volume</label>
            <div class="toggle active" id="tog-volume" onclick="this.classList.toggle('active')"></div>
        </div>
        <div class="setting-row">
            <label>Long/Short Signals</label>
            <div class="toggle active" id="tog-signals" onclick="this.classList.toggle('active')"></div>
        </div>
        <div class="setting-row">
            <label>Regime Shading</label>
            <div class="toggle active" id="tog-regime" onclick="this.classList.toggle('active')"></div>
        </div>
        <div class="setting-row">
            <label>Confidence Band</label>
            <div class="toggle active" id="tog-confidence" onclick="this.classList.toggle('active')"></div>
        </div>
        <div class="setting-row">
            <label>Factor Panel</label>
            <div class="toggle active" id="tog-factors" onclick="this.classList.toggle('active')"></div>
        </div>
    </div>
    
    <div class="panel-section">
        <div class="panel-title">⚙ Engine Parameters</div>
        <div class="setting-group">
            <label class="setting-label">Confidence Threshold (%)</label>
            <input type="number" class="setting-input" value="60" min="0" max="100" id="set-conf-thresh">
        </div>
        <div class="setting-group">
            <label class="setting-label">Signal Smoothing</label>
            <input type="number" class="setting-input" value="3" min="1" max="10" id="set-smoothing">
        </div>
        <div class="setting-group">
            <label class="setting-label">Entropy Window</label>
            <input type="number" class="setting-input" value="20" min="5" max="100" id="set-entropy-win">
        </div>
        <div class="setting-group">
            <label class="setting-label">Swing Period</label>
            <input type="number" class="setting-input" value="5" min="3" max="20" id="set-swing-period">
        </div>
    </div>
</div>

<!-- TOOLTIP -->
<div class="custom-tooltip" id="tooltip"></div>

<script>
// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════
const candleData = {json.dumps(candle_data)};
const volumeData = {json.dumps(volume_data)};
const signalData = {json.dumps(signal_data)};
const confidenceData = {json.dumps(confidence_data)};
const kalmanData = {json.dumps(kalman_data)};
const longMarkers = {json.dumps(long_markers)};
const shortMarkers = {json.dumps(short_markers)};
const factorData = {json.dumps(factor_data)};
const regimeData = {json.dumps(regime_data)};

// ═══════════════════════════════════════════════════════════════
// MAIN CHART
// ═══════════════════════════════════════════════════════════════
const mainEl = document.getElementById('main-chart');
const mainChart = LightweightCharts.createChart(mainEl, {{
    width: mainEl.clientWidth,
    height: mainEl.clientHeight,
    layout: {{
        background: {{ type: 'solid', color: '{visual_config["bg_color"]}' }},
        textColor: '{visual_config["text_color"]}',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: {visual_config['font_size']},
    }},
    grid: {{
        vertLines: {{ color: '{visual_config["grid_color"]}' }},
        horzLines: {{ color: '{visual_config["grid_color"]}' }},
    }},
    crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {{ labelBackgroundColor: '#2962FF' }},
        horzLine: {{ labelBackgroundColor: '#2962FF' }},
    }},
    rightPriceScale: {{
        borderColor: '{visual_config["grid_color"]}',
        scaleMargins: {{ top: 0.05, bottom: 0.15 }},
    }},
    timeScale: {{
        borderColor: '{visual_config["grid_color"]}',
        timeVisible: true,
        secondsVisible: false,
    }},
    handleScroll: {{ vertTouchDrag: false }},
}});

// Candlestick series
const candleSeries = mainChart.addCandlestickSeries({{
    upColor: '{visual_config["up_color"]}',
    downColor: '{visual_config["down_color"]}',
    borderUpColor: '{visual_config["candle_border_up"]}',
    borderDownColor: '{visual_config["candle_border_down"]}',
    wickUpColor: '{visual_config["candle_wick_up"]}',
    wickDownColor: '{visual_config["candle_wick_down"]}',
}});
candleSeries.setData(candleData);

// Volume
const volumeSeries = mainChart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'volume',
}});
mainChart.priceScale('volume').applyOptions({{
    scaleMargins: {{ top: 0.85, bottom: 0 }},
}});
volumeSeries.setData(volumeData);

// Kalman line
const kalmanSeries = mainChart.addLineSeries({{
    color: '{visual_config["kalman_color"]}',
    lineWidth: {visual_config['line_width']},
    lineStyle: 0,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
}});
kalmanSeries.setData(kalmanData);

// Signal markers — de-duplicate nearby signals to prevent overlap
let allMarkers = [...longMarkers, ...shortMarkers].sort((a,b) => a.time - b.time);
if (allMarkers.length > 1) {{
    const filtered = [allMarkers[0]];
    for (let i = 1; i < allMarkers.length; i++) {{
        const prev = filtered[filtered.length - 1];
        const curr = allMarkers[i];
        if (curr.time - prev.time < 3 * 86400 && prev.position === curr.position) {{
            if ((curr.text || '').length >= (prev.text || '').length) filtered[filtered.length - 1] = curr;
        }} else filtered.push(curr);
    }}
    allMarkers = filtered;
}}
candleSeries.setMarkers(allMarkers);

// ═══════════════════════════════════════════════════════════════
// SIGNAL SUB-CHART
// ═══════════════════════════════════════════════════════════════
const sigEl = document.getElementById('signal-chart');
const sigChart = LightweightCharts.createChart(sigEl, {{
    width: sigEl.clientWidth,
    height: 180,
    layout: {{
        background: {{ type: 'solid', color: '{visual_config["bg_color"]}' }},
        textColor: '{visual_config["text_color"]}',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
    }},
    grid: {{
        vertLines: {{ color: '{visual_config["grid_color"]}' }},
        horzLines: {{ color: '{visual_config["grid_color"]}' }},
    }},
    rightPriceScale: {{
        borderColor: '{visual_config["grid_color"]}',
    }},
    timeScale: {{
        borderColor: '{visual_config["grid_color"]}',
        timeVisible: true,
        visible: false,
    }},
    crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
    }},
}});

// Signal line (histogram for visual impact)
const signalHistSeries = sigChart.addHistogramSeries({{
    priceFormat: {{ type: 'custom', formatter: (v) => v.toFixed(1) }},
}});

const signalHistData = signalData.map(d => ({{
    time: d.time,
    value: d.value,
    color: d.value > 20 ? 'rgba(0,230,118,0.7)' : 
           d.value < -20 ? 'rgba(255,23,68,0.7)' : 
           'rgba(255,235,59,0.4)',
}}));
signalHistSeries.setData(signalHistData);

// Confidence line overlay
const confSeries = sigChart.addLineSeries({{
    color: '#FFD700',
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
}});
confSeries.setData(confidenceData);

// Zero line
const zeroLine = sigChart.addLineSeries({{
    color: 'rgba(255,255,255,0.2)',
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
}});
if (signalData.length > 0) {{
    zeroLine.setData([
        {{ time: signalData[0].time, value: 0 }},
        {{ time: signalData[signalData.length-1].time, value: 0 }}
    ]);
}}

// Threshold lines
const threshUp = sigChart.addLineSeries({{
    color: 'rgba(0,230,118,0.3)',
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
}});
const threshDown = sigChart.addLineSeries({{
    color: 'rgba(255,23,68,0.3)',
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
}});
if (signalData.length > 0) {{
    threshUp.setData([
        {{ time: signalData[0].time, value: 20 }},
        {{ time: signalData[signalData.length-1].time, value: 20 }}
    ]);
    threshDown.setData([
        {{ time: signalData[0].time, value: -20 }},
        {{ time: signalData[signalData.length-1].time, value: -20 }}
    ]);
}}

// ═══════════════════════════════════════════════════════════════
// FACTOR SUB-CHART
// ═══════════════════════════════════════════════════════════════
const facEl = document.getElementById('factor-chart');
const facChart = LightweightCharts.createChart(facEl, {{
    width: facEl.clientWidth,
    height: 120,
    layout: {{
        background: {{ type: 'solid', color: '{visual_config["bg_color"]}' }},
        textColor: '{visual_config["text_color"]}',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
    }},
    grid: {{
        vertLines: {{ color: '{visual_config["grid_color"]}' }},
        horzLines: {{ color: '{visual_config["grid_color"]}' }},
    }},
    rightPriceScale: {{ borderColor: '{visual_config["grid_color"]}' }},
    timeScale: {{ borderColor: '{visual_config["grid_color"]}', timeVisible: true, visible: false }},
}});

// Add factor lines
const factorSeries = {{}};
for (const [name, info] of Object.entries(factorData)) {{
    const s = facChart.addLineSeries({{
        color: info.color,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        title: name,
    }});
    s.setData(info.data);
    factorSeries[name] = s;
}}

// ═══════════════════════════════════════════════════════════════
// SYNC CROSSHAIRS
// ═══════════════════════════════════════════════════════════════
function syncCharts(sourceChart, targetCharts) {{
    sourceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
        if (range) {{
            targetCharts.forEach(tc => tc.timeScale().setVisibleLogicalRange(range));
        }}
    }});
}}

syncCharts(mainChart, [sigChart, facChart]);
syncCharts(sigChart, [mainChart, facChart]);
syncCharts(facChart, [mainChart, sigChart]);

// ═══════════════════════════════════════════════════════════════
// CROSSHAIR DATA DISPLAY
// ═══════════════════════════════════════════════════════════════
mainChart.subscribeCrosshairMove(param => {{
    if (!param || !param.time) return;
    
    const candle = param.seriesData.get(candleSeries);
    if (candle) {{
        const pct = ((candle.close - candle.open) / candle.open * 100).toFixed(2);
        const sign = pct >= 0 ? '+' : '';
        document.getElementById('price-display').textContent = 
            '$' + candle.close.toFixed(2) + ' (' + sign + pct + '%)';
        document.getElementById('price-display').className = 
            'price-change ' + (pct >= 0 ? 'price-up' : 'price-down');
    }}
    
    // Find closest signal data
    const time = param.time;
    const sigPoint = signalData.find(d => d.time === time);
    const confPoint = confidenceData.find(d => d.time === time);
    
    if (sigPoint) {{
        document.getElementById('ind-signal').textContent = sigPoint.value.toFixed(1);
        document.getElementById('ind-signal').style.color = 
            sigPoint.value > 20 ? 'var(--long-color)' : 
            sigPoint.value < -20 ? 'var(--short-color)' : '#FFD700';
    }}
    if (confPoint) {{
        document.getElementById('ind-confidence').textContent = confPoint.value.toFixed(0) + '%';
    }}
    
    // Regime
    const regPoint = regimeData.find(d => d.time === time);
    if (regPoint) {{
        const regEl = document.getElementById('ind-regime');
        regEl.textContent = regPoint.name;
        regEl.style.color = regPoint.regime === 0 ? 'var(--long-color)' : 
                            regPoint.regime === 1 ? 'var(--short-color)' : '#FFEB3B';
    }}
}});

// ═══════════════════════════════════════════════════════════════
// REGIME BANNER
// ═══════════════════════════════════════════════════════════════
if (regimeData.length > 0) {{
    const lastRegime = regimeData[regimeData.length - 1];
    const banner = document.getElementById('regime-banner');
    const cls = lastRegime.regime === 0 ? 'regime-bull' : 
                lastRegime.regime === 1 ? 'regime-bear' : 'regime-range';
    banner.innerHTML = `<span class="regime-tag ${{cls}}">⬤ ${{lastRegime.name}} REGIME</span>`;
}}

// ═══════════════════════════════════════════════════════════════
// FACTOR BREAKDOWN IN SIDE PANEL
// ═══════════════════════════════════════════════════════════════
const breakdownEl = document.getElementById('factor-breakdown');
const factorNames = {json.dumps({name: color for _, name, color in factor_cols})};
for (const [name, color] of Object.entries(factorNames)) {{
    const data = factorData[name];
    if (!data) continue;
    const lastVal = data.data[data.data.length - 1]?.value || 0;
    const row = document.createElement('div');
    row.className = 'stat-row';
    row.innerHTML = `
        <span class="stat-label" style="display:flex;align-items:center;gap:4px;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${{data.color}}"></span>
            ${{name}}
        </span>
        <span class="stat-value" style="color: ${{lastVal > 0 ? 'var(--long-color)' : lastVal < 0 ? 'var(--short-color)' : 'var(--text-secondary)'}}">${{lastVal.toFixed(1)}}</span>
    `;
    breakdownEl.appendChild(row);
}}

// ═══════════════════════════════════════════════════════════════
// UI CONTROLS
// ═══════════════════════════════════════════════════════════════
function togglePanel() {{
    document.getElementById('side-panel').classList.toggle('open');
    document.getElementById('settings-panel').classList.remove('open');
}}

function toggleSettings() {{
    document.getElementById('settings-panel').classList.toggle('open');
    document.getElementById('side-panel').classList.remove('open');
}}

function setTF(tf) {{
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    // In a live system, this would reload data for the selected timeframe
    console.log('Timeframe set to:', tf);
}}

function toggleIndicator(name) {{
    console.log('Toggle:', name);
    // Placeholder for live toggle functionality
}}

function takeScreenshot() {{
    mainChart.takeScreenshot();
}}

// ═══════════════════════════════════════════════════════════════
// DRAG-RESIZE SUB-CHARTS
// ═══════════════════════════════════════════════════════════════
function setupResize(handleId, topEl, bottomEl, topChart, bottomChart) {{
    const handle = document.getElementById(handleId);
    if (!handle) return;
    let startY, startTopH, startBottomH;
    handle.addEventListener('mousedown', (e) => {{
        startY = e.clientY; startTopH = topEl.clientHeight; startBottomH = bottomEl.clientHeight;
        const onMove = (e2) => {{
            const dy = e2.clientY - startY;
            const newTop = Math.max(60, startTopH + dy);
            const newBottom = Math.max(60, startBottomH - dy);
            topEl.style.height = newTop + 'px'; topEl.style.flex = 'none';
            bottomEl.style.height = newBottom + 'px';
            topChart.applyOptions({{ width: topEl.clientWidth, height: newTop }});
            bottomChart.applyOptions({{ width: bottomEl.clientWidth, height: newBottom }});
        }};
        const onUp = () => {{ document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }};
        document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
    }});
}}
setupResize('resize1', mainEl, sigEl, mainChart, sigChart);
setupResize('resize2', sigEl, facEl, sigChart, facChart);

// ═══════════════════════════════════════════════════════════════
// RESIZE HANDLER
// ═══════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {{
    mainChart.applyOptions({{ width: mainEl.clientWidth, height: mainEl.clientHeight }});
    sigChart.applyOptions({{ width: sigEl.clientWidth }});
    facChart.applyOptions({{ width: facEl.clientWidth }});
}});

// Update last price display
if (candleData.length > 0) {{
    const last = candleData[candleData.length - 1];
    const prev = candleData.length > 1 ? candleData[candleData.length - 2] : last;
    const pct = ((last.close - prev.close) / prev.close * 100).toFixed(2);
    const sign = pct >= 0 ? '+' : '';
    document.getElementById('price-display').textContent = 
        '$' + last.close.toFixed(2) + ' (' + sign + pct + '%)';
}}

console.log('NAU Quantum Alpha Engine v3.0 loaded successfully');
console.log('Total candles:', candleData.length);
console.log('Long signals:', longMarkers.length);
console.log('Short signals:', shortMarkers.length);
</script>
</body>
</html>"""
    
    return html_content


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution - generate data, compute indicator, create chart."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         NAU QUANTUM ALPHA ENGINE v3.0 — Starting...        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Generate sample data
    print("[1/4] Generating realistic OHLCV data...")
    df = generate_realistic_ohlcv(symbol="AAPL", days=120, timeframe='1H', seed=42)
    print(f"      → {len(df)} candles generated")
    
    # Initialize and compute indicator
    print("[2/4] Computing NAU Quantum Alpha Indicator...")
    indicator = NAUQuantumAlphaIndicator()
    df = indicator.compute(df)
    
    # Print summary
    total_long = int(df['NAU_Long'].sum())
    total_short = int(df['NAU_Short'].sum())
    avg_conf = df['NAU_Confidence'].mean() * 100
    last_sig = df['NAU_Signal'].iloc[-1]
    
    print(f"      → Long signals:  {total_long}")
    print(f"      → Short signals: {total_short}")
    print(f"      → Avg confidence: {avg_conf:.1f}%")
    print(f"      → Last signal: {last_sig:.1f}")
    
    # Generate HTML chart
    print("[3/4] Generating interactive chart...")
    html = generate_html_chart(df)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nau_quantum_chart.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"      → Chart saved to: {output_path}")
    
    # Export data
    print("[4/4] Exporting indicator data...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nau_indicator_data.csv')
    export_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'NAU_Signal', 'NAU_Confidence', 'NAU_Regime',
                   'NAU_Long', 'NAU_Short']
    df[export_cols].to_csv(csv_path)
    print(f"      → Data exported to: {csv_path}")
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    ✅ COMPLETE!                             ║")
    print("║  Open nau_quantum_chart.html in your browser to view.      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    return df, html


if __name__ == '__main__':
    df, html = main()
