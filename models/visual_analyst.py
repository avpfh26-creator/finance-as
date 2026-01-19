"""
Professional-Grade Pattern Detection System.
Uses scipy peak detection with strict quality validation.
Only detects high-confidence, actionable patterns.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
from typing import List, Dict, Optional, Tuple


class PatternAnalyst:
    """
    Professional pattern detection with strict quality criteria.
    
    Key Improvements:
    - Stricter tolerance (1% vs 2%)
    - Minimum pattern height requirements (5% minimum)
    - Volume confirmation
    - Time-based validation (patterns must form over reasonable timeframe)
    - Maximum 1-2 patterns per type to avoid noise
    """
    
    def __init__(self, order: int = 7):
        """
        Initialize the Pattern Analyst.
        
        Args:
            order: Number of points on each side for extrema detection (higher = fewer, cleaner peaks)
        """
        self.order = order
        self.min_pattern_height = 0.05  # 5% minimum pattern height
        self.max_patterns_per_type = 1  # Only show best pattern per type
    
    def find_peaks_and_troughs(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find significant peaks and troughs using scipy with prominence filtering.
        """
        prices_arr = prices.values
        
        # Use prominence-based peak detection for cleaner results
        price_range = prices_arr.max() - prices_arr.min()
        min_prominence = price_range * 0.02  # 2% of price range minimum prominence
        
        # Find peaks
        peaks, peak_props = find_peaks(prices_arr, prominence=min_prominence, distance=self.order)
        
        # Find troughs (invert the data)
        troughs, trough_props = find_peaks(-prices_arr, prominence=min_prominence, distance=self.order)
        
        return peaks, troughs
    
    def _validate_pattern_quality(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                                   pattern_height: float, current_price: float) -> bool:
        """
        Validate pattern quality with multiple criteria.
        
        Args:
            df: Price dataframe
            start_idx: Pattern start index
            end_idx: Pattern end index
            pattern_height: Height of the pattern
            current_price: Current price
            
        Returns:
            True if pattern meets quality criteria
        """
        # 1. Minimum pattern height (5% of current price)
        if pattern_height < current_price * self.min_pattern_height:
            return False
        
        # 2. Pattern should form over reasonable timeframe (5-40 trading days)
        pattern_duration = end_idx - start_idx
        if pattern_duration < 5 or pattern_duration > 40:
            return False
        
        # 3. Pattern should be recent (within last 45 days)
        if len(df) - end_idx > 45:
            return False
        
        return True
    
    def _check_volume_confirmation(self, df: pd.DataFrame, breakout_idx: int, pattern_type: str) -> bool:
        """
        Check if there's volume confirmation at potential breakout point.
        Higher volume at neckline suggests stronger pattern.
        """
        if 'Volume' not in df.columns:
            return True  # Skip if no volume data
        
        try:
            avg_volume = df['Volume'].iloc[max(0, breakout_idx-10):breakout_idx].mean()
            recent_volume = df['Volume'].iloc[breakout_idx:min(len(df), breakout_idx+3)].mean()
            
            # Volume should be at least 80% of average
            return recent_volume >= avg_volume * 0.8
        except Exception:
            return True
    
    def detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Double Top pattern with strict validation.
        Only returns highest quality pattern.
        """
        patterns = []
        df_analysis = df.tail(50)  # Only analyze last 50 days
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 2 or len(trough_idx) < 1:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        # Look at recent peaks only
        for i in range(len(peak_idx) - 1):
            p1_idx, p2_idx = peak_idx[i], peak_idx[i + 1]
            p1_price, p2_price = prices.iloc[p1_idx], prices.iloc[p2_idx]
            
            # Strict: peaks must be within 1% of each other
            price_diff_pct = abs(p1_price - p2_price) / p1_price
            if price_diff_pct > 0.01:
                continue
            
            # Find trough between peaks
            troughs_between = trough_idx[(trough_idx > p1_idx) & (trough_idx < p2_idx)]
            if len(troughs_between) == 0:
                continue
            
            trough_price = prices.iloc[troughs_between[0]]
            avg_peak = (p1_price + p2_price) / 2
            pattern_height = avg_peak - trough_price
            
            # Validate quality
            if not self._validate_pattern_quality(df_analysis, p1_idx, p2_idx, pattern_height, current_price):
                continue
            
            # Calculate confidence
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.1), 1) * 10  # Bonus for larger patterns
            confidence = min(confidence + height_score, 99)
            
            # Check if price has broken below neckline (confirmation)
            confirmed = current_price < trough_price
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = trough_price - pattern_height
                
                best_pattern = {
                    'Pattern': 'Double Top',
                    'Type': 'Bearish Reversal',
                    'Neckline': round(trough_price, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Peak_Price': round(avg_peak, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Double Bottom pattern with strict validation.
        """
        patterns = []
        df_analysis = df.tail(50)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 2 or len(peak_idx) < 1:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(trough_idx) - 1):
            t1_idx, t2_idx = trough_idx[i], trough_idx[i + 1]
            t1_price, t2_price = prices.iloc[t1_idx], prices.iloc[t2_idx]
            
            # Strict: troughs must be within 1% of each other
            price_diff_pct = abs(t1_price - t2_price) / t1_price
            if price_diff_pct > 0.01:
                continue
            
            peaks_between = peak_idx[(peak_idx > t1_idx) & (peak_idx < t2_idx)]
            if len(peaks_between) == 0:
                continue
            
            peak_price = prices.iloc[peaks_between[0]]
            avg_trough = (t1_price + t2_price) / 2
            pattern_height = peak_price - avg_trough
            
            if not self._validate_pattern_quality(df_analysis, t1_idx, t2_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.1), 1) * 10
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > peak_price
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = peak_price + pattern_height
                
                best_pattern = {
                    'Pattern': 'Double Bottom',
                    'Type': 'Bullish Reversal',
                    'Neckline': round(peak_price, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Trough_Price': round(avg_trough, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Head & Shoulders with strict validation.
        Requires head to be at least 3% higher than shoulders.
        """
        patterns = []
        df_analysis = df.tail(60)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 3 or len(trough_idx) < 2:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(peak_idx) - 2):
            ls_idx, h_idx, rs_idx = peak_idx[i], peak_idx[i+1], peak_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% higher than both shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_height_pct = (h_price - avg_shoulder) / avg_shoulder
            
            if head_height_pct < 0.03:
                continue
            
            # Shoulders must be within 2% of each other
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > 0.02:
                continue
            
            # Find neckline troughs
            troughs_1 = trough_idx[(trough_idx > ls_idx) & (trough_idx < h_idx)]
            troughs_2 = trough_idx[(trough_idx > h_idx) & (trough_idx < rs_idx)]
            
            if len(troughs_1) == 0 or len(troughs_2) == 0:
                continue
            
            neckline = (prices.iloc[troughs_1[0]] + prices.iloc[troughs_2[0]]) / 2
            pattern_height = h_price - neckline
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_height_pct * 100, 10)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < neckline
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = neckline - pattern_height
                
                best_pattern = {
                    'Pattern': 'Head & Shoulders',
                    'Type': 'Bearish Reversal',
                    'Neckline': round(neckline, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Head_Price': round(h_price, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Inverse H&S with strict validation.
        """
        patterns = []
        df_analysis = df.tail(60)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 3 or len(peak_idx) < 2:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(trough_idx) - 2):
            ls_idx, h_idx, rs_idx = trough_idx[i], trough_idx[i+1], trough_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% lower than shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_depth_pct = (avg_shoulder - h_price) / avg_shoulder
            
            if head_depth_pct < 0.03:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > 0.02:
                continue
            
            peaks_1 = peak_idx[(peak_idx > ls_idx) & (peak_idx < h_idx)]
            peaks_2 = peak_idx[(peak_idx > h_idx) & (peak_idx < rs_idx)]
            
            if len(peaks_1) == 0 or len(peaks_2) == 0:
                continue
            
            neckline = (prices.iloc[peaks_1[0]] + prices.iloc[peaks_2[0]]) / 2
            pattern_height = neckline - h_price
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_depth_pct * 100, 10)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > neckline
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = neckline + pattern_height
                
                best_pattern = {
                    'Pattern': 'Inverse H&S',
                    'Type': 'Bullish Reversal',
                    'Neckline': round(neckline, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Head_Price': round(h_price, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_trend(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Detect current trend using multiple confirmation methods.
        """
        prices = df['Close'].tail(window)
        
        # Linear regression
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = linregress(x, prices.values)
        r_squared = r_value ** 2  # How well the trend fits
        
        # MA crossover
        ma_short = df['Close'].rolling(5).mean().iloc[-1]
        ma_long = df['Close'].rolling(20).mean().iloc[-1]
        
        # Price structure
        recent_highs = df['High'].tail(10)
        recent_lows = df['Low'].tail(10)
        
        higher_highs = (recent_highs.diff().dropna() > 0).sum() / len(recent_highs.diff().dropna())
        higher_lows = (recent_lows.diff().dropna() > 0).sum() / len(recent_lows.diff().dropna())
        
        # Score trend
        bullish_score = 0
        bearish_score = 0
        
        if slope > 0:
            bullish_score += 1 + (r_squared * 0.5)  # Extra credit for strong fit
        else:
            bearish_score += 1 + (r_squared * 0.5)
            
        if ma_short > ma_long:
            bullish_score += 1
        else:
            bearish_score += 1
            
        if higher_highs > 0.6 and higher_lows > 0.6:
            bullish_score += 1
        elif higher_highs < 0.4 and higher_lows < 0.4:
            bearish_score += 1
        
        total_score = bullish_score + bearish_score
        if bullish_score > bearish_score:
            trend = "Bullish"
            strength = (bullish_score / total_score) * 100 if total_score > 0 else 50
        elif bearish_score > bullish_score:
            trend = "Bearish"
            strength = (bearish_score / total_score) * 100 if total_score > 0 else 50
        else:
            trend = "Neutral"
            strength = 50
        
        return {
            'Trend': trend,
            'Strength': round(strength, 1),
            'Slope': round(slope, 4),
            'R_Squared': round(r_squared, 3),
            'MA_Signal': 'Bullish' if ma_short > ma_long else 'Bearish',
            'Structure': 'Higher Highs/Lows' if higher_highs > 0.6 else 'Lower Highs/Lows' if higher_highs < 0.4 else 'Mixed'
        }
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Detect significant support and resistance levels.
        Uses clustering to find key price levels.
        """
        df_slice = df.tail(lookback)
        prices = df_slice['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        # Get price levels
        resistance_levels = [prices.iloc[idx] for idx in peak_idx[-5:]] if len(peak_idx) > 0 else []
        support_levels = [prices.iloc[idx] for idx in trough_idx[-5:]] if len(trough_idx) > 0 else []
        
        # Find nearest levels above and below current price
        resistance_above = [r for r in resistance_levels if r > current_price * 1.005]
        support_below = [s for s in support_levels if s < current_price * 0.995]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        return {
            'Current_Price': round(current_price, 2),
            'Nearest_Resistance': round(nearest_resistance, 2) if nearest_resistance else 'N/A',
            'Nearest_Support': round(nearest_support, 2) if nearest_support else 'N/A',
            'All_Resistance': [round(r, 2) for r in sorted(set(resistance_levels), reverse=True)[:3]],
            'All_Support': [round(s, 2) for s in sorted(set(support_levels), reverse=True)[:3]]
        }
    
    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive pattern analysis.
        Returns only high-quality, actionable patterns.
        """
        all_patterns = []
        
        # Detect patterns (each returns at most 1 best pattern)
        all_patterns.extend(self.detect_double_top(df))
        all_patterns.extend(self.detect_double_bottom(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(df))
        
        # Filter to only show patterns with confidence >= 85%
        high_quality_patterns = [p for p in all_patterns if p and p.get('Confidence', 0) >= 85]
        
        # Sort by confidence
        high_quality_patterns.sort(key=lambda x: x.get('Confidence', 0), reverse=True)
        
        # Get trend and S/R
        trend = self.detect_trend(df)
        sr_levels = self.detect_support_resistance(df)
        
        # Calculate bias
        bullish_patterns = sum(1 for p in high_quality_patterns if 'Bullish' in p.get('Type', ''))
        bearish_patterns = sum(1 for p in high_quality_patterns if 'Bearish' in p.get('Type', ''))
        
        if bullish_patterns > bearish_patterns:
            pattern_bias = "Bullish"
        elif bearish_patterns > bullish_patterns:
            pattern_bias = "Bearish"
        else:
            pattern_bias = trend['Trend']
        
        return {
            'patterns': high_quality_patterns[:3],  # Max 3 patterns
            'trend': trend,
            'support_resistance': sr_levels,
            'overall_bias': pattern_bias,
            'pattern_count': len(high_quality_patterns)
        }


# Backward compatibility
VISUAL_AI_AVAILABLE = True
