import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.cuda.is_available = lambda: False

import torchmetrics.metric as tm


_original_apply = tm.Metric._apply
def _safe_apply(self_metric, fn):
    try:
        return _original_apply(self_metric, fn)
    except AssertionError:
        return self_metric
tm.Metric._apply = _safe_apply

import pandas as pd
import json
import yfinance as yf
import ta
import pickle
import inspect
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer


class TechnicalPredictor:

    def __init__(self, symbol):
        self.symbol = symbol
        self.model_dir = r"ml_models/technical_prediction"
        self.device = torch.device("cpu")

        with open(os.path.join(self.model_dir, "training_dataset.pkl"), "rb") as f:
            self.training_ref = pickle.load(f)

        ckpt_path = os.path.join(self.model_dir, "tft-best.ckpt")
        clean_path = os.path.join(self.model_dir, "tft-best-clean.ckpt")

        if os.path.exists(clean_path):
            self.tft = self._load_checkpoint_cpu(clean_path)
        else:
            self.tft = self._load_and_clean_checkpoint(ckpt_path, clean_path)

        self.tft.eval()

        print(f"Model loaded | Stocks in training: {len(self.training_ref.decoded_index['stock'].unique())}")

    def _load_checkpoint_cpu(self, path):
        model = TemporalFusionTransformer.load_from_checkpoint(
            path,
            map_location=self.device,
            output_transformer=self.training_ref.target_normalizer 
        )
        model.to(self.device)
        return model

    def _load_and_clean_checkpoint(self, ckpt_path, clean_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        valid_params = set(inspect.signature(TemporalFusionTransformer.__init__).parameters.keys())
        hparams = ckpt["hyper_parameters"]
        for k in [k for k in hparams if k not in valid_params]:
            hparams.pop(k)

        for key in ckpt["state_dict"]:
            if torch.is_tensor(ckpt["state_dict"][key]):
                ckpt["state_dict"][key] = ckpt["state_dict"][key].cpu()

        torch.save(ckpt, clean_path)
        print("Clean checkpoint saved")

        model = TemporalFusionTransformer.load_from_checkpoint(
            clean_path,
            map_location=self.device,
            output_transformer=self.training_ref.target_normalizer
        )
        return model.cpu()

    def check_symbol(self) -> bool:
        with open(
            r"data/raw/market_data/stocks_symbols.json",
            "r"
        ) as f:
            stock_symbols = json.load(f)
        return self.symbol in stock_symbols

    def get_data(self):
        data = yf.download(self.symbol + ".NS", period="6mo", interval="1d")
        data.columns = [col[0].lower() for col in data.columns]
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        data["stock"] = self.symbol
        print(data)
        return data

    def add_indicators(self, stock_df):
        stock_df["sma_7"]  = stock_df["close"].rolling(window=7).mean()
        stock_df["sma_20"] = stock_df["close"].rolling(window=20).mean()
        stock_df["sma_50"] = stock_df["close"].rolling(window=50).mean()
        stock_df["ema_9"]  = stock_df["close"].ewm(span=9, adjust=False).mean()
        stock_df["ema_21"] = stock_df["close"].ewm(span=21, adjust=False).mean()
        stock_df["ema_50"] = stock_df["close"].ewm(span=50, adjust=False).mean()
        stock_df["macd"]   = ta.trend.MACD(stock_df["close"]).macd()
        stock_df["rsi"]    = ta.momentum.RSIIndicator(stock_df["close"]).rsi()
        stock_df["bollinger_bands_upper"] = ta.volatility.BollingerBands(stock_df["close"]).bollinger_hband()
        stock_df["bollinger_bands_lower"] = ta.volatility.BollingerBands(stock_df["close"]).bollinger_lband()

        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df["time_idx"] = range(len(stock_df))
        stock_df.dropna(inplace=True)
        stock_df = stock_df.reset_index(drop=True)
        return stock_df
    

    def _extract_indicator_snapshot(self, df: pd.DataFrame) -> dict:
        """
        Compresses the 60-day dataframe into a clean indicator summary.
        LLM gets intelligence, not raw rows.
        """
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        current_price = float(latest["close"])

        price_context = {
            "current_price":     round(current_price, 2),
            "60d_high":          round(float(df["high"].max()), 2),
            "60d_low":           round(float(df["low"].min()), 2),
            "60d_avg_close":     round(float(df["close"].mean()), 2),
            "pct_from_60d_high": round((current_price - df["high"].max()) / df["high"].max() * 100, 2),
            "pct_from_60d_low":  round((current_price - df["low"].min()) / df["low"].min() * 100, 2),
        }

        ma_context = {}
        for col in ["sma_7", "sma_20", "sma_50", "ema_9", "ema_21", "ema_50"]:
            if col in df.columns:
                val = round(float(latest[col]), 2)
                ma_context[col] = {
                    "value":        val,
                    "price_vs_ma":  "ABOVE" if current_price > val else "BELOW",
                    "pct_distance": round((current_price - val) / val * 100, 2)
                }

        cross = "NONE"
        if "sma_20" in df.columns and "sma_50" in df.columns:
            if latest["sma_20"] > latest["sma_50"] and prev["sma_20"] <= prev["sma_50"]:
                cross = "GOLDEN CROSS — sma20 just crossed above sma50 (BULLISH)"
            elif latest["sma_20"] < latest["sma_50"] and prev["sma_20"] >= prev["sma_50"]:
                cross = "DEATH CROSS — sma20 just crossed below sma50 (BEARISH)"
        ma_context["cross_signal"] = cross

        rsi_context = {}
        if "rsi" in df.columns:
            rsi_val      = round(float(latest["rsi"]), 2)
            rsi_last_5   = [round(float(x), 2) for x in df["rsi"].tail(5).tolist()]
            rsi_context  = {
            "current":     rsi_val,
            "zone":        "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NEUTRAL"),
            "last_5_days": rsi_last_5,
            "direction":   "RISING" if rsi_last_5[-1] > rsi_last_5[0] else "FALLING",

            "price_trend_5d":   "UP" if float(df["close"].iloc[-1]) > float(df["close"].iloc[-5]) else "DOWN",
        }
        if rsi_context["price_trend_5d"] == "DOWN" and rsi_context["direction"] == "RISING":
            rsi_context["divergence"] = "BULLISH DIVERGENCE — price falling but RSI rising (potential reversal)"
        elif rsi_context["price_trend_5d"] == "UP" and rsi_context["direction"] == "FALLING":
            rsi_context["divergence"] = "BEARISH DIVERGENCE — price rising but RSI falling (potential exhaustion)"
        else:
            rsi_context["divergence"] = "NONE"


        macd_context = {}
        if "macd" in df.columns:
            macd_val   = round(float(latest["macd"]), 4)
            prev_macd  = round(float(prev["macd"]), 4)


        macd_signal_series = df["macd"].ewm(span=9, adjust=False).mean()
        signal_val  = round(float(macd_signal_series.iloc[-1]), 4)
        prev_signal = round(float(macd_signal_series.iloc[-2]), 4)
        hist_val    = round(macd_val - signal_val, 4)
        prev_hist   = round(prev_macd - prev_signal, 4)

        macd_context = {
            "macd_line":        macd_val,
            "signal_line":      signal_val,
            "histogram":        hist_val,
            "histogram_trend":  "EXPANDING" if abs(hist_val) > abs(prev_hist) else "CONTRACTING",
            "position":         "ABOVE ZERO" if macd_val > 0 else "BELOW ZERO",
            "crossover":        (
                "BULLISH CROSSOVER" if macd_val > signal_val and prev_macd <= prev_signal
                else "BEARISH CROSSOVER" if macd_val < signal_val and prev_macd >= prev_signal
                else "NO RECENT CROSSOVER"
            )
        }
        bb_context = {}
        if "bollinger_bands_upper" in df.columns and "bollinger_bands_lower" in df.columns:
            bb_upper = round(float(latest["bollinger_bands_upper"]), 2)
            bb_lower = round(float(latest["bollinger_bands_lower"]), 2)
            bb_mid   = round((bb_upper + bb_lower) / 2, 2)
            bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2)

            widths_20d    = ((df["bollinger_bands_upper"] - df["bollinger_bands_lower"]) /
                         ((df["bollinger_bands_upper"] + df["bollinger_bands_lower"]) / 2) * 100)
            avg_width_20d = round(float(widths_20d.tail(20).mean()), 2)

            bb_context = {
                "upper":           bb_upper,
                "middle":          bb_mid,
                "lower":           bb_lower,
                "band_width_pct":  bb_width,
                "avg_width_20d":   avg_width_20d,
                "squeeze":         bb_width < (avg_width_20d * 0.75),
                "price_position":  (
                    "NEAR UPPER BAND (overbought pressure)" if current_price >= bb_upper * 0.99
                    else "NEAR LOWER BAND (oversold pressure)" if current_price <= bb_lower * 1.01
                    else "NEAR MIDDLE" if abs(current_price - bb_mid) / bb_mid < 0.01
                    else "BETWEEN MIDDLE AND UPPER" if current_price > bb_mid
                    else "BETWEEN MIDDLE AND LOWER"
                )
            }

        volume_context = {}
        if "volume" in df.columns:
            vol_today   = int(latest["volume"])
            vol_20d_avg = int(df["volume"].tail(20).mean())
            vol_ratio   = round(vol_today / vol_20d_avg, 2)

            volume_context = {
            "today":        vol_today,
            "avg_20d":      vol_20d_avg,
            "ratio_vs_avg": vol_ratio,
            "signal":       (
                "HIGH VOLUME — strong participation" if vol_ratio > 1.5
                else "LOW VOLUME — weak participation" if vol_ratio < 0.7
                else "NORMAL VOLUME"
            ),
            "confirms_price_move": (
                "YES" if (current_price > float(prev["close"]) and vol_ratio > 1.0)
                      or (current_price < float(prev["close"]) and vol_ratio > 1.0)
                else "NO — price move not confirmed by volume (divergence)"
            )
            }


        recent_highs = df["high"].tail(20)
        recent_lows  = df["low"].tail(20)

        sr_context = {
            "resistance_levels": sorted(
            [round(float(x), 2) for x in recent_highs.nlargest(3).unique()], reverse=True
            ),
            "support_levels": sorted(
            [round(float(x), 2) for x in recent_lows.nsmallest(3).unique()]
            )
        }

        return {
        "price_context":      price_context,
        "moving_averages":    ma_context,
        "rsi":                rsi_context,
        "macd":               macd_context,
        "bollinger_bands":    bb_context,
        "volume":             volume_context,
        "support_resistance": sr_context,
        }

    def predict(self):
        if not self.check_symbol():
            return {"error": f"{self.symbol} not in supported stocks list"}

        df = self.get_data()
        df = self.add_indicators(df)
        df["time_idx"] = range(len(df))

        inference_dataset = TimeSeriesDataSet.from_dataset(
            self.training_ref,
            df,
            predict=True
        )

        dataloader = inference_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0
        )
        quantiles = self.tft.loss.quantiles
        print("Model quantiles:", quantiles)

        output = self.tft.predict(dataloader, mode="quantiles")
        quantile_preds = output.cpu().numpy()

        if quantile_preds.ndim == 3:
            quantile_preds = quantile_preds[0]  

        print("Shape after squeeze:", quantile_preds.shape)  

        q_list = [round(float(q), 2) for q in quantiles]

        try:
            idx_pessimistic = q_list.index(0.1)
        except ValueError:
            idx_pessimistic = 1  

        try:
            idx_base = q_list.index(0.5)
        except ValueError:
            idx_base = 3  

        try:
            idx_optimistic = q_list.index(0.9)
        except ValueError:
            idx_optimistic = 5  

        dates = pd.bdate_range(start=pd.Timestamp.today(), periods=7)

        forecast = {}
        for i, date in enumerate(dates.strftime("%Y-%m-%d")):
            base         = round(float(quantile_preds[i][idx_base]), 2)
            pessimistic  = round(float(quantile_preds[i][idx_pessimistic]), 2)
            optimistic   = round(float(quantile_preds[i][idx_optimistic]), 2)
            band_width   = round((optimistic - pessimistic) / base * 100, 2)

            forecast[f"day_{i+1}"] = {
                "date":                   date,
                "pessimistic":            pessimistic,
                "base":                   base,
                "optimistic":             optimistic,
                "band_width_pct":         band_width,
                "pct_change_from_today":  round((base - float(df["close"].iloc[-1])) / float(df["close"].iloc[-1]) * 100, 2),
                "confidence":             "HIGH" if band_width < 2 else ("MEDIUM" if band_width < 5 else "LOW")
            }

        indicator_snapshot = self._extract_indicator_snapshot(df)

        return {
            "forecast":   forecast,
            "indicators": indicator_snapshot
        }

