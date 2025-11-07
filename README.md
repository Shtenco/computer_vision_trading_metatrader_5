# 💹 Computer Vision for Trading: Teaching Machines to See Markets

## What is this?

I'm teaching neural networks to see financial markets the way professional traders do - as visual landscapes full of patterns, textures, and hidden signals. Instead of feeding raw numbers into algorithms, I transform price data into RGB images that convolutional neural networks can analyze like photographs.

This project takes 60 bars of EURUSD data and converts them into 128×128 pixel images where:
- Red channel shows price movements (OHLC)
- Green channel displays technical indicators (SMA, MACD, Bollinger Bands)
- Blue channel represents oscillators (RSI, Stochastic, volatility)

The model learns to spot patterns that traders spend years learning to recognize - head and shoulders, double bottoms, flags - except it sees them all at once across multiple dimensions.

## Why RGB images?

When you convert numbers into images, something magical happens. Patterns that hide in spreadsheets suddenly become visible. It's like the difference between describing someone's face with measurements versus actually seeing their photograph. The neural network can now use computer vision techniques that have been perfected on millions of images to understand market structure.

I also overlay candlestick patterns onto the images. Bullish candles glow green like hope, bearish candles burn red like warning signs. The wicks draw as shadows showing the battle between buyers and sellers.
```python
if close_y < open_y:  # Bullish candle
    for y in range(close_y, open_y):
        if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
            img[y, x, 1] = 1.0  # Green
else:  # Bearish candle
    for y in range(open_y, close_y):
        if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
            img[y, x, 0] = 1.0  # Red
```

## The Architecture

My model has three parallel paths, each looking at the market from a different angle:

**Local path** with short convolutional filters catches sudden movements - breakouts, reversals, sharp jumps.

**Trend path** with longer filters watches for big trends that flow like rivers through market cycles.

**Volatility path** monitors oscillators to understand if the market is calm or about to explode.
```python
local_path = Conv1D(64, 3, padding='same', activation='relu')(inputs[:,:,:,0])
trend_path = Conv1D(64, 7, padding='same', activation='relu')(inputs[:,:,:,1])
vol_path = Conv1D(64, 5, padding='same', activation='relu')(inputs[:,:,:,2])
```

These paths merge through an attention mechanism that teaches the model to focus on what matters, filtering out noise like a trader scanning charts for the one key signal. Then a bidirectional LSTM looks both forward and backward through time to catch complex patterns - long consolidations, hidden reversals, moments when the market holds its breath before a move.

The model predicts two things at once: direction (up or down) and magnitude (how much it will move in percent). As Peter Lynch said, knowing a stock will go up is one thing, predicting it will rise exactly 8% is something else entirely.

## Looking Inside the Model

The most exciting part is that I can see what the model is thinking. I created attention map visualizations that show which parts of the chart the model focuses on when making decisions.
```python
attention = np.mean(np.abs(attention_maps[i]), axis=-1)
heatmap = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
```

When I analyzed the internal neurons, I discovered they specialized naturally - some fire only on sharp reversals, others on smooth trends, others on quiet periods before storms. The model built its own map of the market, classifying situations in ways no human analyst does. It even found non-standard divergences between RSI and price that experienced traders later confirmed as significant.

This isn't just an algorithm. It's a new way to see markets.

## Technical Details

I use RobustScaler for normalization because markets are chaotic and news spikes can distort everything. RobustScaler keeps the model stable by ignoring outliers.
```python
scaler = RobustScaler()
if window_data[indicator].std() != 0:
    img_data[j] = scaler.fit_transform(window_data[indicator].values.reshape(-1, 1)).flatten()
```

Gaussian blur smooths noise so the model focuses on patterns rather than random spikes. It's like putting on glasses to see the forest instead of individual trees.
```python
img = gaussian_filter1d(img, sigma=0.5, axis=0)
img = gaussian_filter1d(img, sigma=0.5, axis=1)
```

Training uses callbacks to prevent overfitting - EarlyStopping kills the process if the model starts memorizing, ModelCheckpoint saves the best version, ReduceLROnPlateau slows down when progress stalls.

## Results

Accuracy stays above 53% on test data even with minimal training epochs. That might sound modest, but in algorithmic trading even 1-2% edge over random chance compounds exponentially with proper risk management.

The model maintains precision around 0.54, recall near 0.52, and MAE under 0.5% for price change predictions. These aren't just numbers - they represent real predictive power on one of the most liquid and efficient markets in the world.

## Quick Start
```bash
git clone https://github.com/yourusername/cv-trading-system.git
cd cv-trading-system
pip install -r requirements.txt
python cv_trading_system.py
```

You need Python 3.8+ and MetaTrader 5 connected to a broker. The system will automatically load 5000 bars of EURUSD history, create RGB images, train the model, and generate visualizations.

## What Gets Created

The system produces attention maps showing where the model looks when making decisions, animated GIFs of sliding window predictions, confusion matrices and learning curves, and detailed prediction visualizations with confidence levels.

All outputs save to the `outputs/` folder. The best model weights save to `model_checkpoints/best_model.keras`.

## Future Plans

I want to integrate fundamental data - economic indicators, news events, market sentiment. Transformers could convert Bloomberg headlines and Fed reports into vectors that merge with technical data.

Markets are fractals where patterns repeat across timeframes. Teaching the model to analyze multiple timeframes simultaneously - from minutes to months - could reveal nested trends that escape even the best analysts.

Metalearning would let the model adapt itself to changing market regimes. If volatility spikes, it adjusts parameters automatically like a captain steering through a storm.

Deeper visualization of attention and activations could teach human traders to see markets differently. Maybe the model discovers patterns nobody noticed before and traders start calling it the "neural cross" in its honor.

## Warning

This is for education and research only. Trading carries serious risks and can destroy capital. I'm not responsible for losses from using this system. Always test on demo accounts first, use strict risk management, and never trade money you can't afford to lose.

Remember, 53% accuracy means nothing without proper position sizing and risk control.

## About Me

**Yevgeniy Koshtenko**

Qualified investor in Kazakhstan and Russia. AI and quantum computing specialist. 10+ years in algorithmic trading since 2016. Published 100+ research papers in 15 languages on MQL5.com.

**Contact:**
- Email: koshtenco@gmail.com
- Telegram: @Shtenco
- VK: https://vk.com/altradinger
- MQL5: [https://www.mql5.com/ru/users/yevgeniy.koshtenko](https://www.mql5.com/ru/users/koshtenko)

## License

MIT License. Use freely with attribution.
