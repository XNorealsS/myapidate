import express from 'express';
import * as technicalIndicators from 'technicalindicators';
import { linearRegression } from 'simple-statistics';
import axios from 'axios';
const app = express();
const port = 3000;


const BINANCE_API_URL = 'https://fapi.binance.com/fapi/v1';

app.get('/analyze', async (req, res) => {
    try {
        const { symbol } = req.query; // Ambil parameter 'symbol' dari query string
        const { interval } = req.query; // Ambil parameter 'symbol' dari query string
        const accuracyThreshold = 60;
        const Volatilitylct = 'All';

        if (symbol) {
            // Analisis hanya untuk simbol tertentu
            const tradeOutput = await analyzeMarket(symbol, interval, accuracyThreshold, Volatilitylct);
            return res.json({
                status: 'success',
                message: 'Analysis completed for specific symbol',
                data: tradeOutput,
            });
        } else {
            // Analisis untuk semua simbol
            const symbols = await getFuturesSymbolsFromBinance();
            const results = [];

            for (const symbol of symbols) {
                try {
                    const tradeOutput = await analyzeMarket(symbol, interval, accuracyThreshold, Volatilitylct);
                    results.push(tradeOutput);
                } catch (error) {
                    console.error(`Error analyzing market for symbol ${symbol}:`, error);
                }
            }

            return res.json({
                status: 'success',
                message: 'Analysis completed for all symbols',
                data: results,
            });
        }
    } catch (error) {
        console.error('Error in /analyze endpoint:', error);
        res.status(500).json({ status: 'error', message: error.message });
    }
});


app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});



async function getFuturesSymbolsFromBinance() {
    try {
        const { data } = await axios.get(`${BINANCE_API_URL}/exchangeInfo`);
        return data.symbols
            .filter(symbol => symbol.symbol.endsWith('USDT'))
            .map(symbol => symbol.symbol);
    } catch (error) {
        console.error('Error fetching symbols from Binance Futures API:', error);
        return [];
    }
}

async function analyzeMarket(symbol, timeframe, accuracyThreshold, Volatilitylct) {
    try {
        const candleData = await fetchCandlestickData(symbol, timeframe);
        const candles = processCandles(candleData);

        if (candles.length < 200) {
            console.warn(`Not enough candles available for ${symbol} on ${timeframe}`);
            return null;
        }

        const lastClosedCandle = candles[candles.length - 2];
        const currentPrice = lastClosedCandle.close;
        const indicators = calculateIndicators(candles.slice(0, -1));
        const analysisData = analyzeMarketStructure(candles.slice(0, -1), indicators, currentPrice);
        const marketDirection = determineMarketDirection(candles);
        const volatility = calculateVolatility(candles);
        const closeTime = lastClosedCandle.closeTime;
        const date = new Date(closeTime);
        const formatter = new Intl.DateTimeFormat('en-US', {
            timeZone: 'Asia/Jakarta',
            year: 'numeric',
            month: 'numeric',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
        });
        const wibTime = formatter.format(date);

        const tradeOutput = generateTradeOutput(
            symbol,
            timeframe,
            currentPrice,
            analysisData,
            indicators,
            marketDirection,
            volatility,
            wibTime
        );

        return tradeOutput;
    } catch (error) {
        console.error(`Error analyzing market for ${symbol}: ${error.message}`);
        throw error;
    }
}



async function fetchCandlestickData(symbol, timeframe) {
    const { data } = await axios.get(`${BINANCE_API_URL}/klines`, {
        params: { symbol, interval: timeframe, limit: 500 }
    });
    return data.slice(0, -1);
}

function processCandles(candles) {
    return candles.map(([openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume]) => ({
        openTime: parseInt(openTime),
        open: parseFloat(open),
        high: parseFloat(high),
        low: parseFloat(low),
        close: parseFloat(close),
        volume: parseFloat(volume),
        closeTime: parseInt(closeTime),
        quoteAssetVolume: parseFloat(quoteAssetVolume),
        numberOfTrades: parseInt(numberOfTrades),
        takerBuyBaseAssetVolume: parseFloat(takerBuyBaseAssetVolume),
        takerBuyQuoteAssetVolume: parseFloat(takerBuyQuoteAssetVolume)
    }));
}



function calculateIndicators(candles) {
    const closes = candles.map(c => c.close);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const volumes = candles.map(c => c.volume);

    return {
        sma: technicalIndicators.SMA.calculate({period: 14, values: closes}),
        ema: technicalIndicators.EMA.calculate({period: 14, values: closes}),
        rsi: technicalIndicators.RSI.calculate({period: 14, values: closes}),
        macd: technicalIndicators.MACD.calculate({fastPeriod: 12, slowPeriod: 26, signalPeriod: 9, values: closes, SimpleMAOscillator: false, SimpleMASignal: false}),
        bb: technicalIndicators.BollingerBands.calculate({period: 20, stdDev: 2, values: closes}),
        atr: technicalIndicators.ATR.calculate({high: highs, low: lows, close: closes, period: 14}),
        obv: technicalIndicators.OBV.calculate({close: closes, volume: volumes}),
        adx: technicalIndicators.ADX.calculate({high: highs, low: lows, close: closes, period: 14}),
        stochastic: technicalIndicators.Stochastic.calculate({high: highs, low: lows, close: closes, period: 14, signalPeriod: 3}),
        ichimoku: technicalIndicators.IchimokuCloud.calculate({high: highs, low: lows, close: closes, conversionPeriod: 9, basePeriod: 26, spanPeriod: 52, displacement: 26}),
        vwap: technicalIndicators.VWAP.calculate({high: highs, low: lows, close: closes, volume: volumes}),
        trendStrength: calculateTrendStrength(closes),
        volumeProfile: calculateVolumeProfile(candles),
        marketPhase: determineMarketPhase(closes, volumes),

        parabolicSAR: calculatePSAR(highs, lows),
        cmf: calculateCMF(highs, lows, closes, volumes),
        cci: calculateCCI(highs, lows, closes)
    };
}

function calculateSMA(values, period) {
    return technicalIndicators.SMA.calculate({period, values});
}


function calculatePSAR(highs, lows, step = 0.02, max = 0.2) {
    let psar = [lows[0]];
    let ep = highs[0];
    let af = step;
    let bull = true;

    for (let i = 1; i < highs.length; i++) {
        psar.push(psar[i-1] + af * (ep - psar[i-1]));

        if (bull) {
            if (lows[i] < psar[i]) {
                bull = false;
                psar[i] = ep;
                ep = lows[i];
                af = step;
            } else {
                if (highs[i] > ep) {
                    ep = highs[i];
                    af = Math.min(af + step, max);
                }
            }
        } else {
            if (highs[i] > psar[i]) {
                bull = true;
                psar[i] = ep;
                ep = highs[i];
                af = step;
            } else {
                if (lows[i] < ep) {
                    ep = lows[i];
                    af = Math.min(af + step, max);
                }
            }
        }
    }

    return psar;
}

// Chaikin Money Flow (CMF)
function calculateCMF(highs, lows, closes, volumes, period = 20) {
    let cmf = [];

    for (let i = period - 1; i < highs.length; i++) {
        let sumMoneyFlowVolume = 0;
        let sumVolume = 0;

        for (let j = i - period + 1; j <= i; j++) {
            let moneyFlowMultiplier = ((closes[j] - lows[j]) - (highs[j] - closes[j])) / (highs[j] - lows[j]);
            let moneyFlowVolume = moneyFlowMultiplier * volumes[j];

            sumMoneyFlowVolume += moneyFlowVolume;
            sumVolume += volumes[j];
        }

        cmf.push(sumMoneyFlowVolume / sumVolume);
    }

    return cmf;
}

// Commodity Channel Index (CCI)
function calculateCCI(highs, lows, closes, period = 20) {
    let cci = [];

    for (let i = period - 1; i < highs.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += (highs[j] + lows[j] + closes[j]) / 3;
        }
        let sma = sum / period;

        let meanDeviation = 0;
        for (let j = i - period + 1; j <= i; j++) {
            meanDeviation += Math.abs(((highs[j] + lows[j] + closes[j]) / 3) - sma);
        }
        meanDeviation /= period;

        let typicalPrice = (highs[i] + lows[i] + closes[i]) / 3;
        cci.push((typicalPrice - sma) / (0.015 * meanDeviation));
    }

    return cci;
}

function calculateVolatility(candles) {
    const closes = candles.map(c => c.close);
    const returns = closes.slice(1).map((close, i) => (close - closes[i]) / closes[i]);
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);

    // Konversi ke persen
    const volatilityPercent = volatility * 100;

    // Kategorikan volatilitas
    let category = 'Low';
    if (volatilityPercent >= 1 && volatilityPercent < 2) {
        category = 'Medium';
    } else if (volatilityPercent >= 2) {
        category = 'High';
    }

    return {
        category,
        percentage: parseFloat(volatilityPercent.toFixed(2))
    };
}


function calculateTrendStrength(closes) {
    const periods = [14, 50, 200];
    return periods.map(period => {
        const sma = calculateSMA(closes, period);
        const lastSMA = sma[sma.length - 1];
        const currentPrice = closes[closes.length - 1];
        return (currentPrice - lastSMA) / lastSMA;
    });
}

function calculateVolumeProfile(candles) {
    const priceRange = {
        min: Math.min(...candles.map(c => c.low)),
        max: Math.max(...candles.map(c => c.high))
    };
    const bins = 10;
    const binSize = (priceRange.max - priceRange.min) / bins;

    const profile = new Array(bins).fill(0);
    candles.forEach(candle => {
        const binIndex = Math.floor((candle.close - priceRange.min) / binSize);
        profile[binIndex] += candle.volume;
    });

    return profile;
}

function determineMarketPhase(closes, volumes) {
    const recentCloses = closes.slice(-30);
    const recentVolumes = volumes.slice(-30);
    const priceChange = (recentCloses[recentCloses.length - 1] - recentCloses[0]) / recentCloses[0];
    const volumeChange = (recentVolumes[recentVolumes.length - 1] - recentVolumes[0]) / recentVolumes[0];

    if (priceChange > 0.05 && volumeChange > 0.1) return "Accumulation";
    if (priceChange > 0.1 && volumeChange > 0.2) return "Mark-Up";
    if (priceChange < -0.05 && volumeChange > 0.1) return "Distribution";
    if (priceChange < -0.1 && volumeChange > 0.2) return "Mark-Down";
    return "Consolidation";
}

function analyzeMarketStructure(candles, indicators, currentPrice) {
    return {
        supportResistance: findSupportResistanceLevels(candles, indicators, currentPrice),
        fvg: identifyFVG(candles, indicators),
        liquidityAreas: calculateLiquidityAreas(candles, indicators),
        orderBlocks: identifyOrderBlocks(candles, indicators),
        trendLines: calculateTrendLines(candles),
        pivotPoints: calculatePivotPoints(candles),
        candles
    };
}

function findSupportResistanceLevels(candles, indicators, currentPrice) {
    const lookback = 100;
    const levels = candles.slice(lookback, -lookback).reduce((acc, _, i) => {
        const windowStart = i;
        const windowEnd = i + lookback * 2;
        const currentCandle = candles[i + lookback];

        const isSupport = candles.slice(windowStart, windowEnd).every(c => c.low >= currentCandle.low);
        const isResistance = candles.slice(windowStart, windowEnd).every(c => c.high <= currentCandle.high);

        if (isSupport && currentCandle.low < currentPrice) {
            acc.supports.push(createSupportResistanceObject(currentCandle, 'support', candles, i + lookback, indicators));
        }
        if (isResistance && currentCandle.high > currentPrice) {
            acc.resistances.push(createSupportResistanceObject(currentCandle, 'resistance', candles, i + lookback, indicators));
        }

        return acc;
    }, { supports: [], resistances: [] });

    return levels;
}

function createSupportResistanceObject(candle, type, allCandles, index, indicators) {
    return {
        price: type === 'support' ? candle.low : candle.high,
        strength: calculateLevelStrength(allCandles, index, type),
        rsi: indicators.rsi[index] || null,
        atr: indicators.atr[index] || null,
        adx: indicators.adx[index]?.adx || null
    };
}

function calculateLevelStrength(candles, index, type) {
    const touchCount = candles.filter(c =>
        type === 'support'
            ? Math.abs(c.low - candles[index].low) < candles[index].low * 0.001
            : Math.abs(c.high - candles[index].high) < candles[index].high * 0.001
    ).length;
    return touchCount / candles.length;
}

function identifyFVG(candles, indicators) {
    return candles.slice(2).reduce((fvgAreas, currentCandle, i) => {
        const previousCandle = candles[i + 1];
        const indicatorIndex = i + 1; // Adjust index for sliced array

        // Toleransi untuk gap harga
        const tolerance = 0.005 * previousCandle.high;

        if (currentCandle.low > previousCandle.high + tolerance && previousCandle.close < previousCandle.open) {
            fvgAreas.push(createFVGObject('Bullish', currentCandle, previousCandle, indicators, indicatorIndex));
        } else if (currentCandle.high < previousCandle.low - tolerance && previousCandle.close > previousCandle.open) {
            fvgAreas.push(createFVGObject('Bearish', currentCandle, previousCandle, indicators, indicatorIndex));
        }

        return fvgAreas;
    }, []);
}
function createFVGObject(type, currentCandle, previousCandle, indicators, index) {
    const low = type === 'Bullish' ? previousCandle.high : currentCandle.high;
    const high = type === 'Bullish' ? currentCandle.low : previousCandle.low;
    return {
        type: `${type} FVG`,
        low,
        high,
        strength: Math.abs(high - low) / low,
        volume: currentCandle.volume,
        rsi: indicators.rsi[index],
        macdHistogram: indicators.macd[index]?.histogram,
        adx: indicators.adx[index]?.adx
    };
}

function calculateLiquidityAreas(candles, indicators) {
    const averageVolume = candles.reduce((sum, candle) => sum + candle.volume, 0) / candles.length;
    const highVolumeThreshold = 2 * averageVolume;
    const mediumVolumeThreshold = 1.5 * averageVolume;

    return candles.filter(candle => candle.volume > mediumVolumeThreshold)
        .map((candle, index) => ({
            priceRange: `${candle.low} - ${candle.high}`,
            low: candle.low,
            high: candle.high,
            totalVolume: candle.volume,
            impact: candle.volume > highVolumeThreshold ? "High Impact" : "Medium Impact",
            obv: indicators.obv[index - 1] || null,
            bollingerPosition: calculateBollingerPosition(candle, indicators.bb[index - 1]),
            adx: indicators.adx[index - 1]?.adx || null
        }));
}

function calculateBollingerPosition(candle, bb) {
    if (!bb) return null;
    return (candle.close - bb.lower) / (bb.upper - bb.lower);
}

function identifyOrderBlocks(candles, indicators) {
    const orderBlocks = [];
    for (let i = 2; i < candles.length; i++) {
        const currentCandle = candles[i];
        const previousCandle = candles[i - 1];
        const twoCandlesAgo = candles[i - 2];

        if (currentCandle.close > previousCandle.high && twoCandlesAgo.close < previousCandle.low) {
            orderBlocks.push({
                type: 'Bullish',
                low: previousCandle.low,
                high: previousCandle.high,
                strength: (currentCandle.close - previousCandle.high) / previousCandle.high
            });
        } else if (currentCandle.close < previousCandle.low && twoCandlesAgo.close > previousCandle.high) {
            orderBlocks.push({
                type: 'Bearish',
                low: previousCandle.low,
                high: previousCandle.high,
                strength: (previousCandle.low - currentCandle.close) / previousCandle.low
            });
        }
    }
    return orderBlocks;
}

function calculateTrendLines(candles) {
    const highs = candles.map((c, i) => [i, c.high]);
    const lows = candles.map((c, i) => [i, c.low]);

    const highTrendLine = linearRegression(highs);
    const lowTrendLine = linearRegression(lows);

    return {
        highTrendLine: {
            slope: highTrendLine.m,
            intercept: highTrendLine.b
        },
        lowTrendLine: {
            slope: lowTrendLine.m,
            intercept: lowTrendLine.b
        }
    };
}

function calculatePivotPoints(candles, currentPrice) {
    const lastCandle = candles[candles.length - 1];
    const pivot = (lastCandle.high + lastCandle.low + lastCandle.close) / 3;
    const range = lastCandle.high - lastCandle.low;

    const levels = {
        pivot,
        r1: 2 * pivot - lastCandle.low,
        r2: pivot + range,
        r3: pivot + 2 * range,
        s1: 2 * pivot - lastCandle.high,
        s2: pivot - range,
        s3: pivot - 2 * range
    };

    // Sorot level yang paling dekat dengan harga saat ini
    const closestLevel = Object.keys(levels).reduce((closest, key) => {
        return Math.abs(levels[key] - currentPrice) < Math.abs(levels[closest] - currentPrice) ? key : closest;
    }, 'pivot');

    return {
        ...levels,
        closestLevel
    };
}


function calculateSMAv2(data, period) {
    if (data.length < period) {
        return null;
    }
    const slice = data.slice(-period);
    const sma = slice.reduce((sum, value) => sum + value, 0) / period;
    return sma;
}

function determineMarketDirection(candles) {
    const closes = candles.map(candle => candle.close);
    const timestamps = candles.map(candle => candle.closeTime);

    const closePrice = closes[closes.length - 1];
    const timestamp = timestamps[timestamps.length - 2];

    const sma7 = calculateSMAv2(closes.slice(0, -1), 7);
    const sma25 = calculateSMAv2(closes.slice(0, -1), 25);
    const sma99 = calculateSMAv2(closes.slice(0, -1), 99);

    const sma7Percentage = Math.abs(((sma7 - closePrice) / closePrice) * 100);
    const sma25Percentage = Math.abs(((sma25 - closePrice) / closePrice) * 100);
    const sma99Percentage = Math.abs(((sma99 - closePrice) / closePrice) * 100);

    let marketDirection;
    if (sma7Percentage < 1.5 && sma25Percentage < 1.5 && sma99Percentage > 10) {
        marketDirection = sma7 < sma25 ? 'Long' : (sma7 > sma25 ? 'Short' : 'Neutral');
    } else {
        marketDirection = 'Neutral';
    }

    return {
        directionx: marketDirection,
        sma7,
        sma25,
        sma99,
        sma7Percentage,
        sma25Percentage,
        sma99Percentage
    };
}


function determineMarketStructure(candles) {
    const recentCandles = candles.slice(-20);
    const highs = recentCandles.map(c => c.high);
    const lows = recentCandles.map(c => c.low);
    return (highs[highs.length - 1] > Math.max(...highs.slice(0, -1)) &&
            lows[lows.length - 1] > Math.max(...lows.slice(0, -1))) ? "Uptrend" : "Downtrend";
}


function determineDirection(currentPrice, indicators, marketDirection) {
    const { sma, ema, rsi, macd, stochastic, ichimoku, parabolicSAR, cmf, cci } = indicators;
    const directions = [
        currentPrice > sma[sma.length - 1] ? "Long" : "Short",
        currentPrice > ema[ema.length - 1] ? "Long" : "Short",
        rsi[rsi.length - 1] > 50 ? "Long" : "Short",
        macd[macd.length - 1].histogram > 0 ? "Long" : "Short",
        stochastic[stochastic.length - 1].k > stochastic[stochastic.length - 1].d ? "Long" : "Short",
        currentPrice > ichimoku[ichimoku.length - 1].spanA ? "Long" : "Short",
        marketDirection.directionx,

        currentPrice > parabolicSAR[parabolicSAR.length - 1] ? "Long" : "Short",
        cmf[cmf.length - 1] > 0 ? "Long" : "Short",
        cci[cci.length - 1] > 0 ? "Long" : "Short"
    ];
    const longCount = directions.filter(d => d === "Long").length;
    return longCount > directions.length / 2 ? "Long" : "Short";
}





function calculateEntryPrice(direction, currentPrice, supportResistance, indicators, orderBlocks, volatility) {
    const { supports, resistances } = supportResistance;
    const { atr, bb } = indicators;
    const lastAtr = atr[atr.length - 1];

    let potentialEntries = [];
    if (direction === "Long") {
        potentialEntries = [
            ...supports.map(s => s.price),
            currentPrice - lastAtr * volatility,
            bb[bb.length - 1].lower,
            ...orderBlocks.filter(ob => ob.type === 'Bullish').map(ob => ob.high)
        ].filter(price => price > 0 && price <= currentPrice);
    } else {
        potentialEntries = [
            ...resistances.map(r => r.price),
            currentPrice + lastAtr * volatility,
            bb[bb.length - 1].upper,
            ...orderBlocks.filter(ob => ob.type === 'Bearish').map(ob => ob.low)
        ].filter(price => price >= currentPrice);
    }

    return direction === "Long" ? Math.max(...potentialEntries) : Math.min(...potentialEntries);
}


function calculateExitLevels(direction, entry, indicators, supportResistance, pivotPoints) {
    const { atr } = indicators;
    const lastAtr = atr[atr.length - 1];
    const slDistance = 1.5 * lastAtr;

    const stopLoss = direction === "Long" ? Math.max(entry - slDistance, 0) : entry + slDistance;

    let takeProfits = [
        direction === "Long" ? entry + 2 * slDistance : Math.max(entry - 2 * slDistance, 0),
        direction === "Long" ? entry + 3 * slDistance : Math.max(entry - 3 * slDistance, 0),
        direction === "Long" ? entry + 5 * slDistance : Math.max(entry - 5 * slDistance, 0)
    ];

    if (direction === "Long") {
        const ceilings = [
            ...supportResistance.resistances.map(r => r.price),
            pivotPoints.r1,
            pivotPoints.r2,
            pivotPoints.r3
        ].sort((a, b) => a - b);

        takeProfits = takeProfits.map((tp, index) => {
            const ceiling = ceilings.find(c => c > tp);
            return ceiling ? Math.min(tp, ceiling) : tp;
        });
    } else {
        const floors = [
            ...supportResistance.supports.map(s => s.price),
            pivotPoints.s1,
            pivotPoints.s2,
            pivotPoints.s3
        ].sort((a, b) => b - a);

        takeProfits = takeProfits.map((tp, index) => {
            const floor = floors.find(f => f < tp);
            return floor ? Math.max(tp, floor) : tp;
        });
    }

    return { stopLoss, takeProfits };
}

function findNearestLiquidityArea(liquidityAreas, entryPrice) {
    return liquidityAreas.reduce((nearest, current) => {
        const currentDistance = Math.min(Math.abs(current.low - entryPrice), Math.abs(current.high - entryPrice));
        const nearestDistance = Math.min(Math.abs(nearest.low - entryPrice), Math.abs(nearest.high - entryPrice));
        return currentDistance < nearestDistance ? current : nearest;
    });
}




function calculatePriceDominance(candles) {
    const closes = candles.map(c => c.close);
    const volumes = candles.map(c => c.volume);
    const priceChanges = closes.map((close, i) => i === 0 ? 0 : close - closes[i - 1]);

    const totalVolume = volumes.reduce((sum, volume) => sum + volume, 0);

    if (totalVolume === 0) {
        return {
            buyerPercentage: 0,
            sellerPercentage: 0,
            dominance: 'Neutral'
        };
    }

    const buyerDominance = priceChanges.reduce((sum, change, i) => {
        return sum + (change > 0 ? volumes[i] : 0);
    }, 0);

    const sellerDominance = priceChanges.reduce((sum, change, i) => {
        return sum + (change < 0 ? volumes[i] : 0);
    }, 0);

    const buyerPercentage = (buyerDominance / totalVolume) * 100;
    const sellerPercentage = (sellerDominance / totalVolume) * 100;

    const dominance = buyerPercentage > sellerPercentage ? 'Buyer' : 'Seller';

    return {
        buyerPercentage,
        sellerPercentage,
        dominance
    };
}




function generateTradeOutput(symbol, timeframe, currentPrice, analysisData, indicators,marketDirection,volatility,wibTime) {

    const { supportResistance, fvg, liquidityAreas, orderBlocks, trendLines, pivotPoints, candles } = analysisData;
    const marketStructure = determineMarketStructure(candles);
    const direction = determineDirection(currentPrice, indicators,marketDirection);
    const entry = calculateEntryPrice(direction, currentPrice, supportResistance, indicators, orderBlocks);
    const { stopLoss, takeProfits } = calculateExitLevels(direction, entry, indicators, supportResistance, pivotPoints);
    const nearestLiquidityArea = findNearestLiquidityArea(liquidityAreas, entry);
    const priceDominance = calculatePriceDominance(candles);

    const reasoning = generateReasoning(currentPrice, indicators, marketStructure, direction, marketDirection);

    return {
        symbol,
        timeframe,
        currentPrice,
        direction,
        marketStructure,
        entryPrice: entry,
        stopLoss,
        takeProfits,
        priceDominance,
        marketDirection,
        reasoning: reasoning.text,
        reasoningPercentage: reasoning.overallPercentage,
        nearestLiquidityArea,
        fvg,
        orderBlocks,
        trendLines,
        volatility,
        wibTime,
        pivotPoints
    };
}

function generateReasoning(currentPrice, indicators, marketStructure, direction, marketDirection) {
    const { sma, ema, rsi, macd, obv, atr, stochastic, ichimoku, bb, adx, vwap, trendStrength, volumeProfile, marketPhase, parabolicSAR, cmf, cci } = indicators;

    let totalWeightedScore = 0;
    let totalWeight = 0;

    // Helper function for Bollinger Band position calculation (assumed it was defined elsewhere)
    function calculateBollingerPosition(price, band) {
        const range = band.upper - band.lower;
        return range !== 0 ? (price - band.lower) / range : 0.5;
    }

    // Create structured analysis points
    const analysisPoints = [
        {
            indicator: "Market Structure",
            value: marketStructure,
            description: `Market structure shows a ${marketStructure.toLowerCase()}`,
            weight: 3,
            score: marketStructure === direction ? 1 : 0,
            supports: marketStructure === direction
        },
        {
            indicator: "SMA",
            value: sma.length > 0 ? sma[sma.length - 1] : null,
            description: sma.length > 0 ? `Price is ${currentPrice > sma[sma.length - 1] ? "above" : "below"} SMA` : "SMA data unavailable",
            weight: 2,
            score: sma.length > 0 ? ((direction === 'Long' && currentPrice > sma[sma.length - 1]) || (direction === 'Short' && currentPrice < sma[sma.length - 1]) ? 1 : 0) : null,
            supports: sma.length > 0 ? ((direction === 'Long' && currentPrice > sma[sma.length - 1]) || (direction === 'Short' && currentPrice < sma[sma.length - 1])) : null
        },
        {
            indicator: "EMA",
            value: ema.length > 0 ? ema[ema.length - 1] : null,
            description: ema.length > 0 ? `Price is ${currentPrice > ema[ema.length - 1] ? "above" : "below"} EMA` : "EMA data unavailable",
            weight: 2,
            score: ema.length > 0 ? ((direction === 'Long' && currentPrice > ema[ema.length - 1]) || (direction === 'Short' && currentPrice < ema[ema.length - 1]) ? 1 : 0) : null,
            supports: ema.length > 0 ? ((direction === 'Long' && currentPrice > ema[ema.length - 1]) || (direction === 'Short' && currentPrice < ema[ema.length - 1])) : null
        },
        {
            indicator: "RSI",
            value: rsi.length > 0 ? rsi[rsi.length - 1] : null,
            condition: rsi.length > 0 ? (rsi[rsi.length - 1] > 70 ? "overbought" : rsi[rsi.length - 1] < 30 ? "oversold" : "neutral") : null,
            description: rsi.length > 0 ? `RSI is at ${rsi[rsi.length - 1].toFixed(2)}, indicating ${rsi[rsi.length - 1] > 70 ? "overbought" : rsi[rsi.length - 1] < 30 ? "oversold" : "neutral"} conditions` : "RSI data unavailable",
            weight: 2.5,
            score: rsi.length > 0 ? ((direction === 'Long' && rsi[rsi.length - 1] < 70) || (direction === 'Short' && rsi[rsi.length - 1] > 30) ? 1 : 0) : null,
            supports: rsi.length > 0 ? ((direction === 'Long' && rsi[rsi.length - 1] < 70) || (direction === 'Short' && rsi[rsi.length - 1] > 30)) : null
        },
        {
            indicator: "MACD",
            value: macd.length > 0 ? macd[macd.length - 1].histogram : null,
            condition: macd.length > 0 ? (macd[macd.length - 1].histogram > 0 ? "positive" : "negative") : null,
            description: macd.length > 0 ? `MACD histogram is ${macd[macd.length - 1].histogram > 0 ? "positive" : "negative"}, suggesting ${macd[macd.length - 1].histogram > 0 ? "long" : "short"} momentum` : "MACD data unavailable",
            weight: 3,
            score: macd.length > 0 ? ((direction === 'Long' && macd[macd.length - 1].histogram > 0) || (direction === 'Short' && macd[macd.length - 1].histogram < 0) ? 1 : 0) : null,
            supports: macd.length > 0 ? ((direction === 'Long' && macd[macd.length - 1].histogram > 0) || (direction === 'Short' && macd[macd.length - 1].histogram < 0)) : null
        },
        {
            indicator: "Stochastic",
            value: stochastic.length > 0 ? { k: stochastic[stochastic.length - 1].k, d: stochastic[stochastic.length - 1].d } : null,
            condition: stochastic.length > 0 ? (stochastic[stochastic.length - 1].k > stochastic[stochastic.length - 1].d ? "overbought" : "oversold") : null,
            description: stochastic.length > 0 ? `Stochastic is ${stochastic[stochastic.length - 1].k > stochastic[stochastic.length - 1].d ? "overbought" : "oversold"}` : "Stochastic data unavailable",
            weight: 2,
            score: stochastic.length > 0 ? ((direction === 'Long' && stochastic[stochastic.length - 1].k > stochastic[stochastic.length - 1].d) || (direction === 'Short' && stochastic[stochastic.length - 1].k < stochastic[stochastic.length - 1].d) ? 1 : 0) : null,
            supports: stochastic.length > 0 ? ((direction === 'Long' && stochastic[stochastic.length - 1].k > stochastic[stochastic.length - 1].d) || (direction === 'Short' && stochastic[stochastic.length - 1].k < stochastic[stochastic.length - 1].d)) : null
        },
        {
            indicator: "Ichimoku",
            value: ichimoku.length > 0 ? ichimoku[ichimoku.length - 1].spanA : null,
            condition: ichimoku.length > 0 ? (currentPrice > ichimoku[ichimoku.length - 1].spanA ? "bullish" : "bearish") : null,
            description: ichimoku.length > 0 ? `Ichimoku cloud is ${currentPrice > ichimoku[ichimoku.length - 1].spanA ? "bullish" : "bearish"}` : "Ichimoku data unavailable",
            weight: 2.5,
            score: ichimoku.length > 0 ? ((direction === 'Long' && currentPrice > ichimoku[ichimoku.length - 1].spanA) || (direction === 'Short' && currentPrice < ichimoku[ichimoku.length - 1].spanA) ? 1 : 0) : null,
            supports: ichimoku.length > 0 ? ((direction === 'Long' && currentPrice > ichimoku[ichimoku.length - 1].spanA) || (direction === 'Short' && currentPrice < ichimoku[ichimoku.length - 1].spanA)) : null
        },
        {
            indicator: "ADX",
            value: adx.length > 0 ? adx[adx.length - 1].adx : null,
            condition: adx.length > 0 ? (adx[adx.length - 1].adx > 25 ? "strong" : "weak") : null,
            description: adx.length > 0 ? `ADX is at ${adx[adx.length - 1].adx.toFixed(2)}, indicating ${adx[adx.length - 1].adx > 25 ? "strong" : "weak"} trend` : "ADX data unavailable",
            weight: 2.5,
            score: adx.length > 0 ? (adx[adx.length - 1].adx > 25 ? 1 : 0) : null,
            supports: adx.length > 0 ? adx[adx.length - 1].adx > 25 : null
        },
        {
            indicator: "Bollinger Bands",
            value: bb.length > 0 ? bb[bb.length - 1] : null,
            position: bb.length > 0 ? calculateBollingerPosition(currentPrice, bb[bb.length - 1]) : null,
            description: bb.length > 0 ? `Current price is ${calculateBollingerPosition(currentPrice, bb[bb.length - 1]) < 0.5 ? "in the lower" : "in the upper"} half of the Bollinger Bands` : "Bollinger Bands data unavailable",
            weight: 2,
            score: bb.length > 0 ? ((direction === 'Long' && calculateBollingerPosition(currentPrice, bb[bb.length - 1]) > 0.5) || (direction === 'Short' && calculateBollingerPosition(currentPrice, bb[bb.length - 1]) < 0.5) ? 1 : 0) : null,
            supports: bb.length > 0 ? ((direction === 'Long' && calculateBollingerPosition(currentPrice, bb[bb.length - 1]) > 0.5) || (direction === 'Short' && calculateBollingerPosition(currentPrice, bb[bb.length - 1]) < 0.5)) : null
        },
        {
            indicator: "VWAP",
            value: vwap.length > 0 ? vwap[vwap.length - 1] : null,
            description: vwap.length > 0 ? `VWAP is at ${vwap[vwap.length - 1].toFixed(2)}` : "VWAP data unavailable",
            weight: 2,
            score: vwap.length > 0 ? ((direction === 'Long' && currentPrice > vwap[vwap.length - 1]) || (direction === 'Short' && currentPrice < vwap[vwap.length - 1]) ? 1 : 0) : null,
            supports: vwap.length > 0 ? ((direction === 'Long' && currentPrice > vwap[vwap.length - 1]) || (direction === 'Short' && currentPrice < vwap[vwap.length - 1])) : null
        },
        {
            indicator: "Trend Strength",
            value: trendStrength.length > 0 ? trendStrength[trendStrength.length - 1] : null,
            description: trendStrength.length > 0 ? `Trend strength is ${trendStrength[trendStrength.length - 1].toFixed(2)}` : "Trend strength data unavailable",
            weight: 2.5,
            score: trendStrength.length > 0 ? (trendStrength[trendStrength.length - 1] > 0 ? 1 : 0) : null,
            supports: trendStrength.length > 0 ? trendStrength[trendStrength.length - 1] > 0 : null
        },
        {
            indicator: "Volume Profile",
            value: volumeProfile.length > 0 ? volumeProfile[volumeProfile.length - 1] : null,
            description: volumeProfile.length > 0 ? `Volume profile indicates ${volumeProfile[volumeProfile.length - 1].toFixed(2)}` : "Volume profile data unavailable",
            weight: 2,
            score: null,
            supports: null
        },
        {
            indicator: "Market Phase",
            value: marketPhase,
            description: marketPhase ? `Market phase is ${marketPhase}` : "Market phase data unavailable",
            weight: 3,
            score: marketPhase ? (marketPhase === direction ? 1 : 0) : null,
            supports: marketPhase ? marketPhase === direction : null
        },
        {
            indicator: "OBV",
            value: obv.length > 0 ? { current: obv[obv.length - 1], previous: obv[obv.length - 2] } : null,
            condition: obv.length > 0 ? (obv[obv.length - 1] > obv[obv.length - 2] ? "increasing" : "decreasing") : null,
            description: obv.length > 0 ? `On-Balance Volume (OBV) is ${obv[obv.length - 1] > obv[obv.length - 2] ? "increasing" : "decreasing"}` : "OBV data unavailable",
            weight: 2.5,
            score: obv.length > 0 ? ((direction === 'Long' && obv[obv.length - 1] > obv[obv.length - 2]) || (direction === 'Short' && obv[obv.length - 1] < obv[obv.length - 2]) ? 1 : 0) : null,
            supports: obv.length > 0 ? ((direction === 'Long' && obv[obv.length - 1] > obv[obv.length - 2]) || (direction === 'Short' && obv[obv.length - 1] < obv[obv.length - 2])) : null
        },
        {
            indicator: "Parabolic SAR",
            value: parabolicSAR.length > 0 ? parabolicSAR[parabolicSAR.length - 1] : null,
            condition: parabolicSAR.length > 0 ? (parabolicSAR[parabolicSAR.length - 1] < currentPrice ? "bullish" : "bearish") : null,
            description: parabolicSAR.length > 0 ? `Parabolic SAR is ${parabolicSAR[parabolicSAR.length - 1] < currentPrice ? "bullish" : "bearish"}` : "Parabolic SAR data unavailable",
            weight: 2,
            score: parabolicSAR.length > 0 ? ((direction === 'Long' && parabolicSAR[parabolicSAR.length - 1] < currentPrice) || (direction === 'Short' && parabolicSAR[parabolicSAR.length - 1] > currentPrice) ? 1 : 0) : null,
            supports: parabolicSAR.length > 0 ? ((direction === 'Long' && parabolicSAR[parabolicSAR.length - 1] < currentPrice) || (direction === 'Short' && parabolicSAR[parabolicSAR.length - 1] > currentPrice)) : null
        },
        {
            indicator: "CMF",
            value: cmf.length > 0 ? cmf[cmf.length - 1] : null,
            condition: cmf.length > 0 ? (cmf[cmf.length - 1] > 0 ? "positive" : "negative") : null,
            description: cmf.length > 0 ? `Chaikin Money Flow (CMF) is ${cmf[cmf.length - 1] > 0 ? "positive" : "negative"}` : "CMF data unavailable",
            weight: 2,
            score: cmf.length > 0 ? ((direction === 'Long' && cmf[cmf.length - 1] > 0) || (direction === 'Short' && cmf[cmf.length - 1] < 0) ? 1 : 0) : null,
            supports: cmf.length > 0 ? ((direction === 'Long' && cmf[cmf.length - 1] > 0) || (direction === 'Short' && cmf[cmf.length - 1] < 0)) : null
        },
        {
            indicator: "ATR",
            value: atr.length > 0 ? atr[atr.length - 1] : null,
            description: atr.length > 0 ? `Average True Range (ATR) is ${atr[atr.length - 1].toFixed(2)}` : "ATR data unavailable",
            weight: 1.5,
            score: null,
            supports: null
        },
        {
            indicator: "CCI",
            value: cci.length > 0 ? cci[cci.length - 1] : null,
            description: cci.length > 0 ? `Commodity Channel Index (CCI) is ${cci[cci.length - 1].toFixed(2)}` : "CCI data unavailable",
            weight: 2,
            score: cci.length > 0 ? ((direction === 'Long' && cci[cci.length - 1] > 100) || (direction === 'Short' && cci[cci.length - 1] < -100) ? 1 : 0) : null,
            supports: cci.length > 0 ? ((direction === 'Long' && cci[cci.length - 1] > 100) || (direction === 'Short' && cci[cci.length - 1] < -100)) : null
        },
        {
            indicator: "SMA Analysis Direction",
            value: marketDirection.directionx,
            description: `Market direction based on SMA analysis: ${marketDirection.directionx}`,
            weight: 3,
            score: marketDirection.directionx === direction ? 1 : 0,
            supports: marketDirection.directionx === direction
        },
        {
            indicator: "SMA7 Percentage",
            value: marketDirection.sma7Percentage,
            description: `SMA7 percentage: ${marketDirection.sma7Percentage.toFixed(2)}%`,
            weight: 1,
            score: null,
            supports: null
        },
        {
            indicator: "SMA25 Percentage",
            value: marketDirection.sma25Percentage,
            description: `SMA25 percentage: ${marketDirection.sma25Percentage.toFixed(2)}%`,
            weight: 1,
            score: null,
            supports: null
        },
        {
            indicator: "SMA99 Percentage",
            value: marketDirection.sma99Percentage,
            description: `SMA99 percentage: ${marketDirection.sma99Percentage.toFixed(2)}%`,
            weight: 1,
            score: null,
            supports: null
        }
    ];

    analysisPoints.forEach(point => {
        if (point.score !== null) {
            totalWeightedScore += point.score * point.weight;
            totalWeight += point.weight;
        }
    });

    const overallPercentage = totalWeight > 0 ? (totalWeightedScore / totalWeight) * 100 : 0;

    // Generate teks yang mudah dibaca
    const reasonTexts = analysisPoints.map(point => point.description);
    // reasonTexts.push(`Overall direction: ${direction}`);
    // reasonTexts.push(`Overall Weighted Indicator Agreement: ${overallPercentage.toFixed(2)}%`);

    return {
        text: reasonTexts.join('. '),
        overallPercentage
    };
}


