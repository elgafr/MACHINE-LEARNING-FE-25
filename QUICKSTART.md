# ⚡ Quick Start Guide - Vehicle Classification Dashboard

## 🚀 Run in 3 Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run dashboard
streamlit run app.py

# 3. Open browser
# → http://localhost:8501
```

## 📱 Using the Dashboard

### Step 1: Select Model (Sidebar)
Choose from 7 available models:
- **ResNet50** (Best accuracy: 92%) 🏆
- **MobileNetV2** (Fastest: 80ms) ⚡
- **EfficientNetB0** (Balanced) ⚖️
- **Base CNN** (Lightweight) 🔧

### Step 2: Upload Image
- Click "Browse files" or drag & drop
- Supported: JPG, PNG
- Recommended: Clear vehicle photo

### Step 3: Analyze
- Click "🚀 Analyze Image" button
- View prediction with confidence
- Check all category scores

## 🎯 What You'll See

### Main Prediction
```
🚗 City Car
🎯 High Confidence: 94.32%
```

### Confidence Distribution
Interactive bar chart showing all 7 categories with their confidence percentages.

### Detailed Table
Ranked list of all predictions with exact percentages.

## 💡 Quick Tips

✅ **DO:**
- Use clear, well-lit photos
- Center the vehicle in frame
- Upload single vehicle images

❌ **DON'T:**
- Use blurry/dark images
- Upload multiple vehicles
- Use images <100x100 px

## 🏆 Model Recommendations

| Need | Use This Model |
|------|----------------|
| Best Accuracy | ResNet50 (92%) |
| Fastest Speed | Base CNN (50ms) |
| Mobile Deploy | MobileNetV2 (82%) |
| Balanced | EfficientNetB0 (80%) |

## 🐛 Quick Troubleshooting

**Can't find streamlit command?**
```bash
pip install streamlit
```

**Model not loading?**
- Check `model/` folder exists
- Verify .keras or .h5 files present

**Slow performance?**
- First load takes 10-30 sec (model loading)
- Next predictions are instant (cached)

**Wrong predictions?**
- Check image quality
- Try different model
- Ensure vehicle visible

## 📊 Expected Performance

| Model | Speed | Accuracy |
|-------|-------|----------|
| ResNet50 | 120ms | 92% |
| MobileNetV2 | 80ms | 82% |
| EfficientNetB0 | 100ms | 80% |
| Base CNN | 50ms | 84% |

## 🎨 Dashboard Tabs

1. **🔍 Prediction** - Main classification interface
2. **📊 Model Comparison** - Performance metrics
3. **📖 How It Works** - Educational content


---

**That's it! You're ready to classify vehicles! 🚗✨**

