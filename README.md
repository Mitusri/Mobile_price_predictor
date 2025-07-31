# 📱 Mobile Price Predictor

A machine learning-based mobile phone price prediction system that analyzes phone specifications and predicts the price range category.

## 🎯 Features

- **Multiple Interfaces**: Command-line, programmatic, and web-based interfaces
- **High Accuracy**: 88% accuracy using Random Forest algorithm
- **Real-time Predictions**: Instant price range predictions
- **Feature Importance**: Shows which specifications matter most
- **Confidence Scores**: Provides prediction confidence levels

## 📊 Dataset Overview

The system uses a mobile phone dataset with 20 features:

### Key Features:
- **RAM** (Most important - 48% importance)
- **Battery Power** (7.3% importance)
- **Pixel Resolution** (5.6% importance each)
- **Camera Specifications**
- **Connectivity Features** (Bluetooth, WiFi, 4G, etc.)

### Price Categories:
- **0**: Low Cost (0-5000)
- **1**: Medium Cost (5000-10000)
- **2**: High Cost (10000-15000)
- **3**: Very High Cost (15000+)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mobile-price-predictor.git
cd mobile-price-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Web Interface
Open your browser and go to: http://localhost:5000

## 📋 Usage Examples

### Web Interface
- Open http://localhost:5000
- Fill in the phone specifications
- Click "Predict Price Range"
- View results with confidence scores

### Programmatic Usage
```python
from app import predictor

# Example prediction
features = {
    'battery_power': 1500,
    'blue': 1,
    'clock_speed': 2.0,
    'dual_sim': 1,
    'fc': 8,
    'four_g': 1,
    'int_memory': 32,
    'm_dep': 0.7,
    'mobile_wt': 150,
    'n_cores': 4,
    'pc': 12,
    'px_height': 1080,
    'px_width': 1920,
    'ram': 2000,
    'sc_h': 12,
    'sc_w': 6,
    'talk_time': 10,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1
}

result = predictor.predict(features)
print(f"Price Category: {result['price_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📈 Feature Ranges

| Feature | Min | Max | Type |
|---------|-----|-----|------|
| Battery Power | 501 | 1998 | mAh |
| RAM | 256 | 3998 | MB |
| Internal Memory | 2 | 64 | GB |
| Clock Speed | 0.5 | 3.0 | GHz |
| Primary Camera | 0 | 20 | MP |
| Front Camera | 0 | 19 | MP |
| Pixel Height | 0 | 1960 | px |
| Pixel Width | 500 | 1998 | px |
| Mobile Weight | 80 | 200 | g |
| Talk Time | 2 | 20 | hours |

## 🔧 Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 88%
- **Features**: 20 specifications
- **Classes**: 4 price ranges
- **Training Data**: 2000 samples
- **Validation**: 20% holdout set

## 📁 File Structure

```
mobile-price-predictor/
├── app.py                    # Main Flask application
├── train.csv                 # Training dataset
├── pa_mobile_model.pkl      # Pre-trained model
├── requirements.txt          # Python dependencies
├── README.md               # This file
└── templates/
    └── index.html           # Web interface template
```

## 🎨 Generated Files

The application creates several useful files:
- `target_distribution.png` - Price range distribution
- `feature_distributions.png` - Feature histograms
- `correlation_matrix.png` - Feature correlations
- `feature_importance.png` - Model feature importance

## 🔍 Example Predictions

### High-End Phone (Price Range 3)
- RAM: 4000 MB
- Battery: 1800 mAh
- Camera: 20 MP
- 4G: Yes
- Touch Screen: Yes

### Budget Phone (Price Range 0)
- RAM: 512 MB
- Battery: 800 mAh
- Camera: 5 MP
- 4G: No
- Touch Screen: No

## 🛠️ Customization

### Adding New Features
1. Update the feature list in predictor classes
2. Retrain the model
3. Update the web interface

### Model Tuning
```python
# In app.py, modify the RandomForestClassifier parameters
rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=10,      # Control tree depth
    random_state=42
)
```

## 📊 Performance Metrics

- **Overall Accuracy**: 88%
- **Class-wise Performance**:
  - Class 0 (Low Cost): 95% precision
  - Class 1 (Medium Cost): 82% precision
  - Class 2 (High Cost): 81% precision
  - Class 3 (Very High Cost): 93% precision

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues:

1. **Model not found**: Make sure `pa_mobile_model.pkl` is in the root directory
2. **Import errors**: Install requirements with `pip install -r requirements.txt`
3. **Port already in use**: Change the port in `app.py` or kill the existing process

### Getting Help:
- Check the generated visualizations for data insights
- Review the feature importance rankings
- Test with known phone specifications

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Happy Predicting! 📱✨**

## 👨‍💻 Author

**Mitusri Mandal**
- GitHub: https://github.com/Mitusri/
- Email: mitusrimandal@gmail.com

## 🙏 Acknowledgments

Special thanks to **Mr. Raihan Mistry** for his invaluable guidance and support throughout the development of this project. 
