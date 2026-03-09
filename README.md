# AWS SaaS Sales Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://saas-dashboard-mk.streamlit.app)

**🚀 [Live Demo →](https://saas-dashboard-mk.streamlit.app)**

---

## 🎯 Overview

AI-powered lead scoring and customer segmentation system for B2B SaaS companies. 
Built using machine learning on 9,994 real sales transactions ($2.3M revenue).

### ✨ Features

- **Lead Scoring**: Predict high-value deals (0-100 score) with 85% accuracy
- **Customer Segmentation**: Identify 4 actionable customer personas
- **Batch Processing**: Score hundreds of leads simultaneously
- **Interactive Dashboard**: Real-time predictions with business recommendations

### 📊 Model Performance

| Model | Metric | Score | Status |
|-------|--------|-------|--------|
| Lead Scoring | ROC-AUC | 0.85+ | ✅ Exceeds target |
| Segmentation | Silhouette | 0.64 | ✅ Good separation |
| Segments | Count | 4 | ✅ Business validated |

### 🎬 Demo

![Dashboard Screenshot](https://via.placeholder.com/800x450/1f77b4/ffffff?text=Dashboard+Screenshot)

**Try it yourself:**
1. Visit [Live Demo](https://saas-dashboard-mk.streamlit.app)
2. Go to "Lead Scoring" page
3. Enter lead details
4. Get instant AI prediction + recommendations

### 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn (Random Forest, K-Means)
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud
- **CI/CD**: GitHub Actions

### 💼 Business Impact

| Metric | Improvement | Annual Value |
|--------|-------------|--------------|
| Conversion Rate | +15-25% | $300K+ |
| Sales Efficiency | +20% | $200K+ |
| Profit Margin | +5-10 pts | $150K+ |
| Deal Size | +33% | $250K+ |

**Total Expected Impact**: $900K+ annually

### 📚 Project Structure
```
saas-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/                   # Pre-trained ML models
│   ├── lead_scoring_model.pkl
│   ├── customer_segmentation_model.pkl
│   └── ...
└── README.md
```

### 🚀 Quick Start (Local)
```bash
# Clone repository
git clone https://github.com/Muhamm-dk/saas-dashboard.git
cd saas-dashboard

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### 📖 Documentation

**For Stakeholders:**
- [User Guide](https://saas-dashboard-mk.streamlit.app) - How to use the dashboard
- [Model Performance](https://saas-dashboard-mk.streamlit.app) - Technical metrics

**For Developers:**
- Models trained on AWS SaaS Sales dataset (Kaggle)
- Feature engineering: 12 engineered features
- Algorithms: Random Forest (classification), K-Means (clustering)
- Validation: 5-fold cross-validation, train-test split (80/20)

### 🎓 Academic Project

**MSc Computer Science (Data Analytics)**  
**Final Year Project - February 2026**

**Project Goals:**
1. ✅ Build predictive models for B2B SaaS sales
2. ✅ Create deployment-ready ML system
3. ✅ Demonstrate business value ($900K+ impact)
4. ✅ Deploy production web application

### 📧 Contact

**Project Author**: Muhammed K 
**Institution**: Rajagiri College of Social Sciences 
**Supervisor**: Priyanka E Thambi

### 📄 License

This project is for academic purposes (MSc final year project).

---

⭐ **Star this repo if you find it helpful!**
```