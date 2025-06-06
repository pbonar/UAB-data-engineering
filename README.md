# Network Intrusion Detection with Machine Learning

This project, developed as part of the Data Engineering course at Universitat Autònoma de Barcelona, explores the application of machine learning techniques to enhance real-time network intrusion detection. The project utilizes the comprehensive UNSW-NB15 dataset to implement and evaluate various algorithms for detecting and predicting cyber threats.

## 📁 Repository Structure

- `data/`: Contains the UNSW-NB15 dataset used for training and testing.
- `notebooks/`: Jupyter notebooks detailing data preprocessing, feature reduction, clustering, classification, and collaborative filtering.
- `results/`: Visualizations and evaluation metrics from the implemented models.
- `README.md`: Overview and documentation of the project.

## 📊 Dataset

The UNSW-NB15 dataset, developed by the Australian Centre for Cyber Security at UNSW, comprises over 2.5 million records, each with 49 features capturing various network traffic characteristics. The dataset includes both normal traffic and nine categories of attacks:

- Fuzzers
- Analysis
- Backdoors
- DoS
- Exploits
- Generic
- Reconnaissance
- Shellcode
- Worms

The dataset is divided into a training set of 175,341 records and a testing set of 82,332 records.

## 🧪 Methodology

### 1. Data Preparation and Feature Reduction

- **Standardization**: Applied standard scaling to normalize the data.
- **Feature Selection**: Removed non-informative features such as labels and IDs.
- **Principal Component Analysis (PCA)**: Reduced dimensionality while preserving variance:
  - 2 components explained ~70% variance.
  - 3 components explained ~80% variance.

### 2. Clustering and Classification

- **K-Means Clustering**:
  - Explored cluster sizes between 3 and 5.
  - Optimal clustering observed at K=3, distinguishing between normal and attack traffic.
- **K-Nearest Neighbors (KNN)**:
  - Applied on PCA-reduced data.
  - Achieved 93% accuracy with high precision and recall for both normal and attack categories.

### 3. Collaborative Filtering

- **Interaction Matrix**: Created to analyze associations between attack types and targeted services.
- **Similarity Analysis**:
  - Identified that certain attacks, like Backdoor and DoS, share high feature similarity.
  - Generic attacks showed low similarity with others, indicating distinct characteristics.
- **Prediction**:
  - Implemented a system to predict potential attacks on normal traffic based on nearest neighbors.
  - Demonstrated the ability to anticipate likely attack types, aiding in proactive defense strategies.

## ✅ Results

- **PCA**: Efficiently reduced data dimensionality, enhancing training speed with minimal loss of information.
- **K-Means**: Provided meaningful clustering, though required manual labeling and was sensitive to noise.
- **KNN**: Delivered high accuracy but was computationally intensive on large datasets.
- **Collaborative Filtering**: Offered novel insights into attack patterns and potential vulnerabilities, though effectiveness depended on the availability of historical data.

## 🔮 Conclusion

The integration of PCA and KNN proved most effective, balancing accuracy and computational efficiency. Collaborative filtering introduced a predictive dimension, enabling the anticipation of potential threats based on historical patterns. While each method has its limitations, their combined application enhances the robustness of intrusion detection systems.

## 🚀 Future Work

- **Model Optimization**: Explore advanced algorithms to improve detection accuracy and reduce computational load.
- **Real-Time Implementation**: Develop systems capable of real-time intrusion detection and response.
- **Dataset Expansion**: Incorporate additional datasets to improve model generalization across diverse network environments.
- **Cold-Start Problem**: Address challenges in collaborative filtering when encountering previously unseen threats.

## 👥 Authors

- **Piotr Bonar** (ID: 1759684)
- **Maddox Hurlbert** (ID: 1729809)
- **Sonia Serra** (ID: 1753004)

*Universitat Autònoma de Barcelona, May 24, 2025*
