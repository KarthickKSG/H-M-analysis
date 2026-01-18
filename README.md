
# 🧠 NeuroCluster: Human Stress Level Analysis

NeuroCluster is a high-performance web application designed to compare traditional and advanced clustering algorithms to understand human stress patterns.

## 🌟 Key Features
- **Algorithm Comparison**: Compare 7 different clustering techniques including K-Means, K-Medoids, DBSCAN, and HDBSCAN.
- **Advanced Visuals**: Animated 3D scatter plots and area charts powered by Plotly.
- **Dynamic Tuning**: Real-time hyperparameter adjustment via sidebar.
- **Scientific Metrics**: Automatic calculation of Silhouette Scores to measure cluster quality.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stress-clustering-app.git
   cd stress-clustering-app
Install Dependencies:
code
Bash
pip install -r requirements.txt
Run the Application:
code
Bash
streamlit run app.py
🧪 Algorithms Included
K-Means: Baseline centroid-based clustering.
K-Medoids: Exemplar-based clustering, robust to noise.
DBSCAN: Density-based clustering for spatial data.
OPTICS: Handles clusters of varying densities.
HDBSCAN: Hierarchical density clustering.
Gaussian Mixture (GMM): Distribution-based probabilistic clustering.
BIRCH: Optimized grid-based clustering for large-scale data.
📂 Data Structure
The app uses country_wise_latest.csv. The metrics are mapped as follows:
Confirmed -> Stress Trigger Frequency
Active -> Current Stress Load
Recovered -> Resilience Score
code
Code
### How to run this:
1. Ensure you have Python installed.
2. Save the code above into the respective files.
3. Run `pip install streamlit pandas scikit-learn scikit-learn-extra plotly hdbscan`.
4. Launch by typing `streamlit run app.py` in your terminal.
5. The "Extraordinary Animations" are handled by Plotly's 3D engine and the `animation_frame` parameter, which will allow you to play the data like a movie.
