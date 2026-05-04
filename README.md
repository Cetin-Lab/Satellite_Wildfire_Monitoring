# Satellite_Wildfire_Monitoring
🌍 Space-to-Soil AI: Onboard Intelligence for Earth Observation
🚀 Overview

This repository provides a scalable AI framework for real-time monitoring of environmental events such as:

🔥 Wildfire detection
🌪️ Wind erosion monitoring

using satellite and aerial imagery.

Our approach enables onboard intelligence, allowing edge devices and satellites to process data locally and transmit only high-value insights.

🧠 Key Idea

Modern deep learning models are too computationally expensive for onboard deployment.

We address this by:

Converting large neural networks into efficient Hadamard-based models
Using knowledge distillation
Enabling real-time inference on resource-constrained platforms

⚡ Features
✅ Real-time video/image inference
✅ Tile-based large-scale image processing
✅ WHT-based efficient neural networks
✅ Edge-ready deployment pipeline
✅ Supports wildfire & environmental monitoring

🛰️ Applications
Wildfire early detection
Wind erosion monitoring
Crop stress analysis
Environmental anomaly detection

📊 Model Efficiency
Model	Parameters
Standard CNN	~14M
Ours (WHT)	~370K
Optimized	~169K

🏗️ Project Structure
models/         # Neural network architectures
utils/          # Preprocessing, tiling, visualization
inference/      # Inference scripts
configs/        # Config files
outputs/        # Results

▶️ Usage
python Satellite\ Monitoring.py \
    --input input.mp4 \
    --output outputs/
    
🧪 Data

We use publicly available NASA Earth observation datasets:

MODIS (thermal fire detection)
VIIRS (active fire monitoring)
Landsat 8/9 (multispectral imagery)

🔬 Research Contributions
Transform-domain AI using Walsh–Hadamard Transform
Multiplication-free inference
Efficient model distillation pipeline
Adaptive sensing for intelligent Earth observation

🚀 Future Work
Adaptive satellite tasking
Multi-modal sensing integration
Deployment on FPGA / edge accelerators
