# 🧬 FlowSense: AI-Guided Flow Cytometry

**FlowSense** is an end-to-end pipeline for analyzing flow cytometry `.fcs` files using ML and GNNs. It handles everything from live gating and cell classification to anomaly detection — all wrapped in a clean Streamlit UI.

---

## 🔥 Key Features

- Upload `.fcs` and assign fluorochrome-marker mappings via UI
- Automated gating for live/singlet cells
- CD4/CD8 classification using Graph Neural Network
- Isolation Forest for anomaly detection
- Violin plots and UMAPs to explore marker expression
- Clean local app using Streamlit

---

## 🚀 Run the App

```bash
pip install -r requirements.txt
streamlit run src/ui/streamlit_app.py


---
## 🧠 Models Used

| Task                 | Model               | Output                    |
|----------------------|---------------------|---------------------------|
| Viability Gating     | Rule-based          | Singlet/live filter       |
| CD4/CD8 Classification | GNN (GraphSAGE)   | 0 = CD4+, 1 = CD8+        |
| Anomaly Detection    | Isolation Forest    | Outlier flag (0 or 1)     |
| Marker Visualization | UMAP + Seaborn      | Expression plots          |

---

## 📈 Pipeline

1. Upload `.fcs` file
2. Map fluorochrome → marker
3. Automatic live/singlet gating
4. Build kNN → Graph → Run GNN
5. Predict CD4/CD8 class
6. Detect outliers
7. Show UMAPs and marker plots

---

## 🧪 Future Directions

- Add support for batch-wise comparison
- Integrate contrastive learning + SimCLR
- Multimodal integration (e.g. scRNA + flow)
- Deploy with Docker

---

## 👩‍💻 Author

**Nidhi Dev**  
🧬 Cell Therapy | 🧠 ML-Bio Research | 💻 Pipeline Developer  
🔗 [LinkedIn](https://linkedin.com/in/nididev) • [GitHub](https://github.com/nididev)

---

## 🛡️ Disclaimer

This is a research prototype built for learning purposes. Not for clinical use.

