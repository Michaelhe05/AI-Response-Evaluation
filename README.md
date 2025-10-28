# 🧠 AI Response Evaluation Project

This project compares two AI model outputs (A vs B) using a human feedback rubric with five criteria:
**Accuracy**, **Relevance**, **Clarity**, **Actionability**, and **Tone/Safety**.

## 📊 Features
- Annotated 12 prompts from diverse domains (IT support, cloud computing, writing, ethics)
- Python script to compute average scores and generate comparison charts
- Visual results saved in `figures/` for quick insight

## 🧰 Tech Used
- Python (pandas, matplotlib)
- CSV data and markdown documentation

## 🧩 Folder Structure
```
AI-Response-Evaluation/
├── data/
│   └── annotations.csv
├── figures/
├── analyze.py
├── rubric.md
├── report.md
└── README.md
```

## 🚀 How to Run
```bash
pip install pandas matplotlib
python analyze.py
```

## 📈 Example Output
After running `analyze.py`, charts will be saved in `figures/`.

## ✍️ Author
**Miguel Castaneda**  
- U.S. Air Force Veteran | Computer Science Student | Azure Certified (AZ-900)  
- Email: michaelhe05@gmail.com
