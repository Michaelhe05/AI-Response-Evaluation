import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/annotations.csv")

metrics = ["accuracy","relevance","clarity","actionability","tone_safety"]
model_means = df.groupby("model")[metrics].mean().round(2)

df["overall"] = df[metrics].mean(axis=1)
overall_means = df.groupby("model")["overall"].mean().round(2)

pref = df.copy()
pref["is_win"] = (pref["model"] == pref["preferred"]).astype(int)
wins = pref.groupby("model")["is_win"].sum()

summary = model_means.copy()
summary["overall"] = overall_means
summary["wins"] = wins
summary = summary.fillna(0)
summary.to_csv("summary.csv")
print("== Model Summary ==")
print(summary)

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

ax = model_means.T.plot(kind="bar", figsize=(8,5))
ax.set_ylabel("Average score (1-5)")
ax.set_title("Criterion Scores by Model")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(fig_dir/"criterion_scores.png", dpi=200)

ax2 = df.boxplot(column="overall", by="model", figsize=(6,5))
plt.suptitle("")
plt.title("Overall Score Distribution by Model")
plt.ylabel("Overall (1-5)")
plt.tight_layout()
plt.savefig(fig_dir/"overall_boxplot.png", dpi=200)

print("Saved figures to figures/")
