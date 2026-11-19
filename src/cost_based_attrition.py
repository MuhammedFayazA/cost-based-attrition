"""
Cost-based splitting for employee attrition (IBM HR data).

Compares three decision trees on the same data and the same size limits:
  1. Ordinary CART (Gini)
  2. Cost-based tree (empulse CSTreeClassifier, criterion="cost")
  3. Cost-weighted CART (Gini with class_weight)

Run from the repo root:
    python src/cost_based_attrition.py

Figures are saved to reports/figures/ and tables to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from empulse.models import CSTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "HR-Employee-Attrition.csv"
FIGS = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Costs: change these two numbers and re-run
FP_COST = 1   # false alarm: predicted Leaves, but the employee stays
FN_COST = 5   # missed leaver: predicted Stays, but the employee leaves


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    # Drop columns that carry no information
    df = pd.read_csv(DATA)
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
    # y = target (1 = left, 0 = stayed); X = inputs, with text columns turned into 0/1
    y = (df["Attrition"] == "Yes").astype(int)
    X = pd.get_dummies(df.drop(columns="Attrition"), drop_first=True).astype(int)
    # 70% training, 30% testing
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y), X


# --------------------------------------------------------------------------
# Cost helpers
# --------------------------------------------------------------------------
def node_cost(target):
    cost_if_stays = target.sum() * FN_COST                    # every leaver is missed
    cost_if_leaves = (len(target) - target.sum()) * FP_COST   # every stayer is a false alarm
    return min(cost_if_stays, cost_if_leaves)                 # the group takes the cheaper label


def gini(target):
    p = target.mean()          # share of leavers in the group
    return 2 * p * (1 - p)


def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * FP_COST + fn * FN_COST


# --------------------------------------------------------------------------
# Gini vs cost: which first split would each choose?
# --------------------------------------------------------------------------
def compare_first_splits(X_train, y_train):
    questions = {
        "OverTime = Yes":         X_train["OverTime_Yes"] == 1,
        "YearsAtCompany <= 2":    X_train["YearsAtCompany"] <= 2,
        "TotalWorkingYears <= 2": X_train["TotalWorkingYears"] <= 2,
        "JobLevel = 1":           X_train["JobLevel"] == 1,
        "MonthlyIncome <= 3000":  X_train["MonthlyIncome"] <= 3000,
    }
    rows = []
    for question, mask in questions.items():
        yes, no = y_train[mask], y_train[~mask]      # the two groups after the split
        w = len(yes) / len(y_train)                  # share of employees in the "yes" group
        rows.append({"Question": question,
                     "Group size": len(yes),
                     "Leaver rate": round(yes.mean(), 2),
                     "Gini drop": round(gini(y_train) - (w * gini(yes) + (1 - w) * gini(no)), 4),
                     "Cost drop": node_cost(y_train) - node_cost(yes) - node_cost(no)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plot the cost-based tree (sklearn's plot_tree does not accept this model)
# --------------------------------------------------------------------------
def plot_cost_tree(model, names, path):
    t = model.tree_
    plt.figure(figsize=(18, 8))
    plt.axis("off")

    def draw(node, x, y, gap):
        p = t.value[node][0][1]                                    # share of leavers in this group
        label = "LEAVES" if p * FN_COST > (1 - p) * FP_COST else "Stays"
        text = f"{t.n_node_samples[node]} employees\n{p:.0%} left\n→ {label}"
        if t.children_left[node] != -1:                            # not a leaf: add the question and draw two branches
            text = f"{names[t.feature[node]]} <= {t.threshold[node]:.1f}\n" + text
            for child, dx in [(t.children_left[node], -gap), (t.children_right[node], gap)]:
                plt.plot([x, x + dx], [y, y - 1], color="grey")
                draw(child, x + dx, y - 1, gap / 2)
        colour = "sandybrown" if label == "LEAVES" else "lightblue"
        plt.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round", fc=colour))

    draw(0, 0, 0, 4)
    plt.xlim(-8, 8)
    plt.ylim(-3.5, 0.5)
    plt.title("Cost-Based Tree   (left branch = Yes, right branch = No)")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    (X_train, X_test, y_train, y_test), X = load_data()
    print("Train:", X_train.shape, " Test:", X_test.shape)

    print("\nLabel everyone 'Stays' :", y_train.sum() * FN_COST)
    print("Label everyone 'Leaves':", (len(y_train) - y_train.sum()) * FP_COST)
    print("Cheaper choice         :", node_cost(y_train))

    # Gini vs cost on the first split
    splits = compare_first_splits(X_train, y_train)
    print("\n", splits.to_string(index=False))
    print("\nGini would choose:", splits.loc[splits["Gini drop"].idxmax(), "Question"])
    print("Cost would choose:", splits.loc[splits["Cost drop"].idxmax(), "Question"])
    splits.to_csv(RESULTS / "first_split_comparison.csv", index=False)

    # Same size settings for all three trees so the comparison is fair
    gini_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=20, random_state=42)
    gini_tree.fit(X_train, y_train)

    cost_tree = CSTreeClassifier(criterion="cost", fp_cost=FP_COST, fn_cost=FN_COST,
                                 max_depth=3, min_samples_leaf=20, random_state=42)
    cost_tree.fit(X_train, y_train)
    plot_cost_tree(cost_tree, list(X.columns), FIGS / "cost_based_tree.png")

    weighted_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=20,
                                           class_weight={0: FP_COST, 1: FN_COST}, random_state=42)
    weighted_tree.fit(X_train, y_train)

    # Compare on the test data
    models = {"Ordinary CART (Gini)": gini_tree,
              "Cost-based tree": cost_tree,
              "Cost-weighted CART": weighted_tree}
    preds = {name: model.predict(X_test) for name, model in models.items()}
    preds["No model (everyone Stays)"] = y_test * 0

    rows = []
    for name, pred in preds.items():
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({"Model": name,
                     "Accuracy": round((pred == y_test).mean(), 3),
                     "Leavers caught": tp,
                     "Leavers missed": fn,
                     "False alarms": fp,
                     "Total cost": total_cost(y_test, pred)})
    results = pd.DataFrame(rows)
    print("\n", results.to_string(index=False))
    results.to_csv(RESULTS / "model_comparison.csv", index=False)

    for name, model in models.items():
        pred = model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stays", "Leaves"],
                                                cmap="Blues", colorbar=False)
        plt.title(f"{name}\nTotal cost = {total_cost(y_test, pred)}")
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plt.savefig(FIGS / f"confusion_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.barh(results["Model"], results["Total cost"], color=["grey", "seagreen", "steelblue", "lightgrey"])
    plt.xlabel("Total cost of mistakes (lower is better)")
    plt.title("Which tree is cheapest?")
    plt.gca().invert_yaxis()
    plt.savefig(FIGS / "total_cost_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # What if a missed leaver costs more (or less)?
    rows = []
    for cost in [1, 2, 3, 5, 8, 10]:
        tree = CSTreeClassifier(criterion="cost", fp_cost=1, fn_cost=cost,
                                max_depth=3, min_samples_leaf=20, random_state=42)
        pred = tree.fit(X_train, y_train).predict(X_test)
        caught = ((pred == 1) & (y_test == 1)).sum()
        rows.append({"Cost of a missed leaver": cost,
                     "Employees flagged": pred.sum(),
                     "Leavers caught": caught,
                     "Leavers missed": y_test.sum() - caught})
    sens = pd.DataFrame(rows)
    print("\n", sens.to_string(index=False))
    sens.to_csv(RESULTS / "cost_sensitivity.csv", index=False)

    sens.plot(x="Cost of a missed leaver", y="Employees flagged", marker="o", legend=False,
              ylabel="Employees flagged", title="Higher cost of missing a leaver → the tree flags more people")
    plt.savefig(FIGS / "cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
