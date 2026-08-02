# %%
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

model = joblib.load("best_xgboost_classifier_model_for_sweet_taste_class_StandardScaler.joblib")
train_df = pd.read_csv("filtered_sweet_nonsweet_class_train_set.csv")

X_train = train_df.drop(["Name", "Class"], axis=1)
y_train = train_df["Class"]

feature_names = X_train.columns.to_list()
feature_importances = np.asarray(model.feature_importances_, dtype=float)

den = feature_importances.sum()
normalized_importances = feature_importances / den if den > 0 else feature_importances

importance_df = (
    pd.DataFrame({"Feature": feature_names, "Normalized_Importance": normalized_importances})
    .sort_values("Normalized_Importance", ascending=False)
    .reset_index(drop=True)
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=importance_df.loc[:19, "Feature"],
        y=importance_df.loc[:19, "Normalized_Importance"],
        marker=dict(color=importance_df.loc[:19, "Normalized_Importance"], colorscale="Tealrose"),
    )
)

steps = []
n_features = len(importance_df)
for i in range(10, n_features + 1, 10):
    steps.append(
        dict(
            method="update",
            args=[
                {
                    "x": [importance_df.loc[: i - 1, "Feature"]],
                    "y": [importance_df.loc[: i - 1, "Normalized_Importance"]],
                    "marker.color": [importance_df.loc[: i - 1, "Normalized_Importance"]],
                },
                {"title.text": f"Top {i} Feature Importances"},
            ],
            label=str(i),
        )
    )

max_y = float(importance_df["Normalized_Importance"].max()) if n_features else 0.0

fig.update_layout(
    title=dict(text="Top 20 Feature Importances", x=0.5, y=0.95, xanchor="center", yanchor="top"),
    xaxis=dict(title="Features", tickangle=-45),
    yaxis=dict(title="Normalized Importance", range=[0, max_y * 1.05 if max_y > 0 else 1]),
    template="plotly_white",
    margin=dict(l=50, r=50, t=200, b=50),
    sliders=[
        dict(
            active=1 if len(steps) > 1 else 0,
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1.15,
            currentvalue=dict(visible=False),
            steps=steps,
        )
    ],
)

output_file = "Interactive_Feature_Importance_for_Classification.html"
fig.write_html(output_file)
print(f"Interactive feature importance chart saved as '{output_file}'.")

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset and trained model
df = pd.read_csv('filtered_sweeteners_data_307_features.csv')
best_model = load('best_xgboost_regressor_model_with_filtered_features_MinMaxScaler.joblib')

# Extract input features and target variable
X = df.iloc[:, 1:-1]
y = df['Sweetness']

# Split data into training and test sets FIRST (same as original training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Identify feature types
bit_columns = [col for col in X.columns if 'Bit' in col]
mordred_columns = [col for col in X.columns if 'Bit' not in col]

# Normalize Mordred descriptors (fit only on training data)
scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_train_scaled[mordred_columns] = scaler.fit_transform(X_train[mordred_columns])

# Compute feature importance (model's internal property - doesn't need scaled data)
feature_names = X_train.columns
feature_importances = best_model.feature_importances_

# Create DataFrame for importance values
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_df['Normalized_Importance'] = importance_df['Importance'] / np.sum(importance_df['Importance'])
importance_df = importance_df.sort_values(by='Normalized_Importance', ascending=False)

# Initialize figure
fig = go.Figure()

# Plot top 20 features
fig.add_trace(go.Bar(
    x=importance_df['Feature'][:20],
    y=importance_df['Normalized_Importance'][:20],
    marker=dict(
        color=importance_df['Normalized_Importance'][:20],
        colorscale='RdYlBu',
        showscale=True
    )
))

# Configure layout
fig.update_layout(
    title={"text": "Top 20 Most Important Features", "x": 0.5, "xanchor": "center"},
    xaxis=dict(title="Features", tickangle=-45),
    yaxis=dict(title="Normalized Importance"),
    template="plotly_white",
    margin=dict(l=50, r=50, t=100, b=50)
)

# Add interactive slider
slider_steps = []
for i in range(10, len(importance_df) + 10, 10):
    step = dict(
        method="update",
        args=[{
            "x": [importance_df['Feature'][:i]],
            "y": [importance_df['Normalized_Importance'][:i]]
        }, {
            "title.text": f"Top {i} Most Important Features"
        }],
        label=f"{i}"
    )
    slider_steps.append(step)

fig.update_layout(
    sliders=[{
        "active": 0,
        "xanchor": "center",
        "yanchor": "top",
        "x": 0.5,
        "y": 1.2,
        "currentvalue": {"font": {"size": 14}},
        "steps": slider_steps
    }]
)

# Save interactive chart
output_file = "Interactive_Feature_Importance_for_Regression.html"
fig.write_html(output_file)

print(f"Interactive feature importance chart saved as '{output_file}'.")


# %%
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap

# Load model and data
model = joblib.load('best_xgboost_regressor_model_with_filtered_features_MinMaxScaler.joblib')
df = pd.read_csv('filtered_sweeteners_data_307_features.csv')

# Prepare features and target
X = df.iloc[:, 1:-1]
y = df['Sweetness']

# Train-test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Identify feature types
bit_columns = [col for col in X.columns if 'Bit' in col]
mordred_columns = [col for col in X.columns if 'Bit' not in col]

# Scale full dataset
scaler = MinMaxScaler()
scaler.fit(X_train[mordred_columns])

X_scaled = X.copy()
X_scaled[mordred_columns] = scaler.transform(X[mordred_columns])

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Calculate mean absolute SHAP values
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
mean_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)

# Generate dependence plots
def create_dependence_plot(selected_feature):
    shap_vals = shap_values[:, X.columns.get_loc(selected_feature)]
    
    if selected_feature in mordred_columns:
        mordred_col_idx = mordred_columns.index(selected_feature)
        feature_vals = scaler.inverse_transform(X_scaled[mordred_columns])[:, mordred_col_idx]
    else:
        feature_vals = X[selected_feature].values

    fig = px.scatter(
        x=feature_vals, y=shap_vals,
        color=feature_vals,
        labels={'x': selected_feature, 'y': 'SHAP Value'},
        title=f'SHAP Dependence Plot for {selected_feature}',
        color_continuous_scale='RdBu',
        template='plotly_white'
    )

    fig.add_shape(
        type="line",
        x0=min(feature_vals), x1=max(feature_vals),
        y0=0, y1=0,
        line=dict(color="Black", width=1)
    )

    fig.update_layout(coloraxis_colorbar=dict(title="Feature Value"))
    return fig

# Build interactive HTML
dropdown_options = [{'label': feature, 'value': feature} for feature in mean_shap_values.index]
graphs_html = """
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>SHAP Dependence Plots</h1>
    <label for="feature">Select a Feature:</label>
    <select id="feature" onchange="updateGraph()">
"""

for option in dropdown_options:
    graphs_html += f'<option value="{option["value"]}">{option["label"]}</option>'

graphs_html += """
    </select>
    <div id="graph"></div>
    <script>
        const plots = {
"""

for feature in mean_shap_values.index:
    fig = create_dependence_plot(feature)
    fig_json = fig.to_json()
    graphs_html += f'"{feature}": {fig_json},'

graphs_html = graphs_html.rstrip(',') + """
        };

        function updateGraph() {
            const selectedFeature = document.getElementById("feature").value;
            const graphDiv = document.getElementById("graph");
            Plotly.newPlot(graphDiv, plots[selectedFeature].data, plots[selectedFeature].layout);
        }

        document.addEventListener("DOMContentLoaded", () => {
            updateGraph();
        });
    </script>
</body>
</html>
"""

with open("interactive_shap_dependence_plots_for_regression.html", "w", encoding="utf-8") as f:
    f.write(graphs_html)

print("SHAP dependence plots saved successfully!")


# %%
import json
import joblib
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("best_xgboost_regressor_model_with_filtered_features_MinMaxScaler.joblib")
df = pd.read_csv("filtered_sweeteners_data_307_features.csv")

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
compound_names = df.iloc[:, 0].values

X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(
    X, y, compound_names, test_size=0.3, random_state=5
)

bit_columns = [c for c in X.columns if "Bit" in c]
mordred_columns = [c for c in X.columns if "Bit" not in c]

scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[mordred_columns] = scaler.fit_transform(X_train[mordred_columns])
X_test_scaled[mordred_columns] = scaler.transform(X_test[mordred_columns])

bit_idx = [X_train_scaled.columns.get_loc(c) for c in bit_columns]
categorical_names = {i: [0, 1] for i in bit_idx}

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values,
    feature_names=X_train_scaled.columns.tolist(),
    mode="regression",
    discretize_continuous=False,
    categorical_features=bit_idx,
    categorical_names=categorical_names,
)

lime_explanations = {}
true_vs_predicted = {}

for i in range(len(X_test_scaled)):
    x_i = X_test_scaled.iloc[i].values
    exp = explainer.explain_instance(
        x_i,
        model.predict,
        num_features=X_train_scaled.shape[1],
    )

    feature_importances = exp.as_list()
    feature_names, weights = zip(*feature_importances)

    sorted_indices = np.argsort(-np.abs(np.asarray(weights, dtype=float)))
    sorted_features = [feature_names[j] for j in sorted_indices]
    sorted_weights = [float(weights[j]) for j in sorted_indices]

    key = str(compound_test[i])
    lime_explanations[key] = {"features": sorted_features, "weights": sorted_weights}

    true_vs_predicted[key] = {
        "actual_sweetness": float(y_test.iloc[i]),
        "predicted_sweetness": float(model.predict(x_i.reshape(1, -1))[0]),
    }

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LIME Explanation for Sweetness Intensity Prediction</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
        }
        #graph-container {
            width: 100%;
            height: 800px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 20px;
        }
        .info-box {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        select {
            padding: 5px;
            font-size: 16px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Local Interpretable Model-Agnostic Explanation (LIME) for Sweetness Intensity Prediction</h1>

    <div class="info-box">
        <label for="compound">Select a Compound:</label>
        <select id="compound" onchange="updateGraph()">
"""

for compound in lime_explanations.keys():
    html_content += f'<option value="{compound}">{compound}</option>'

html_content += """
        </select>
        <div id="sweetness-info"></div>
    </div>

    <div id="graph-container">
        <div id="graph"></div>
    </div>

    <script>
"""

html_content += f"const explanations = {json.dumps(lime_explanations)};\n"
html_content += f"const sweetnessData = {json.dumps(true_vs_predicted)};\n"

html_content += """
        function updateGraph() {
            const selectedCompound = document.getElementById("compound").value;
            const data = explanations[selectedCompound];
            const sweetness = sweetnessData[selectedCompound];

            document.getElementById("sweetness-info").innerHTML =
                `<p><b>Actual Sweetness Intensity:</b> ${sweetness.actual_sweetness.toFixed(2)} | <b>Predicted Sweetness Intensity:</b> ${sweetness.predicted_sweetness.toFixed(2)}</p>`
            ;

            const trace = {
                x: data.weights.slice().reverse(),
                y: data.features.slice().reverse(),
                type: "bar",
                orientation: "h",
                marker: {
                    color: data.weights.slice().reverse(),
                    colorscale: "RdBu",
                    showscale: true
                }
            };

            const layout = {
                title: {
                    text: "LIME Feature Importance Analysis",
                    font: { size: 24 }
                },
                xaxis: {
                    title: "Feature Contribution",
                    titlefont: { size: 16 }
                },
                yaxis: {
                    title: "Features",
                    titlefont: { size: 16 },
                    automargin: true
                },
                height: Math.max(800, data.features.length * 20),
                margin: { l: 300, r: 50, t: 100, b: 50 },
                template: "plotly_white"
            };

            Plotly.newPlot("graph", [trace], layout);
        }

        document.addEventListener("DOMContentLoaded", updateGraph);
    </script>
</body>
</html>
"""

output_path = "lime_for_sweetness_intensity_prediction.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"LIME interactive HTML saved as '{output_path}'")