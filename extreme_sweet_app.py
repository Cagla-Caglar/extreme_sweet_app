import streamlit as st

import os

import time

import webbrowser

import base64

from PIL import Image

import io

import pandas as pd

import numpy as np

from lime import lime_tabular

try:

    from backend import sweetness_predictor as sp

except Exception as e:

    st.error(str(e))

    st.stop()

def load_markdown(file_path):

    try:

        with open(file_path, "r", encoding="utf-8") as file:

            return file.read()

    except FileNotFoundError:

        return "# Content Coming Soon\n\nThe content for this section is currently being prepared..."

def open_html_in_new_tab(file_path):

    if os.path.exists(file_path):

        return True, file_path

    else:

        return False, None

def apply_custom_css():

    st.markdown("""

    <style>

    html, body, [data-testid="stAppViewContainer"] {

    background-color: #FFFFFF !important;

        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;

    }

    .main .block-container {

        background-color: #FFFFFF

        padding: 2rem;

    }

    .sidebar .sidebar-content {

        background-color: #FADCE3 !important;

        color: #000000 !important;

        border-right: 2px solid #B03A5B;

        margin-top: 0px !important;

        padding-top: 0px !important;

    }

    h1, h2, h3, h4, h5 {

        color: #B03A5B !important;

        font-weight: 600;

    }

    p, li, label {

        color: #2E2E2E !important;

    }

    .stButton>button {

    background-color: #B03A5B !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 4px !important;

    padding: 0.5rem 1rem !important;

    font-weight: 500 !important;

}

    .stButton>button:hover {

        background-color: #A1344E !important;

    }

    .stSelectbox>div>div {

        background-color: #ffffff !important;

        border: 1px solid #B03A5B !important;

    }

    .stTabs [data-baseweb="tab-list"] {

        gap: 2px;

    }

    .stTabs [data-baseweb="tab"] {

        background-color: #D98E9B !important;

        color: #ffffff !important;

        border-radius: 4px 4px 0 0 !important;

    }

    .stTabs [aria-selected="true"] {

        background-color: #B03A5B !important;

    }

    .footer {

        background-color: #A1344E;

        color: #ffffff;

        padding: 15px;

        text-align: center;

        border-radius: 0 0 10px 10px;

        margin-top: 20px;

    }

    .card {

        background-color: #ffffff;

        border-radius: 8px;

        padding: 20px;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

        margin-bottom: 20px;

    }

    .dark-bg-text {

        color: #ffffff;

    }

    .app-header {

        text-align: left;

        margin-bottom: 1.5rem;

    }

    .app-header h1 {

        color: #B03A5B;

        font-weight: 600;

        letter-spacing: 0.8px;

        font-size: 2.2rem;

        margin-bottom: 0.5rem;

    }

    .stRadio>div {

        background-color: #FADCE3 !important;

        border-radius: 5px;

    }

    .stRadio label {

        color: #000000 !important;

        font-weight: 500;

    }

    .viewerframe {

        width: 100%;

        height: 800px;

        border: none;

        background-color: #ffffff;

        border-radius: 8px;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    }

    .viz-container {

        background-color: #ffffff;

        padding: 20px;

        border-radius: 8px;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

        margin-bottom: 20px;

    }

    .html-button {

        background-color: #B03A5B; 

        color: #ffffff; 

        border: none; 

        border-radius: 4px; 

        padding: 10px 15px; 

        font-weight: 500;

        cursor: pointer;

        text-align: center;

        text-decoration: none;

        display: inline-block;

        margin: 10px 0;

    }

    .html-button:hover {

        background-color: #A1344E;

    }

    .result-box {

        background-color: #ffffff;

        padding: 20px;

        border-radius: 8px;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

        margin-bottom: 20px;

    }

    .error-box {

        background-color: #fef2f2;

        border: 1px solid #fecaca;

        color: #dc2626;

        padding: 15px;

        border-radius: 8px;

        margin-bottom: 20px;

    }

    </style>

    """, unsafe_allow_html=True)

def render_breadcrumbs(main_option, sub_option=None):

    html = "<div style='margin-bottom: 20px; padding: 10px; background-color: #ffffff; border-radius: 5px;'>"

    html += "<a href='#' style='color: #B03A5B; text-decoration: none;'>Home</a>"

    if main_option:

        html += f" > <a href='#' style='color: #B03A5B; text-decoration: none;'>{main_option}</a>"

    if sub_option:

        html += f" > <a href='#' style='color: #B03A5B; text-decoration: none;'>{sub_option}</a>"

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

def display_loading_animation():

    st.markdown("""

    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">

        <div class="spinner" style="border: 4px solid rgba(0, 0, 0, 0.1); width: 36px; height: 36px; border-radius: 50%; border-left-color: #B03A5B; animation: spin 1s linear infinite;"></div>

    </div>

    <style>

        @keyframes spin {

            0% { transform: rotate(0deg); }

            100% { transform: rotate(360deg); }

        }

    </style>

    """, unsafe_allow_html=True)

    time.sleep(0.5)

def get_local_url(file_path):

    return f"file://{os.path.abspath(file_path)}"

def generate_lime_explanation(smiles):
    try:
        exp = sp.get_lime_explanation_xgboost(smiles, num_features=25)

        timestamp = int(time.time())
        html_filename = f"lime_explanation_{timestamp}.html"
        exp.save_to_file(html_filename)

        with open(html_filename, 'r') as file:
            html_content = file.read()

        return html_content, html_filename
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        return None, None

def render_sweetness_predictor():

    st.markdown("<h2 style='color: #B03A5B;'>Sweetness Prediction System</h2>", unsafe_allow_html=True)

    st.markdown("<p>Please enter a valid SMILES string to analyze sweetness probability and predicted intensity.</p>", unsafe_allow_html=True)

    st.markdown("<p><strong>Example SMILES:</strong> CCO (ethanol), CC(=O)O (acetic acid), c1ccccc1 (benzene)</p>", unsafe_allow_html=True)

    

    with st.form(key="sweetness_prediction_form"):

        smiles_input = st.text_input("SMILES String", "", key="smiles_input_field")

        

        predict_button = st.form_submit_button(

            "Predict Sweetness",

            use_container_width=False,

            help="Click to predict sweetness for the entered SMILES string"

        )

        

        st.markdown(

            """

            <style>

            .stButton>button {

                background-color: #B03A5B !important;

                color: #ffffff !important;

                border: none !important;

                border-radius: 4px !important;

                padding: 0.5rem 1rem !important;

                font-weight: 500 !important;

            }

            .stButton>button:hover {

                background-color: #A1344E !important;

            }

            </style>

            """, 

            unsafe_allow_html=True

        )

        

    if predict_button:

        if not smiles_input:

            st.warning("Please enter a SMILES string.")

            return

            

        if not sp.validate_smiles(smiles_input):

            st.markdown("""

            <div class='error-box'>

                <h4>❌ Invalid SMILES String</h4>

                <p><strong>The entered string is not a valid SMILES representation.</strong></p>

                <p>Please check your input and ensure it follows SMILES notation rules:</p>

                <ul>

                    <li>Use standard chemical symbols (C, N, O, S, etc.)</li>

                    <li>Properly balanced parentheses and brackets</li>

                    <li>Valid bond representations (=, #, etc.)</li>

                    <li>Correct ring closure numbers</li>

                </ul>

                <p><strong>Examples of valid SMILES:</strong></p>

                <ul>

                    <li>CCO (ethanol)</li>

                    <li>CC(=O)O (acetic acid)</li>

                    <li>c1ccccc1 (benzene)</li>

                    <li>CC(C)CO (2-methyl-1-propanol)</li>

                </ul>

            </div>

            """, unsafe_allow_html=True)

            return

        

        try:

            with st.spinner("Molecular feature calculation and prediction processes are in progress. This may take a short while; please wait."):

                prediction_result, clf_features = sp.get_sweetness_prediction(smiles_input)

                ad_warnings = prediction_result.get("ad_warnings", [])
                p_sweet = prediction_result["sweet_probability"]
                is_sweet = prediction_result["is_sweet"]

                if p_sweet is None:
                    for w in ad_warnings:
                        st.warning(w)
                    return

                sweet_percentage = p_sweet * 100

                st.markdown("<h3 style='color: #B03A5B;'>Classification Result</h3>", unsafe_allow_html=True)
                st.markdown("<div class='result-box'><p><b>Model:</b> Voting Ensemble Classifier</p></div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"<p><b>Sweetness Probability:</b> {sweet_percentage:.2f}%</p>", unsafe_allow_html=True)
                    if is_sweet:
                        st.markdown("<p><b>Prediction:</b> <span style='color:#15803d'>Sweet compound</span></p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p><b>Prediction:</b> <span style='color:#b91c1c'>Non-sweet compound</span></p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                if is_sweet:
                    st.markdown("<h3 style='color: #B03A5B;'>Sweetness Intensity Prediction</h3>", unsafe_allow_html=True)
                    ensemble_sweetness = prediction_result["ensemble_sweetness"]
                    xgboost_sweetness = prediction_result["xgboost_sweetness"]
                    ensemble_sweetness_original = prediction_result["ensemble_sweetness_original_scale"]
                    xgboost_sweetness_original = prediction_result["xgboost_sweetness_original_scale"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("<p><b>Voting Ensemble Model Prediction</b></p>", unsafe_allow_html=True)
                        if ensemble_sweetness is not None:
                            st.markdown(f"<p>Sweetness Intensity (log10): {ensemble_sweetness:.2f}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p>Sweetness Intensity (Original Scale): {ensemble_sweetness_original:.2f}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p>This compound is outside the applicability domain of this model.</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("<p><b>XGBoost Model Prediction</b></p>", unsafe_allow_html=True)
                        if xgboost_sweetness is not None:
                            st.markdown(f"<p>Sweetness Intensity (log10): {xgboost_sweetness:.2f}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p>Sweetness Intensity (Original Scale): {xgboost_sweetness_original:.2f}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p>This compound is outside the applicability domain of this model.</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if ad_warnings:
                        for w in ad_warnings:
                            st.warning(w)

                    lime_exp = prediction_result.get("xgboost_lime_explanation", None)
                    if lime_exp is not None:
                        st.markdown("<h3 style='color: #B03A5B;'>XGBoost Model Explanation</h3>", unsafe_allow_html=True)
                        try:
                            timestamp = int(time.time())
                            html_filename = f"lime_explanation_{timestamp}.html"
                            lime_exp.save_to_file(html_filename)
                            with open(html_filename, 'r') as file:
                                html_content = file.read()
                            st.components.v1.html(html_content, height=600)
                            st.markdown(
                                '<div style="background: rgba(176,58,91,0.06); border-left: 3px solid #B03A5B;'
                                ' border-radius: 8px; padding: 12px 16px; margin: 8px 0 16px;'
                                ' font-size: 13.5px; line-height: 1.6; color: #2E2E2E;">'
                                '<b style="color:#8F2F4A;">How to read this chart.</b>'
                                '<ul style="margin: 8px 0 0 18px; padding: 0;">'
                                '<li>Each bar is one molecular feature.</li>'
                                '<li>Right side increases the predicted sweetness, left side decreases it.</li>'
                                '<li>Bar length reflects the strength of the contribution.</li>'
                                '<li>The explanation is local to this molecule and is not a global importance ranking.</li>'
                                '<li>Contributions reflect statistical association, not a causal mechanism.</li>'
                                '<li>Definitions of the Mordred descriptors are available in the '
                                '<a href="https://mordred-descriptor.github.io/documentation/master/descriptors.html"'
                                ' target="_blank" rel="noopener">Mordred documentation</a>.</li>'
                                '</ul></div>',
                                unsafe_allow_html=True
                            )
                            with open(html_filename, "rb") as file:
                                btn = st.download_button(
                                    label="Download LIME Explanation",
                                    data=file,
                                    file_name=html_filename,
                                    mime="text/html"
                                )
                        except Exception as e:
                            st.error(f"Error displaying LIME explanation: {str(e)}")

        

        except Exception as e:

            st.error(f"Error processing SMILES string: {str(e)}")

            st.info("Please ensure you've entered a valid SMILES string and that all required files are in the application directory.")

def main():

    st.set_page_config(page_title="Extreme Sweet Database", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

    apply_custom_css()

    menu_options = {

        "About": {

            "main_file": "docs/about.md",

            "is_markdown": True,

            "sub_options": {}

        },

        "Extreme Sweet Database": {

            "main_file": "docs/extreme_sweet_database.md",

            "is_markdown": True,

            "sub_options": {

                "Sweeteners Database": "templates/Sweetener_Database.html"

            }

        },

        "How to Use": {

            "main_file": "docs/how_to_use.md",

            "is_markdown": True,

            "sub_options": {}

        },

        "Model Interpretability": {

            "main_file": "docs/model_interpretability.md",

            "is_markdown": True,

            "sub_options": {

                "Feature Importance Analysis (Classification)": "templates/Interactive_Feature_Importance_for_Classification.html",

                "Feature Importance Analysis (Regression)": "templates/Interactive_Feature_Importance_for_Regression.html",

                "SHAP Dependence Analysis (Regression)": "templates/interactive_shap_dependence_plots_for_regression.html",

                "LIME-Based Explanation for Sweetness Intensity": "templates/lime_for_sweetness_intensity_prediction.html",

                "Regression Analysis: Predicted vs. Observed Sweetness": "templates/Observed_vs_Predicted_Sweetness_With_Original_Scale.html"

            }

        },

        "DFT Database": {

            "main_file": "docs/dft_database.md",

            "is_markdown": True,

            "sub_options": {

                "Optimized 3D Structures": "templates/optimized_3d_structures_viewer.html",

                "Infrared Spectra of Sweeteners": "templates/Sweeteners_IR_Spectrum_Viewer.html",

                "Raman Spectra of Sweeteners": "templates/Sweeteners_Raman_Spectrum_Viewer.html",

                "Raman Activity Analysis of Sweeteners": "templates/Sweeteners_Raman_Activity_Viewer.html"

            }

        },

        "Sweetness Prediction System": {

            "main_file": None,

            "is_markdown": False,

            "is_custom_function": True,

            "custom_function": render_sweetness_predictor,

            "sub_options": {}

        },

        "Molecular Descriptor Dictionary": {

            "main_file": "docs/molecular_descriptors_dictionary.md",

            "is_markdown": True,

            "sub_options": {}

        },

        "Contact": {

            "main_file": "docs/contact.md",

            "is_markdown": True,

            "sub_options": {}

        },

        "Terms of Use": {

            "main_file": "docs/terms_of_use.md",

            "is_markdown": True,

            "sub_options": {}

        }

    }

    with st.sidebar:

        all_items = []

        for main_option, details in menu_options.items():

            all_items.append(main_option)

            for sub_option in details["sub_options"]:

                all_items.append("  " + sub_option)

        selection = st.radio("Navigation", all_items, label_visibility="collapsed")

        if selection.startswith("  "):

            sub = selection.strip()

            main = None

            for k, v in menu_options.items():

                if sub in v["sub_options"]:

                    main = k

                    path = v["sub_options"][sub]

                    break

            if main:

                st.markdown("<p style='color: #000; font-size: 0.9em;'>Adjust Visualization Height:</p>", unsafe_allow_html=True)

                vis_height = st.slider("Visualization height", 600, 1200, 800, 100, key="height_slider_sub")

                selected_main_option = main

                selected_sub_option = sub

                is_markdown = False

                is_custom_function = False

                file_path = path

            else:

                selected_main_option = None

                selected_sub_option = None

                file_path = None

                is_markdown = True

                is_custom_function = False

        else:

            selected_main_option = selection

            selected_sub_option = None

            

            if "is_custom_function" in menu_options[selection] and menu_options[selection]["is_custom_function"]:

                is_custom_function = True

                custom_function = menu_options[selection]["custom_function"]

                is_markdown = False

                file_path = None

            else:

                is_custom_function = False

                is_markdown = menu_options[selection]["is_markdown"]

                file_path = menu_options[selection]["main_file"]

                

            if not menu_options[selection]["sub_options"] and not is_custom_function:

                st.markdown("<p style='color: #000; font-size: 0.9em;'>Adjust Visualization Height:</p>", unsafe_allow_html=True)

                st.slider("Visualization height", 600, 1200, 800, 100, key="height_slider_main")

    

    st.markdown("<div class='app-header'><h1>EXTREME SWEET DATABASE</h1></div>", unsafe_allow_html=True)

    

    if selected_main_option or selected_sub_option:

        render_breadcrumbs(selected_main_option, selected_sub_option)

    

    if is_custom_function:

        custom_function()

    elif is_markdown and file_path:

        content = load_markdown(file_path)

        st.markdown(content)

    elif file_path:

        st.markdown(f"<h2>{selected_sub_option}</h2>", unsafe_allow_html=True)

        exists, path_found = open_html_in_new_tab(file_path)

        if exists:

            h = st.session_state.get("height_slider_sub", 800)

            try:

                html_content = get_file_content(file_path)

                if html_content:

                    st.components.v1.html(html_content, height=h, scrolling=True)

                    

                    with open(file_path, "rb") as file:

                        st.download_button(

                            label="Download HTML File",

                            data=file,

                            file_name=os.path.basename(file_path),

                            mime="text/html",

                            key=f"download_{os.path.basename(file_path)}"

                        )

                else:

                    st.error("Could not read the HTML file content.")

            except Exception as e:

                st.error(f"Error: {str(e)}")

        else:

            st.error("Could not find the HTML file.")

            st.markdown(

                "<div class='viz-container'>"

                "<h3 style='color: #B03A5B;'>Content Coming Soon</h3>"

                "<p>The HTML file for this visualization could not be found. It may be currently in preparation.</p>"

                "</div>",

                unsafe_allow_html=True

            )

        

    st.markdown(

        "<div class='footer'>"

        "<div style='display: flex; justify-content: space-between; align-items: center;'>"

        "<div>© 2026 Extreme Sweet Database</div>"

        "<div>Developed by Çağla Çağlar</div>"

        "</div>"

        "</div>",

        unsafe_allow_html=True

    )

def get_file_content(file_path):

    try:

        with open(file_path, "r", encoding="utf-8") as f:

            return f.read()

    except Exception:

        return None

if __name__ == "__main__":

    main()
