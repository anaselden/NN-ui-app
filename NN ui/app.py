"""
Streamlit UI for Car Evaluation Prediction
Predicts car evaluation class based on 6 features using Neural Network and SVM models
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Car Evaluation Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Car Evaluation Predictor")
st.markdown(
    """
    <style>
        /* Layout tweaks */
        .block-container {max-width: 1100px; padding-top: 1.5rem;}
        /* Card look */
        .glass-card {
            padding: 18px 20px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.2);
            background: linear-gradient(145deg, #0f172a, #111827);
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        }
        /* Metric text */
        .metric-big {font-size: 32px; font-weight: 700; margin-bottom: 4px;}
        .muted {color: #cbd5e1;}
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            padding: 10px 14px;
            font-weight: 600;
            background: linear-gradient(90deg, #06b6d4, #0ea5e9);
            color: white;
            border: none;
            box-shadow: 0 8px 16px rgba(14,165,233,0.35);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0b1224;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Predict the car evaluation class from 6 features using two trained models:
    **Neural Network (95.1% acc)** and **SVM (90.5% acc)**.
    """,
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all models and preprocessors"""
    try:
        nn_model = load_model('neural_network_model.h5')
        svm_model = joblib.load('svm_model.joblib')
        encoder = joblib.load('ordinal_encoder.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open('class_labels.pkl', 'rb') as f:
            class_labels = pickle.load(f)
        
        return nn_model, svm_model, encoder, label_encoder, scaler, feature_columns, class_labels
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please run 'save_models.py' first to save the trained models.")
        st.stop()
        return None, None, None, None, None, None, None

# Load models
nn_model, svm_model, encoder, label_encoder, scaler, feature_columns, class_labels = load_models()

# Example configurations from dataset (representing different classes)
example_configs = {
    "🚫 Unacceptable (Low Safety)": {
        'buying': 'vhigh',
        'maint': 'vhigh',
        'doors': '2',
        'persons': '2',
        'lug_boot': 'small',
        'safety': 'low',
        'expected': 'unacc'
    },
    "🚫 Unacceptable (High Price)": {
        'buying': 'high',
        'maint': 'high',
        'doors': '4',
        'persons': '2',
        'lug_boot': 'med',
        'safety': 'med',
        'expected': 'unacc'
    },
    "✅ Acceptable (Basic)": {
        'buying': 'med',
        'maint': 'med',
        'doors': '4',
        'persons': '4',
        'lug_boot': 'med',
        'safety': 'med',
        'expected': 'acc'
    },
    "✅ Acceptable (Low Price)": {
        'buying': 'low',
        'maint': 'low',
        'doors': '3',
        'persons': '4',
        'lug_boot': 'small',
        'safety': 'med',
        'expected': 'acc'
    },
    "⭐ Good (High Safety)": {
        'buying': 'med',
        'maint': 'med',
        'doors': '4',
        'persons': 'more',
        'lug_boot': 'big',
        'safety': 'high',
        'expected': 'good'
    },
    "⭐ Good (Low Price, High Safety)": {
        'buying': 'low',
        'maint': 'low',
        'doors': '5more',
        'persons': 'more',
        'lug_boot': 'big',
        'safety': 'high',
        'expected': 'good'
    },
    "🌟 Very Good (Premium)": {
        'buying': 'low',
        'maint': 'low',
        'doors': '5more',
        'persons': 'more',
        'lug_boot': 'big',
        'safety': 'high',
        'expected': 'vgood'
    },
    "🌟 Very Good (Best Value)": {
        'buying': 'low',
        'maint': 'med',
        'doors': '5more',
        'persons': 'more',
        'lug_boot': 'big',
        'safety': 'high',
        'expected': 'vgood'
    }
}

# Sidebar for input features
st.sidebar.header("📋 Input Features")

# Feature options based on Car Evaluation dataset
feature_options = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

with st.sidebar:
    st.markdown("### 🎯 Quick Examples")
    st.markdown("*Click to load example configurations:*")
    
    # Create buttons for each example
    selected_example = None
    
    # Group examples by class
    unacc_examples = {k: v for k, v in example_configs.items() if 'Unacceptable' in k}
    acc_examples = {k: v for k, v in example_configs.items() if 'Acceptable' in k}
    good_examples = {k: v for k, v in example_configs.items() if 'Good' in k and 'Very' not in k}
    vgood_examples = {k: v for k, v in example_configs.items() if 'Very Good' in k}
    
    if unacc_examples:
        st.markdown("**🚫 Unacceptable Examples:**")
        for name, config in unacc_examples.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                selected_example = config
    
    if acc_examples:
        st.markdown("**✅ Acceptable Examples:**")
        for name, config in acc_examples.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                selected_example = config
    
    if good_examples:
        st.markdown("**⭐ Good Examples:**")
        for name, config in good_examples.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                selected_example = config
    
    if vgood_examples:
        st.markdown("**🌟 Very Good Examples:**")
        for name, config in vgood_examples.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                selected_example = config
    
    st.markdown("---")
    st.markdown("### ✏️ Manual Input")
    
    # Initialize session state for inputs
    if 'buying' not in st.session_state:
        st.session_state.buying = 'med'
    if 'maint' not in st.session_state:
        st.session_state.maint = 'med'
    if 'doors' not in st.session_state:
        st.session_state.doors = '4'
    if 'persons' not in st.session_state:
        st.session_state.persons = '4'
    if 'lug_boot' not in st.session_state:
        st.session_state.lug_boot = 'med'
    if 'safety' not in st.session_state:
        st.session_state.safety = 'med'
    
    # Update session state if example is selected
    if selected_example:
        st.session_state.buying = selected_example['buying']
        st.session_state.maint = selected_example['maint']
        st.session_state.doors = selected_example['doors']
        st.session_state.persons = selected_example['persons']
        st.session_state.lug_boot = selected_example['lug_boot']
        st.session_state.safety = selected_example['safety']
        st.rerun()
    
    buying = st.selectbox("Buying Price", feature_options['buying'], 
                          index=feature_options['buying'].index(st.session_state.buying))
    maint = st.selectbox("Maintenance Price", feature_options['maint'], 
                         index=feature_options['maint'].index(st.session_state.maint))
    doors = st.selectbox("Number of Doors", feature_options['doors'], 
                        index=feature_options['doors'].index(st.session_state.doors))
    persons = st.selectbox("Number of Persons", feature_options['persons'], 
                          index=feature_options['persons'].index(st.session_state.persons))
    lug_boot = st.selectbox("Luggage Boot Size", feature_options['lug_boot'], 
                            index=feature_options['lug_boot'].index(st.session_state.lug_boot))
    safety = st.selectbox("Safety Level", feature_options['safety'], 
                         index=feature_options['safety'].index(st.session_state.safety))
    
    # Update session state when user changes values manually
    st.session_state.buying = buying
    st.session_state.maint = maint
    st.session_state.doors = doors
    st.session_state.persons = persons
    st.session_state.lug_boot = lug_boot
    st.session_state.safety = safety

st.markdown("### 🧭 Overview")
st.markdown(
    """
    Use the **Quick Examples** in the sidebar to test different car configurations, or manually adjust values below.
    Then hit **Predict** to see results from both models side-by-side with confidence scores.
    """
)

# Check if current inputs match any example
current_config = {
    'buying': buying,
    'maint': maint,
    'doors': doors,
    'persons': persons,
    'lug_boot': lug_boot,
    'safety': safety
}

matched_example = None
for name, config in example_configs.items():
    if all(current_config[key] == config[key] for key in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']):
        matched_example = (name, config['expected'])
        break

if matched_example:
    example_name, expected_class = matched_example
    st.info(f"📌 **Current Configuration:** {example_name} (Expected: {expected_class.upper()})")

# Prepare input data
input_dict = {
    'buying': [buying],
    'maint': [maint],
    'doors': [doors],
    'persons': [persons],
    'lug_boot': [lug_boot],
    'safety': [safety]
}
input_df = pd.DataFrame(input_dict)
input_encoded = encoder.transform(input_df)
input_scaled = scaler.transform(input_encoded)

predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

st.markdown("---")
st.markdown("### 📊 Your Inputs")
st.dataframe(
    pd.DataFrame({
        'Feature': ['Buying Price', 'Maintenance Price', 'Number of Doors',
                    'Number of Persons', 'Luggage Boot Size', 'Safety Level'],
        'Value': [buying, maint, doors, persons, lug_boot, safety]
    }),
    use_container_width=True,
    hide_index=True,
)

if predict_clicked:
    nn_probs = nn_model.predict(input_scaled, verbose=0)[0]
    nn_pred_idx = np.argmax(nn_probs)
    nn_pred_class = label_encoder.inverse_transform([nn_pred_idx])[0]
    nn_confidence = nn_probs[nn_pred_idx] * 100

    svm_pred_idx = svm_model.predict(input_scaled)[0]
    svm_pred_class = label_encoder.inverse_transform([svm_pred_idx])[0]

    st.markdown("---")
    st.markdown("### 🎯 Predictions")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🧠 Neural Network (MLP)**")
        st.markdown(f"<div class='metric-big'>{nn_pred_class.upper()}</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='muted'>Confidence: {nn_confidence:.1f}%</span>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            'Class': [label.upper() for label in class_labels],
            'Probability': [prob * 100 for prob in nn_probs]
        })
        st.bar_chart(prob_df.set_index('Class'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🤖 Support Vector Machine (SVM)**")
        st.markdown(f"<div class='metric-big'>{svm_pred_class.upper()}</div>", unsafe_allow_html=True)
        st.markdown("<span class='muted'>Confidence: deterministic SVM output</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Model agreement check
    if nn_pred_class == svm_pred_class:
        st.success(f"✅ Both models agree: **{nn_pred_class.upper()}**")
    else:
        st.warning(
            f"⚠️ Models disagree — NN: **{nn_pred_class.upper()}** ({nn_confidence:.1f}%), "
            f"SVM: **{svm_pred_class.upper()}**"
        )
    
    # Check if prediction matches expected class (if using example)
    if matched_example:
        expected_class = matched_example[1]
        if nn_pred_class == expected_class:
            st.success(f"🎯 **Neural Network prediction matches expected class:** {expected_class.upper()}")
        else:
            st.info(f"ℹ️ Expected class: **{expected_class.upper()}**, but NN predicted: **{nn_pred_class.upper()}**")
        
        if svm_pred_class == expected_class:
            st.success(f"🎯 **SVM prediction matches expected class:** {expected_class.upper()}")
        elif nn_pred_class != svm_pred_class:
            st.info(f"ℹ️ Expected class: **{expected_class.upper()}**, but SVM predicted: **{svm_pred_class.upper()}**")

    st.markdown("### 📖 Class Descriptions")
    class_descriptions = {
        'unacc': 'Unacceptable — does not meet minimum requirements.',
        'acc': 'Acceptable — meets basic requirements.',
        'good': 'Good — solid choice.',
        'vgood': 'Very Good — excellent, highly recommended.'
    }
    for label in class_labels:
        desc = class_descriptions.get(label, 'Unknown')
        if label == nn_pred_class:
            st.markdown(f"**{label.upper()}**: {desc} 🎯")
        else:
            st.markdown(f"{label.upper()}: {desc}")

with st.expander("ℹ️ About this app"):
    st.markdown(
        """
        - Dataset: Car Evaluation (UCI ML Repository)
        - Models: Neural Network (MLP) and SVM (RBF kernel)
        - Preprocessing: Ordinal encoding + Standard scaling
        """
    )

