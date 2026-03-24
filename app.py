"""
app.py — Forest Fire Detection Streamlit App
Run with: streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io

IMG_SIZE   = (128, 128)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.keras")

st.set_page_config(page_title="Forest Fire Detector", page_icon="🔥", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .fire-badge {
        background: linear-gradient(135deg, #E25822, #FF6B35);
        color: white; padding: 12px 28px; border-radius: 30px;
        font-weight: 700; font-size: 20px; display: inline-block;
        box-shadow: 0 4px 20px rgba(226,88,34,0.5);
        animation: pulse 2s infinite;
    }
    .nofire-badge {
        background: linear-gradient(135deg, #1A6B4A, #2ECC71);
        color: white; padding: 12px 28px; border-radius: 30px;
        font-weight: 700; font-size: 20px; display: inline-block;
        box-shadow: 0 4px 20px rgba(46,204,113,0.5);
    }
    @keyframes pulse {
        0%,100% { box-shadow: 0 4px 20px rgba(226,88,34,0.5); }
        50%      { box-shadow: 0 4px 40px rgba(226,88,34,0.9); }
    }
    .info-box {
        background: #f8f9fa; border-radius: 10px; padding: 14px 18px;
        margin: 10px 0; border-left: 4px solid #E25822;
        font-size: 14px; color: #444;
    }
</style>
""", unsafe_allow_html=True)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer="conv2d_15"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)

    fire_prob = 1.0 - float(predictions[0][0])
    if fire_prob > 0.5:
        heatmap = -heatmap

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(tf.abs(heatmap)) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_uint8, heatmap, alpha=0.4):
    h, w         = img_uint8.shape[:2]
    hmap         = cv2.resize(heatmap, (w, h))
    colored      = cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET)
    colored      = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(img_uint8, 1-alpha, colored, alpha, 0)
    return overlay, hmap


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Run `01_train.ipynb` first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, img_norm):
    raw       = float(model.predict(np.expand_dims(img_norm, 0), verbose=0)[0][0])
    fire_prob = 1.0 - raw
    return fire_prob


# ── App ──────────────────────────────────────────────────────────────────────
st.title("🔥 Forest Fire Detection")
st.markdown("Upload a forest image and the CNN will predict whether it contains fire.")
st.markdown("---")

model = load_model()

uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img  = Image.open(uploaded).convert("RGB")
    img_norm = np.array(pil_img.resize(IMG_SIZE), dtype=np.float32) / 255.0

    fire_prob = predict(model, img_norm)
    is_fire   = fire_prob > 0.5
    label     = "🔥 FIRE DETECTED" if is_fire else "✅ NO FIRE"
    badge     = "fire-badge" if is_fire else "nofire-badge"

    # ── Result ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Result")
        st.markdown(f'<div class="{badge}">{label}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Fire Probability",   f"{fire_prob*100:.1f}%")
        st.metric("No-Fire Probability", f"{(1-fire_prob)*100:.1f}%")
        st.progress(float(fire_prob), text=f"P(fire) = {fire_prob:.4f}")

    st.markdown("---")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Where did the model look? (Grad-CAM)")
    st.caption("Warm colours (red/yellow) = regions that most influenced the prediction.")

    img_uint8        = (img_norm * 255).astype(np.uint8)
    heatmap          = make_gradcam_heatmap(np.expand_dims(img_norm, 0), model)
    overlay, hmap    = overlay_gradcam(img_uint8, heatmap)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_uint8,           caption="Original",  use_container_width=True)
    with c2:
        fig, ax = plt.subplots()
        ax.imshow(hmap, cmap="jet")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("Heatmap")
    with c3:
        st.image(overlay,             caption="Overlay",   use_container_width=True)

    st.markdown("""
    <div class="info-box">
    💡 <b>Grad-CAM</b> shows which parts of the image triggered the prediction.
    If fire is detected, the highlighted area should overlap with flames or smoke.
    </div>
    """, unsafe_allow_html=True)

