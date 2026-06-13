import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image

st.set_page_config(
    page_title="Steel Defect Dataset Explorer",
    page_icon="📊",
    layout="wide"
)

#defect class information
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

DEFECT_INFO = {
    'crazing': {
        'description': 'A network of fine surface cracks resembling dried mud or alligator skin.',
        'cause': 'Thermal stress during cooling — excessive temperature differentials between surface and core.',
        'severity': 'Medium'
    },
    'inclusion': {
        'description': 'Foreign material embedded in the steel surface — dark spots or streaks.',
        'cause': 'Slag carryover or refractory erosion during steelmaking and casting.',
        'severity': 'High'
    },
    'patches': {
        'description': 'Irregular areas of inconsistent surface texture or discoloration.',
        'cause': 'Non-uniform descaling or inconsistent scale formation during reheating.',
        'severity': 'Medium'
    },
    'pitted_surface': {
        'description': 'Small cavities or depressions scattered across the steel surface.',
        'cause': 'Scale entrapment during roughing or finishing mill passes.',
        'severity': 'Medium'
    },
    'rolled-in_scale': {
        'description': 'Oxide scale pressed into the steel surface during rolling.',
        'cause': 'Inadequate descaling before rolling — scale gets compressed into the surface.',
        'severity': 'High'
    },
    'scratches': {
        'description': 'Linear surface marks running parallel or diagonal to rolling direction.',
        'cause': 'Mechanical contact with worn guide equipment or debris on roller table.',
        'severity': 'Low'
    }
}

#training and validation counts
TRAIN_COUNTS = {cls: 240 for cls in CLASSES}
VAL_COUNTS = {cls: 60 for cls in CLASSES}

#title
st.title("📊 NEU Steel Surface Defect Dataset Explorer")
st.markdown("**Interactive exploration of the NEU Steel Surface Defect Dataset used to train the defect detection model**")
st.markdown("Dataset: 1,800 images across 6 defect classes — used to train a ResNet18 model achieving 95% validation accuracy")

st.divider()

#sidebar
with st.sidebar:
    st.header("Dataset Overview")
    st.metric("Total Images", "1,800")
    st.metric("Defect Classes", "6")
    st.metric("Train / Val Split", "80% / 20%")
    st.metric("Images per Class", "300")
    st.metric("Image Size", "200 × 200 px")
    st.metric("Image Type", "Grayscale")
    
    st.divider()
    st.markdown("**Related Project:**")
    st.markdown("[🔬 Steel Defect Detector App](https://steel-defect-detector-m57ct4lq6jkouwkxmbjths.streamlit.app/)")

#section 1 - class distribution
st.header("1. Class Distribution")
st.markdown("All 6 defect classes have equal representation — 240 training and 60 validation images each. This balanced dataset means no class weighting was needed during training.")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(CLASSES))
width = 0.35

bars1 = ax.bar(x - width/2, list(TRAIN_COUNTS.values()), width, label='Training', color='steelblue')
bars2 = ax.bar(x + width/2, list(VAL_COUNTS.values()), width, label='Validation', color='coral')

ax.set_title('Class Distribution — Training vs Validation', fontweight='bold', fontsize=14)
ax.set_xlabel('Defect Class')
ax.set_ylabel('Number of Images')
ax.set_xticks(x)
ax.set_xticklabels([cls.replace('_', ' ').title() for cls in CLASSES], rotation=35, ha='right')
ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1))
ax.set_ylim(0, 300)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.divider()

#section 2 - sample images
st.header("2. Sample Images Per Class")
st.markdown("One representative image from each defect class. All images are grayscale 200×200 pixels captured from a hot-rolling steel strip production line.")

#check if sample images exist
image_files = {cls: f"{cls}_sample.bmp.jpg" for cls in CLASSES}
images_found = all(os.path.exists(f) for f in image_files.values())

if images_found:
    cols = st.columns(3)
    for i, cls in enumerate(CLASSES):
        with cols[i % 3]:
            img = Image.open(image_files[cls])
            st.image(img, caption=cls.replace('_', ' ').title(), use_container_width=True)
else:
    st.warning("Sample images not found. Make sure the 6 .bmp sample images are in the same folder as this app.")

st.divider()


#section 3 - pixel intensity analysis
st.header("3. Pixel Intensity Analysis")
st.markdown("Average pixel brightness per defect class. Higher values mean brighter images. This reveals visual characteristics that distinguish defect types before any modeling.")

if images_found:
    intensities = {}
    for cls in CLASSES:
        img = Image.open(image_files[cls]).convert('L')
        img_array = np.array(img)
        intensities[cls] = img_array.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    bars = ax.bar(
        [cls.replace('_', ' ').title() for cls in CLASSES],
        list(intensities.values()),
        color=colors
    )

    #adding value labels on top of each bar
    for bar, val in zip(bars, intensities.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_title('Average Pixel Intensity per Defect Class', fontweight='bold', fontsize=14)
    ax.set_xlabel('Defect Class')
    ax.set_ylabel('Average Pixel Value (0-255)')
    ax.set_ylim(0, 280)
    plt.xticks(rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Note: Values based on one sample image per class. Results may vary across the full 300-image dataset per class.")


#section 4 - defect reference guide
st.header("3. Defect Reference Guide")
st.markdown("Technical description, root cause, and severity rating for each defect type.")

for cls in CLASSES:
    info = DEFECT_INFO[cls]
    severity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    
    with st.expander(f"{severity_color[info['severity']]} {cls.replace('_', ' ').title()} — Severity: {info['severity']}"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**What it looks like:**")
            st.write(info['description'])
        with col2:
            st.markdown("**Root cause:**")
            st.write(info['cause'])

st.divider()

#section 5 - dataset statistics
st.header("4. Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Training Images", "1,440")
with col2:
    st.metric("Total Validation Images", "360")
with col3:
    st.metric("Train/Val Ratio", "80/20")
with col4:
    st.metric("Classes", "6 — Perfectly Balanced")

st.divider()

#section 6 - severity breakdown
st.header("5. Defect Severity Breakdown")
st.markdown("Classification of defect types by industrial severity — based on impact on steel product quality.")

col1, col2, col3 = st.columns(3)

with col1:
    st.error("🔴 High Severity")
    st.markdown("- **Inclusion** — foreign material contaminates the steel")
    st.markdown("- **Rolled-in Scale** — scale embedded during rolling")

with col2:
    st.warning("🟡 Medium Severity")
    st.markdown("- **Crazing** — surface crack network")
    st.markdown("- **Patches** — inconsistent surface texture")
    st.markdown("- **Pitted Surface** — surface cavities")

with col3:
    st.success("🟢 Low Severity")
    st.markdown("- **Scratches** — linear surface marks")

st.divider()

#footer
st.caption("NEU Steel Surface Defect Dataset — Northeastern University | Used for ResNet18 transfer learning research")
st.caption("Built by Kazi Zaber Faruqui | Memorial University of Newfoundland | Industrial AI Research")