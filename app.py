import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class NeuralInferenceLayer:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, img_array):
        results = self.model(img_array, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        raw_detections = []
        for b, c, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, b)
            raw_detections.append({
                "box": [x1, y1, x2, y2],
                "class_id": c,
                "class_name": self.model.names[c].lower(),
                "confidence": conf
            })
        return raw_detections

class WHOSymbolicClassifier:
    def __init__(self):
        self.lower_chromatin = np.array([120, 70, 50])
        self.upper_chromatin = np.array([170, 255, 255])
        self.lower_cytoplasm = np.array([80, 15, 60])
        self.upper_cytoplasm = np.array([170, 150, 255])
        self.lower_pigment = np.array([0, 0, 0])
        self.upper_pigment = np.array([180, 255, 85])

    def _extract_symbols(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_area = crop.shape[0] * crop.shape[1]
        min_dot_size = max(10, int(total_area * 0.006))

        # Chromatin
        chrom_k_size = max(2, int(crop.shape[1] * 0.01))
        chrom_kernel = np.ones((chrom_k_size, chrom_k_size), np.uint8)
        chrom_mask = cv2.inRange(hsv, self.lower_chromatin, self.upper_chromatin)
        chrom_mask = cv2.morphologyEx(chrom_mask, cv2.MORPH_CLOSE, chrom_kernel)
        num_dots, _, stats, _ = cv2.connectedComponentsWithStats(chrom_mask)

        real_dots = 0
        chromatin_area = 0
        for i in range(1, num_dots):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_dot_size:
                real_dots += 1
                chromatin_area += area

        # Cytoplasm
        cyto_k_size = max(3, int(crop.shape[1] * 0.03))
        cyto_kernel = np.ones((cyto_k_size, cyto_k_size), np.uint8)
        cyto_mask = cv2.inRange(hsv, self.lower_cytoplasm, self.upper_cytoplasm)
        cyto_mask = cv2.morphologyEx(cyto_mask, cv2.MORPH_CLOSE, cyto_kernel)
        cyto_area = np.sum(cyto_mask > 0)

        # Pigment
        pigment_mask = cv2.inRange(hsv, self.lower_pigment, self.upper_pigment)
        pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, chrom_kernel)
        has_pigment = np.sum(pigment_mask > 0) >= min_dot_size

        # Gametocyte Shape
        is_banana = False
        contours, _ = cv2.findContours(cyto_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(cnt)
            if c_area > (total_area * 0.15):
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = c_area / float(hull_area) if hull_area > 0 else 1.0
                if aspect_ratio > 1.8 or solidity < 0.75:
                    is_banana = True

        return {
            "dots": real_dots, "cyto_area": cyto_area, "chromatin_area": chromatin_area,
            "total_area": total_area, "is_banana": is_banana, "has_pigment": has_pigment,
            "chrom_mask": chrom_mask, "cyto_mask": cyto_mask
        }

    def classify(self, crop):
        s = self._extract_symbols(crop)

        if s["cyto_area"] < max(50, s["total_area"] * 0.05):
            stage = "background/artifact"
        elif s["is_banana"]:
            stage = "gametocyte"
        elif s["dots"] >= 5 or (s["cyto_area"] > 0 and s["chromatin_area"] / s["cyto_area"] > 0.80):
            stage = "schizont"
        elif 1 <= s["dots"] <= 4:
            if s["dots"] <= 2 and s["cyto_area"] < (s["total_area"] * 0.55) and not s["has_pigment"]:
                stage = "ring"
            else:
                stage = "trophozoite"
        else:
            stage = "background/artifact"
            
        return stage, s

class NeuroSymbolicSystem:
    def __init__(self, weights_path, infected_class_ids):
        self.neural_layer = NeuralInferenceLayer(weights_path)
        self.symbolic_layer = WHOSymbolicClassifier()
        self.infected_ids = infected_class_ids

    def process_image(self, full_img):
        candidates = self.neural_layer.detect(full_img)
        final_results = []

        for cand in candidates:
            # If YOLO flagged it as one of the infected stages, run WHO Math
            if cand["class_id"] in self.infected_ids:
                x1, y1, x2, y2 = cand["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(full_img.shape[1], x2), min(full_img.shape[0], y2)
                crop = full_img[y1:y2, x1:x2]

                if crop.size == 0: continue

                refined_stage, stats = self.symbolic_layer.classify(crop)

                final_results.append({
                    "box": cand["box"],
                    "neural_conf": cand["confidence"],
                    "yolo_class": cand["class_name"],
                    "symbolic_stage": refined_stage,
                    "stats": stats,
                    "crop": crop
                })
            else:
                # If it's an RBC or Leukocyte, skip math and just pass it through
                final_results.append({
                    "box": cand["box"],
                    "neural_conf": cand["confidence"],
                    "yolo_class": cand["class_name"],
                    "symbolic_stage": cand["class_name"], 
                    "stats": None,
                    "crop": None
                })

        return final_results
# ==========================================
# 2. STREAMLIT UI & MODEL TOGGLE
# ==========================================

st.set_page_config(layout="wide", page_title="Neuro-Symbolic Malaria AI")

# --- THE SIDEBAR TOGGLE ---
st.sidebar.title("Model Configuration")
model_type = st.sidebar.radio(
    "Select YOLOv12 Architecture:",
    ("Merged (Binary Infection)", "Unmerged (Multi-Class)")
)

# Dynamically set the weights path and the infected class IDs based on the toggle
if model_type == "Merged (Binary Infection)":
    WEIGHTS_PATH = "weights/merged_best.pt"
    # In a merged dataset, usually Class 0 is "Infected"
    INFECTED_IDS = [0] 
    st.sidebar.info("Using YOLO to detect general infection, and relying 100% on the Symbolic Layer for stage classification.")
else:
    WEIGHTS_PATH = "weights/unmerged_best.pt"
    # In your unmerged dataset, these are the infected stages
    INFECTED_IDS = [1, 4, 5, 6] 
    st.sidebar.info("Using YOLO for initial stage classification, and using the Symbolic Layer to audit and correct YOLO's predictions.")


# --- HELPER FUNCTIONS & SETTINGS ---
COLOR_MAP = {
    "red blood cell": (150, 150, 150),   # Gray
    "leukocyte": (255, 0, 255),          # Magenta
    "gametocyte": (0, 165, 255),         # Orange
    "ring": (0, 255, 255),               # Yellow
    "trophozoite": (0, 0, 255),          # Red
    "schizont": (128, 0, 128),           # Purple
    "difficult": (100, 100, 100)         # Dark Gray
}

def draw_boxes(img, results, use_symbolic_labels=False):
    """Helper function to draw boxes based on either Neural or Symbolic labels."""
    drawn_img = img.copy()
    for res in results:
        x1, y1, x2, y2 = res["box"]
        # Decide which label to use
        label = res["symbolic_stage"] if use_symbolic_labels else res["yolo_class"]
        if label in ["background/artifact", "difficult"]:
            continue # Don't draw rejected dirt
        color = COLOR_MAP.get(label, (255, 255, 255)) # Default to white if unknown
        cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(drawn_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return drawn_img

# --- SYSTEM INITIALIZATION ---
# Update the cache function to accept the dynamic path
@st.cache_resource
def load_system(weights, infected_ids):
    return NeuroSymbolicSystem(weights, infected_class_ids=infected_ids) 

with st.spinner(f"Loading {model_type} Weights..."):
    system = load_system(WEIGHTS_PATH, INFECTED_IDS)


# --- MAIN UI & PROCESSING ---
st.title("Neurosymbolic Malaria Detection")
st.markdown("Upload a high-resolution Giemsa-stained blood smear to compare the **Black Box Neural Network** against the **Transparent WHO Symbolic Logic**.")

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Processing image through pipeline..."):
        results = system.process_image(img_bgr)

    # Generate the two comparison images
    img_neural = draw_boxes(img_bgr, results, use_symbolic_labels=False)
    img_symbolic = draw_boxes(img_bgr, results, use_symbolic_labels=True)

    # Convert back to RGB for Streamlit rendering
    img_neural_rgb = cv2.cvtColor(img_neural, cv2.COLOR_BGR2RGB)
    img_symbolic_rgb = cv2.cvtColor(img_symbolic, cv2.COLOR_BGR2RGB)

    st.markdown("---")

    if model_type == "Merged (Binary Infection)":
        # SIDE-BY-SIDE VISUAL COMPARISON
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline: YOLOv12 (Neural Only)")
            st.markdown("Black-box predictions based purely on learned training weights.")
            st.image(img_neural_rgb, use_column_width=True)

        with col2:
            st.subheader("Proposed: Neuro-Symbolic Pipeline")
            st.markdown("Predictions overridden and verified by WHO morphological math.")
            st.image(img_symbolic_rgb, use_column_width=True)

        st.markdown("---")

        # EXPLAINABILITY SECTION
        st.subheader("Symbolic WHO Explainability Breakdown")
        st.markdown("Mathematical audit of the infected cells processed by the Symbolic Layer.")
        infected_count = 0
        for res in results:
            if res["stats"] is not None and res["symbolic_stage"] != "background/artifact":
                infected_count += 1
                stats = res["stats"]
                # Highlight if the Symbolic Layer disagreed with YOLO!
                conflict_warning = ""
                if res["yolo_class"] != res["symbolic_stage"]:
                    conflict_warning = f"(Corrected from YOLO's '{res['yolo_class']}')"
                with st.expander(f"Parasite #{infected_count}: {res['symbolic_stage'].upper()}{conflict_warning}"):
                    mask_cols = st.columns(3)
                    with mask_cols[0]:
                        st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), caption="Raw Crop")
                    with mask_cols[1]:
                        st.image(stats["chrom_mask"], caption="Chromatin Mask", clamp=True)
                    with mask_cols[2]:
                        st.image(stats["cyto_mask"], caption="Cytoplasm Mask", clamp=True)
                    st.write(f"**YOLO Neural Confidence:** {res['neural_conf']:.2f}")
                    st.write(f"**WHO Symbolic Metrics:**")
                    st.write(f"- Chromatin Dots: **{stats['dots']}**")
                    st.write(f"- Cytoplasm Density: **{(stats['cyto_area'] / stats['total_area']) * 100:.1f}%**")
                    st.write(f"- Gametocyte Shape (Banana): **{stats['is_banana']}**")
                    st.write(f"- Hemozoin Pigment Found: **{stats['has_pigment']}**")
        if infected_count == 0:
            st.success("No active WHO-verified infections found in this image.")

    else:
        # UNMERGED: NEURAL ONLY VIEW
        st.subheader("YOLOv12 Multi-Class Predictions (Neural Only)")
        st.markdown("End-to-end black-box predictions. The neural network attempts to classify all 7 stages directly.")
        
        # Display just the neural image in the center
        st.image(img_neural_rgb, use_column_width=True)
        
        st.markdown("---")
        
        st.subheader("Neural Detections Breakdown")
        st.markdown("Raw confidence scores from the YOLOv12 model.")
        
        detection_count = 0
        for res in results:
            label = res["yolo_class"]
            
            # Skip background artifacts if you don't want them cluttering the list
            if label not in ["background/artifact", "difficult"]:
                detection_count += 1
                
                with st.expander(f"Detection #{detection_count}: {label.upper()}"):
                    # Check if a crop exists (RBCs might not have crops passed back depending on your setup)
                    if res.get("crop") is not None and res["crop"].size > 0:
                        st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), caption="Raw Crop", width=150)
                    
                    st.write(f"**YOLO Neural Confidence:** {res['neural_conf']:.2f}")
                    st.write("*No Symbolic WHO math applied in this mode.*")
                    
        if detection_count == 0:
            st.success("No cells detected by the Neural Network.")
