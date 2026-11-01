import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
from datetime import datetime

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="DiaRisk 30 - Hệ thống Hỗ trợ Lâm sàng",
    page_icon="🩺",
    layout="wide"
)

# Khởi tạo bộ nhớ lịch sử
if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

# --- 2. TẢI MÔ HÌNH VÀ CẤU HÌNH ĐẶC TRƯNG ---
@st.cache_resource
def load_model_bundle():
    return joblib.load("kltn_diabetes_model.joblib")

try:
    bundle = load_model_bundle()
    # Hỗ trợ lấy model/pipeline tùy thuộc vào cách lưu
    if "calibrator" in bundle:
        model_to_predict = bundle["calibrator"]
    elif "pipeline" in bundle:
        model_to_predict = bundle["pipeline"]
    else:
        model_to_predict = bundle.get("model")
        
    preprocessor = bundle.get("preprocessor")
    core_model = bundle.get("model") # Dùng riêng cho SHAP
    feature_names = bundle.get("feature_names", [])
    
except Exception as e:
    st.error(f"Lỗi khi tải file mô hình `.joblib`: {e}")
    st.stop()

# --- 3. GIAO DIỆN HEADER ---
st.markdown("""
    <style>
        .medical-header {
            background: linear-gradient(135deg, #0A2540 0%, #133E67 100%);
            padding: 24px 30px;
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .medical-title { font-size: 28px; font-weight: 700; margin: 0; }
        .medical-subtitle { font-size: 15px; color: #A3BFFA; margin-top: 6px; }
    </style>
    <div class="medical-header">
        <div class="medical-title">🩺 DiaRisk 30 Pro</div>
        <div class="medical-subtitle">Hệ thống Đánh giá Rủi ro Tái nhập viện Đái tháo đường (Mô hình Hiệu chuẩn)</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. FORM NHẬP LIỆU CHUYÊN SÂU ---
with st.form("prediction_form"):
    st.markdown("### 📋 Thông tin Định danh")
    id_col1, id_col2 = st.columns(2)
    with id_col1: patient_name = st.text_input("Họ và tên bệnh nhân", placeholder="Nguyễn Văn A")
    with id_col2: patient_id = st.text_input("Mã Y tế (ID)", placeholder="BN-2026-8839")

    # Form 1: Hành chính
    st.markdown("### Nhóm 1: Thông tin Hành chính")
    c1_1, c1_2, c1_3 = st.columns(3)
    age = c1_1.selectbox("Nhóm tuổi", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
    gender = c1_2.selectbox("Giới tính", ["Male", "Female"])
    race = c1_3.selectbox("Chủng tộc", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Missing"])
    
    c1_4, c1_5, c1_6 = st.columns(3)
    admission_type_id = c1_4.selectbox("Loại nhập viện (ID)", ["1", "2", "3", "4", "5", "6", "7", "8"])
    admission_source_id = c1_5.selectbox("Nguồn nhập viện (ID)", ["1", "7", "17", "4", "2", "Other"])
    discharge_disposition_id = c1_6.selectbox("Tình trạng xuất viện (ID)", ["1", "2", "3", "4", "5", "6", "7", "8", "22", "23", "24", "25", "27", "28"])

    # Form 2: Tiền sử
    st.markdown("### Nhóm 2: Tiền sử & Sử dụng dịch vụ y tế")
    c2_1, c2_2, c2_3 = st.columns(3)
    time_in_hospital = c2_1.number_input("Số ngày nằm viện hiện tại", min_value=1, max_value=30, value=3)
    total_visits = c2_2.number_input("Tổng số lượt khám trước đây tại viện", min_value=1, max_value=50, value=1)
    number_inpatient = c2_3.number_input("Số lần nội trú (1 năm qua)", min_value=0, max_value=50, value=0)
    
    c2_4, c2_5 = st.columns(2)
    number_outpatient = c2_4.number_input("Số lần ngoại trú (1 năm qua)", min_value=0, max_value=50, value=0)
    number_emergency = c2_5.number_input("Số lần cấp cứu (1 năm qua)", min_value=0, max_value=50, value=0)

    # Form 3: Chẩn đoán & Xét nghiệm
    st.markdown("### Nhóm 3: Chẩn đoán & Xét nghiệm")
    icd_groups = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]
    c3_1, c3_2, c3_3 = st.columns(3)
    diag_1 = c3_1.selectbox("Chẩn đoán chính (ICD-9 Group)", icd_groups, index=3)
    diag_2 = c3_2.selectbox("Chẩn đoán phụ 1", icd_groups, index=0)
    diag_3 = c3_3.selectbox("Chẩn đoán phụ 2", icd_groups, index=8)
    
    c3_4, c3_5, c3_6 = st.columns(3)
    number_diagnoses = c3_4.number_input("Tổng số chẩn đoán (bao gồm bệnh nền)", min_value=1, max_value=20, value=5)
    num_lab_procedures = c3_5.number_input("Số lần xét nghiệm Lab", min_value=1, max_value=150, value=40)
    num_procedures = c3_6.number_input("Số thủ thuật y tế (không gồm xét nghiệm)", min_value=0, max_value=10, value=0)

    # Form 4: Thuốc điều trị
    st.markdown("### Nhóm 4: Thuốc điều trị đái tháo đường")
    c4_1, c4_2, c4_3 = st.columns(3)
    diabetesMed = c4_1.selectbox("Có sử dụng thuốc ĐTĐ không?", ["Yes", "No"])
    change = c4_2.selectbox("Có thay đổi phác đồ/liều lượng không?", ["No", "Ch"])
    num_medications = c4_3.number_input("Tổng số TẤT CẢ các loại thuốc đang dùng", min_value=1, max_value=100, value=12)
    
    c4_4, c4_5, c4_6 = st.columns(3)
    num_active_meds = c4_4.number_input("Số loại thuốc RIÊNG cho ĐTĐ đang dùng", min_value=0, max_value=10, value=1)
    insulin = c4_5.selectbox("Tình trạng dùng Insulin", ["No", "Steady", "Up", "Down"])
    metformin = c4_6.selectbox("Tình trạng dùng Metformin", ["No", "Steady", "Up", "Down"])

    submitted = st.form_submit_button("🔍 Chẩn Đoán & Phân Tầng Rủi Ro", use_container_width=True)

# --- 5. XỬ LÝ VÀ DỰ BÁO ---
if submitted:
    if not patient_name.strip():
        st.warning("⚠️ Vui lòng nhập họ tên bệnh nhân.")
    else:
        # Tự động tính toán các biến Feature Engineering ẩn
        service_utilization = number_outpatient + number_emergency + number_inpatient
        labs_per_day = num_lab_procedures / max(time_in_hospital, 1)

        # Gom toàn bộ dữ liệu vào Dictionary, cấu trúc khớp 100% với feature_names
        raw_data = {
            "age": age, "gender": gender, "race": race,
            "admission_type_id": admission_type_id, "admission_source_id": admission_source_id, "discharge_disposition_id": discharge_disposition_id,
            "time_in_hospital": time_in_hospital, "total_visits": total_visits, "service_utilization": service_utilization,
            "number_inpatient": number_inpatient, "number_emergency": number_emergency, "number_outpatient": number_outpatient,
            "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3, "number_diagnoses": number_diagnoses,
            "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures, "labs_per_day": labs_per_day,
            "diabetesMed": diabetesMed, "change": change, "num_medications": num_medications, "num_active_meds": num_active_meds,
            "insulin": insulin, "metformin": metformin
        }

        # Lọc đúng các cột mà mô hình mong đợi (loại bỏ nếu dư)
        input_data = pd.DataFrame([raw_data])
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0  # Fallback an toàn
        input_data = input_data[feature_names]

        try:
            # 1. Dự báo Xác suất (Hiệu chuẩn)
            prob_risk = model_to_predict.predict_proba(input_data)[0][1]
            prob_percent = prob_risk * 100
            
            # Phân tầng theo chuẩn Y tế
            if prob_percent >= 25.0:
                risk_label, color, alert = "Nguy cơ Cao", "red", "Cần can thiệp lâm sàng và theo dõi sát sao sau xuất viện."
            elif prob_percent >= 12.0:
                risk_label, color, alert = "Nguy cơ Trung bình", "orange", "Cần lưu ý, lập kế hoạch ngoại trú cẩn thận."
            else:
                risk_label, color, alert = "Nguy cơ Thấp", "green", "An toàn, tiếp tục duy trì phác đồ hiện tại."

            st.divider()
            st.markdown(f"### 📊 Kết Quả Đánh Giá: **{patient_name}**")
            
            res_col1, res_col2 = st.columns([1.2, 2])
            with res_col1:
                st.metric(label="Xác suất tái nhập viện (<30 ngày)", value=f"{prob_percent:.1f}%")
                st.markdown(f"<h4 style='color: {color};'>{risk_label}</h4>", unsafe_allow_html=True)
                st.info(f"**Khuyến nghị:** {alert}")
                
            with res_col2:
                st.markdown("#### 🧬 Đóng Góp Yếu Tố Lâm Sàng (SHAP)")
                try:
                    # Trích xuất preprocessor và XGBoost core model để vẽ SHAP
                    if preprocessor is not None and core_model is not None:
                        processed_input = preprocessor.transform(input_data)
                        # Nếu có selector trong pipeline, phải lấy qua selector
                        if hasattr(model_to_predict, 'named_steps') and 'selector' in model_to_predict.named_steps:
                            processed_input = model_to_predict.named_steps['selector'].transform(processed_input)
                            
                        explainer = shap.TreeExplainer(core_model)
                        shap_values = explainer.shap_values(processed_input)
                        
                        s_vals = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0])
                        
                        shap_df = pd.DataFrame({
                            "Yếu tố lâm sàng": feature_names[:len(s_vals)],
                            "Mức độ tác động": s_vals[:len(feature_names)]
                        }).reindex(shap_df["Mức độ tác động"].abs().sort_values(ascending=False).index).head(8)
                        
                        colors = ["#DC2626" if v > 0 else "#2563EB" for v in shap_df["Mức độ tác động"]]
                        fig_shap = go.Figure(go.Bar(
                            x=shap_df["Mức độ tác động"], y=shap_df["Yếu tố lâm sàng"],
                            orientation="h", marker_color=colors, text=[f"{v:+.2f}" for v in shap_df["Mức độ tác động"]], textposition="auto"
                        ))
                        fig_shap.update_layout(title="Yếu tố TĂNG (Đỏ) hoặc GIẢM (Xanh) nguy cơ", height=350, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_shap, use_container_width=True)
                except Exception:
                    st.warning("Biểu đồ phân tích chuyên sâu tạm thời không khả dụng.")

            # Lưu lịch sử
            st.session_state.patient_history.insert(0, {
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M"), "Mã BN": patient_id, "Họ Tên": patient_name,
                "Xác suất (%)": f"{prob_percent:.1f}%", "Phân loại": risk_label
            })

        except Exception as e:
            st.error(f"Lỗi xử lý dự báo: {e}")

# --- 6. LỊCH SỬ PHIÊN ---
if st.session_state.patient_history:
    st.divider()
    st.markdown("### 🗂️ Danh sách Bệnh nhân vừa đánh giá")
    st.dataframe(pd.DataFrame(st.session_state.patient_history), use_container_width=True, hide_index=True)
    if st.button("🗑️ Xóa lịch sử phiên"):
        st.session_state.patient_history = []
        st.rerun()