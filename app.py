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

# --- 2. KHỞI TẠO BỘ NHỚ LỊCH SỬ HỒ SƠ ---
if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

# --- 3. TẢI MÔ HÌNH VÀ CẤU HÌNH ĐẶC TRƯNG AN TOÀN TUYỆT ĐỐI ---
@st.cache_resource
def load_model_bundle():
    return joblib.load("kltn_diabetes_model.joblib")

try:
    model_bundle = load_model_bundle()
    
    model = None
    preprocessor = None
    
    # Kiểm tra linh hoạt cấu trúc file joblib
    if isinstance(model_bundle, dict):
        # Thử tìm theo các tên key thông dụng
        model = model_bundle.get("model") or model_bundle.get("clf") or model_bundle.get("estimator")
        preprocessor = model_bundle.get("preprocessor") or model_bundle.get("scaler")
        
        # Nếu dict không có key chuẩn, quét tự động xem giá trị nào có hàm predict_proba
        if model is None:
            for val in model_bundle.values():
                if hasattr(val, "predict_proba"):
                    model = val
                    break
    else:
        # Trường hợp file joblib lưu thẳng đối tượng mô hình/pipeline
        model = model_bundle

    # Kiểm tra xem mô hình đã tồn tại và hợp lệ chưa
    if model is None or not hasattr(model, "predict_proba"):
        st.error("⚠️ File `.joblib` tải lên không chứa mô hình máy học hợp lệ (thiếu hàm predict_proba).")
        st.stop()

    feature_names = [
        'age', 'time_in_hospital', 'num_lab_procedures', 
        'num_procedures', 'num_medications', 'number_outpatient', 
        'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]

except Exception as e:
    st.error(f"Lỗi khi tải file mô hình `.joblib`: {e}")
    st.stop()

# --- 4. GIAO DIỆN Y TẾ CHUYÊN NGHIỆP (HEADER) ---
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
        .medical-title {
            font-size: 28px;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: -0.5px;
        }
        .medical-subtitle {
            font-size: 15px;
            color: #A3BFFA;
            margin-top: 6px;
            font-weight: 400;
        }
        .medical-footer {
            text-align: center;
            color: #888888;
            font-size: 13px;
            margin-top: 50px;
            padding: 15px;
            border-top: 1px solid #333333;
        }
    </style>
    <div class="medical-header">
        <div class="medical-title">
            <span>🩺</span> DiaRisk 30
        </div>
        <div class="medical-subtitle">
            Hệ thống hỗ trợ lâm sàng quản lý hồ sơ và dự báo xác suất tái nhập viện đái tháo đường trong 30 ngày.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 5. FORM NHẬP LIỆU LÂM SÀNG ---
st.markdown("### 📋 Thông tin Hành chính & Chỉ số Lâm sàng Bệnh nhân")
with st.form("prediction_form"):
    st.markdown("#### 👤 Thông tin định danh")
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        patient_name = st.text_input("Họ và tên bệnh nhân", placeholder="Ví dụ: Nguyễn Văn A")
    with id_col2:
        patient_id = st.text_input("Mã định danh / Mã Y tế (ID)", placeholder="Ví dụ: BN-2026-8839")

    st.markdown("#### 🔬 Các chỉ số lâm sàng")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_group = st.selectbox("Nhóm tuổi", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
        time_in_hospital = st.number_input("Số ngày nằm viện", min_value=1, max_value=30, value=3)
        num_lab_procedures = st.number_input("Số xét nghiệm Lab", min_value=0, max_value=200, value=30)
        
    with col2:
        num_procedures = st.number_input("Số thủ thuật y tế", min_value=0, max_value=10, value=1)
        num_medications = st.number_input("Số loại thuốc sử dụng", min_value=0, max_value=100, value=15)
        number_outpatient = st.number_input("Số lần khám ngoại trú (1 năm qua)", min_value=0, max_value=50, value=0)
        
    with col3:
        number_emergency = st.number_input("Số lần cấp cứu (1 năm qua)", min_value=0, max_value=50, value=0)
        number_inpatient = st.number_input("Số lần điều trị nội trú (1 năm qua)", min_value=0, max_value=50, value=1)
        number_diagnoses = st.number_input("Tổng số bệnh lý chẩn đoán kèm theo", min_value=1, max_value=20, value=5)

    submitted = st.form_submit_button("🔍 Thực hiện Đánh giá & Lưu Hồ sơ", use_container_width=True)

# --- 6. XỬ LÝ DỰ BÁO VÀ LƯU TRỮ HỒ SƠ ---
if submitted:
    if not patient_name.strip():
        st.warning("⚠️ Vui lòng nhập họ tên bệnh nhân trước khi thực hiện đánh giá.")
    else:
        input_data = pd.DataFrame({
            'age': [age_group],
            'time_in_hospital': [time_in_hospital],
            'num_lab_procedures': [num_lab_procedures],
            'num_procedures': [num_procedures],
            'num_medications': [num_medications],
            'number_outpatient': [number_outpatient],
            'number_emergency': [number_emergency],
            'number_inpatient': [number_inpatient],
            'number_diagnoses': [number_diagnoses]
        })
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_names]

        try:
            if preprocessor is not None:
                processed_data = preprocessor.transform(input_data)
            else:
                processed_data = input_data

            prediction_proba = model.predict_proba(processed_data)[0][1]
            risk_label = "Nguy cơ cao (≥ 50%)" if prediction_proba >= 0.5 else "Nguy cơ thấp (< 50%)"
            
            # Lưu hồ sơ vào session state
            record_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_record = {
                "Thời gian": record_time,
                "Mã ID": patient_id if patient_id else "N/A",
                "Họ và tên": patient_name,
                "Xác suất tái nhập viện": f"{prediction_proba * 100:.2f}%",
                "Đánh giá": risk_label
            }
            st.session_state.patient_history.insert(0, new_record)
            
            st.divider()
            st.markdown(f"### 📊 Kết quả Phân tích Lâm sàng cho bệnh nhân: **{patient_name}**")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Xác suất tái nhập viện trong 30 ngày", value=f"{prediction_proba * 100:.2f}%")
                if prediction_proba >= 0.5:
                    st.error("⚠️ **Đánh giá:** Nguy cơ cao tái nhập viện. Cần lập kế hoạch theo dõi sát sao.")
                else:
                    st.success("✅ **Đánh giá:** Nguy cơ thấp. Tiếp tục duy trì phác đồ hiện tại.")
                    
            with res_col2:
                st.markdown("#### 🧬 GIẢI THÍCH NGUYÊN NHÂN CÁ NHÂN HÓA (SHAP)")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(processed_data)
                    
                    if isinstance(shap_values, list):
                        s_vals = shap_values[1][0]
                    elif len(shap_values.shape) == 3:
                        s_vals = shap_values[0, :, 1]
                    else:
                        s_vals = shap_values[0]
                        
                    shap_df = pd.DataFrame({
                        "Yếu tố lâm sàng": feature_names[:len(s_vals)],
                        "Mức tác động (SHAP)": s_vals[:len(feature_names)]
                    }).sort_values(by="Mức tác động (SHAP)", key=abs, ascending=False).head(10)
                    
                    colors = ["#DC2626" if v > 0 else "#2563EB" for v in shap_df["Mức tác động (SHAP)"]]
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df["Mức tác động (SHAP)"],
                        y=shap_df["Yếu tố lâm sàng"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in shap_df["Mức tác động (SHAP)"]],
                        textposition="auto"
                    ))
                    
                    x_min = shap_df["Mức tác động (SHAP)"].min()
                    x_max = shap_df["Mức tác động (SHAP)"].max()
                    padding = (x_max - x_min) * 0.2 if x_max != x_min else 0.1

                    fig_shap.update_layout(
                        title="Top yếu tố làm TĂNG (Đỏ) hoặc GIẢM (Xanh) nguy cơ",
                        xaxis_title="Giá trị tác động SHAP (Log-odds)",
                        yaxis_title="",
                        height=380,
                        margin=dict(l=10, r=20, t=40, b=20),
                        xaxis=dict(range=[x_min - padding, x_max + padding])
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as shap_err:
                    st.info("Biểu đồ giải thích chi tiết đang được cập nhật.")
                    
        except Exception as err:
            st.error(f"Lỗi trong quá trình dự báo: {err}")

# --- 7. HIỂN THỊ DANH SÁCH LỊCH SỬ HỒ SƠ ĐÃ KHÁM TRONG PHIÊN ---
if len(st.session_state.patient_history) > 0:
    st.divider()
    st.markdown("### 🗂️ Danh sách Hồ sơ Bệnh nhân đã đánh giá trong phiên")
    history_df = pd.DataFrame(st.session_state.patient_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Xóa lịch sử phiên làm việc"):
        st.session_state.patient_history = []
        st.rerun()

# --- 8. CHÂN TRANG ---
st.markdown("""
    <div class="medical-footer">
        Công cụ hỗ trợ nghiên cứu và tham khảo lâm sàng, không thay thế chẩn đoán chuyên môn của bác sĩ.
    </div>
""", unsafe_allow_html=True)