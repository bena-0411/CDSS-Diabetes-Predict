import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import pytz
import shap
import matplotlib.pyplot as plt
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="DiaRisk 30 - Hệ thống Hỗ trợ Lâm sàng",
    page_icon="🩺",
    layout="wide"
)

if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

# --- 2. TẢI MÔ HÌNH VÀ CẤU HÌNH ĐẶC TRƯNG AN TOÀN ---
@st.cache_resource
def load_model_bundle():
    return joblib.load("kltn_diabetes_model.joblib")

DEFAULT_FEATURES = [
    "age", "gender", "race", 
    "admission_type_id", "admission_source_id", "discharge_disposition_id",
    "time_in_hospital", "total_visits", "service_utilization", 
    "number_inpatient", "number_emergency", "number_outpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "num_lab_procedures", "num_procedures", "labs_per_day",
    "insulin", "metformin", "change", "diabetesMed", 
    "num_medications", "num_active_meds"
]

try:
    bundle = load_model_bundle()
    
    if isinstance(bundle, dict):
        model_to_predict = bundle.get("calibrator") or bundle.get("pipeline") or bundle.get("model")
        preprocessor = bundle.get("preprocessor")
        core_model = bundle.get("model")
        saved_cols = bundle.get("feature_names") or bundle.get("selected_columns") or bundle.get("columns")
        feature_names = list(saved_cols) if saved_cols is not None and len(saved_cols) > 0 else DEFAULT_FEATURES
    else:
        model_to_predict = bundle
        preprocessor = None
        core_model = None
        feature_names = DEFAULT_FEATURES

    if hasattr(model_to_predict, "feature_names_in_"):
        feature_names = list(model_to_predict.feature_names_in_)
    elif preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        feature_names = list(preprocessor.feature_names_in_)

except Exception as e:
    st.error(f"Lỗi khi tải file mô hình `.joblib`: {e}")
    st.stop()

# --- TẠO SHAP EXPLAINER ---
@st.cache_resource
def get_shap_explainer(_model):
    try:
        model_core = _model
        
        if hasattr(model_core, "steps"):
            model_core = model_core.steps[-1][1]
            
        if hasattr(model_core, "calibrated_classifiers_"):
            model_core = model_core.calibrated_classifiers_[0].estimator
        elif hasattr(model_core, "estimator"):
            model_core = model_core.estimator
            
        explainer = shap.TreeExplainer(model_core)
        return explainer
    except Exception as e:
        print(f"Lỗi khởi tạo SHAP Explainer: {e}")
        return None

explainer = get_shap_explainer(core_model or model_to_predict)

# --- 3. HEADER GIAO DIỆN ---
st.markdown("""
    <div style="background: linear-gradient(135deg, #0A2540 0%, #133E67 100%); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white;">🩺 DiaRisk 30 Pro</h2>
        <p style="margin: 5px 0 0 0; color: #A3BFFA; font-size: 15px;">Hệ thống Hỗ trợ Ra Quyết định Lâm sàng - Dự báo Tái nhập viện Đái tháo đường < 30 ngày</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. FORM NHẬP LIỆU CHUYÊN SÂU ---
with st.form("prediction_form"):
    st.subheader("📋 Thông tin Định danh Bệnh nhân")
    id_col1, id_col2 = st.columns(2)
    with id_col1: patient_name = st.text_input("Họ và tên bệnh nhân (Bắt buộc)", placeholder="Nhập họ và tên...")
    with id_col2: patient_id = st.text_input("Mã Hồ sơ Bệnh án (Bắt buộc)", placeholder="Nhập mã hồ sơ...")

    st.subheader("Thông tin Hành chính & Nhập viện")
    c1_1, c1_2, c1_3 = st.columns(3)
    age = c1_1.selectbox("Nhóm tuổi", [None, "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    gender = c1_2.selectbox("Giới tính", [None, "Male", "Female"])
    race = c1_3.selectbox("Chủng tộc", [None, "Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Missing"])
    
    c1_4, c1_5, c1_6 = st.columns(3)
    admission_type_id = c1_4.selectbox("Hình thức nhập viện (Type ID)", [None, "1", "2", "3", "4", "5", "6", "7", "8"])
    admission_source_id = c1_5.selectbox("Nguồn nhập viện (Source ID)", [None, "1", "7", "17", "4", "2", "Other"])
    discharge_disposition_id = c1_6.selectbox("Tình trạng xuất viện (Discharge ID)", [None, "1", "2", "3", "4", "5", "6", "7", "8", "22", "23", "24", "25", "27", "28"])

    st.subheader("Tiền sử & Sử dụng Dịch vụ Y tế")
    c2_1, c2_2, c2_3 = st.columns(3)
    time_in_hospital = c2_1.number_input("Số ngày nằm viện hiện tại", min_value=1, max_value=30, value=None, placeholder="Nhập số ngày...")
    total_visits = c2_2.number_input("Tổng số lượt khám trước đây tại viện", min_value=1, max_value=50, value=None, placeholder="Nhập số lượt...")
    number_inpatient = c2_3.number_input("Số lần điều trị nội trú (1 năm qua)", min_value=0, max_value=50, value=None, placeholder="Nhập số lần...")
    
    c2_4, c2_5 = st.columns(2)
    number_outpatient = c2_4.number_input("Số lần khám ngoại trú (1 năm qua)", min_value=0, max_value=50, value=None, placeholder="Nhập số lần...")
    number_emergency = c2_5.number_input("Số lần vào cấp cứu (1 năm qua)", min_value=0, max_value=50, value=None, placeholder="Nhập số lần...")

    st.subheader("Chẩn đoán & Xét nghiệm")
    icd_groups = [None, "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]
    c3_1, c3_2, c3_3 = st.columns(3)
    diag_1 = c3_1.selectbox("Chẩn đoán chính (ICD-9)", icd_groups)
    diag_2 = c3_2.selectbox("Chẩn đoán kèm theo 1", icd_groups)
    diag_3 = c3_3.selectbox("Chẩn đoán kèm theo 2", icd_groups)
    
    c3_4, c3_5, c3_6 = st.columns(3)
    number_diagnoses = c3_4.number_input("Tổng số chẩn đoán ghi nhận", min_value=1, max_value=25, value=None, placeholder="Nhập tổng số...")
    num_lab_procedures = c3_5.number_input("Số xét nghiệm Lab đã thực hiện", min_value=1, max_value=150, value=None, placeholder="Nhập số xét nghiệm...")
    num_procedures = c3_6.number_input("Số thủ thuật y tế (phẫu thuật/can thiệp)", min_value=0, max_value=15, value=None, placeholder="Nhập số thủ thuật...")

    st.subheader("Thuốc & Phác đồ Điều trị ĐTĐ")
    c4_1, c4_2, c4_3 = st.columns(3)
    diabetesMed = c4_1.selectbox("Có chỉ định dùng thuốc ĐTĐ không?", [None, "Yes", "No"])
    change = c4_2.selectbox("Có điều chỉnh liều/đổi thuốc không?", [None, "No", "Ch"])
    num_medications = c4_3.number_input("Tổng số loại thuốc đang sử dụng", min_value=1, max_value=100, value=None, placeholder="Nhập tổng số...")
    
    c4_4, c4_5, c4_6 = st.columns(3)
    num_active_meds = c4_4.number_input("Số loại thuốc ĐTĐ hoạt động", min_value=0, max_value=10, value=None, placeholder="Nhập số loại...")
    insulin = c4_5.selectbox("Tình trạng dùng Insulin", [None, "No", "Steady", "Up", "Down"])
    metformin = c4_6.selectbox("Tình trạng dùng Metformin", [None, "No", "Steady", "Up", "Down"])

    submitted = st.form_submit_button("🔍 Thực hiện Đánh giá & Lưu Hồ sơ", use_container_width=True)

# --- Hàm hỗ trợ điền giá trị mặc định nếu người dùng không nhập ---
def fill_default(val, default_val):
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default_val
    return val

# --- 5. XỬ LÝ DỮ LIỆU & DỰ BÁO ---
if submitted:
    # 1. Chỉ bắt buộc nhập Tên và Mã Bệnh Nhân
    if not patient_name.strip() or not patient_id.strip():
        st.warning("⚠️ Vui lòng nhập 'Họ và tên bệnh nhân' và 'Mã Hồ sơ Bệnh án' để tiếp tục.")
    else:
        # 2. Xử lý các giá trị None, điền giá trị an toàn/mặc định (0 cho số, 'Missing' cho text)
        safe_age = fill_default(age, "Missing")
        safe_gender = fill_default(gender, "Missing")
        safe_race = fill_default(race, "Missing")
        
        safe_admission_type_id = str(fill_default(admission_type_id, "Missing"))
        safe_admission_source_id = str(fill_default(admission_source_id, "Missing"))
        safe_discharge_disposition_id = str(fill_default(discharge_disposition_id, "Missing"))
        
        safe_time_in_hospital = fill_default(time_in_hospital, 1)
        safe_total_visits = fill_default(total_visits, 0)
        safe_number_inpatient = fill_default(number_inpatient, 0)
        safe_number_outpatient = fill_default(number_outpatient, 0)
        safe_number_emergency = fill_default(number_emergency, 0)
        
        safe_diag_1 = fill_default(diag_1, "Missing")
        safe_diag_2 = fill_default(diag_2, "Missing")
        safe_diag_3 = fill_default(diag_3, "Missing")
        
        safe_number_diagnoses = fill_default(number_diagnoses, 1)
        safe_num_lab_procedures = fill_default(num_lab_procedures, 0)
        safe_num_procedures = fill_default(num_procedures, 0)
        
        safe_diabetesMed = fill_default(diabetesMed, "Missing")
        safe_change = fill_default(change, "Missing")
        safe_num_medications = fill_default(num_medications, 0)
        safe_num_active_meds = fill_default(num_active_meds, 0)
        safe_insulin = fill_default(insulin, "Missing")
        safe_metformin = fill_default(metformin, "Missing")

        # Tính toán đặc trưng kỹ thuật
        service_utilization = int(safe_number_outpatient + safe_number_emergency + safe_number_inpatient)
        labs_per_day = float(safe_num_lab_procedures / max(safe_time_in_hospital, 1))

        # Khởi tạo dữ liệu thô
        raw_dict = {
            "age": safe_age, "gender": safe_gender, "race": safe_race,
            "admission_type_id": safe_admission_type_id,
            "admission_source_id": safe_admission_source_id,
            "discharge_disposition_id": safe_discharge_disposition_id,
            "time_in_hospital": safe_time_in_hospital,
            "total_visits": safe_total_visits,
            "service_utilization": service_utilization,
            "number_inpatient": safe_number_inpatient,
            "number_emergency": safe_number_emergency,
            "number_outpatient": safe_number_outpatient,
            "diag_1": safe_diag_1, "diag_2": safe_diag_2, "diag_3": safe_diag_3,
            "number_diagnoses": safe_number_diagnoses,
            "num_lab_procedures": safe_num_lab_procedures,
            "num_procedures": safe_num_procedures,
            "labs_per_day": labs_per_day,
            "diabetesMed": safe_diabetesMed,
            "change": safe_change,
            "num_medications": safe_num_medications,
            "num_active_meds": safe_num_active_meds,
            "insulin": safe_insulin,
            "metformin": safe_metformin
        }

        input_data = pd.DataFrame([raw_dict])
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_names]

        try:
            # Tiền xử lý cho SHAP
            if preprocessor is not None:
                transformed_data = preprocessor.transform(input_data)
                if hasattr(preprocessor, "get_feature_names_out"):
                    transformed_cols = preprocessor.get_feature_names_out()
                else:
                    transformed_cols = [f"Feature_{i}" for i in range(transformed_data.shape[1])]
                
                if hasattr(transformed_data, "toarray"):
                    transformed_data = transformed_data.toarray()
                shap_input = pd.DataFrame(transformed_data, columns=transformed_cols)
            else:
                shap_input = input_data

            # Dự báo xác suất
            if hasattr(model_to_predict, "predict_proba"):
                try:
                    prob_risk = model_to_predict.predict_proba(input_data)[0][1]
                except Exception:
                    prob_risk = model_to_predict.predict_proba(shap_input)[0][1]
            else:
                st.error("Mô hình không hỗ trợ tính năng tính toán xác suất (predict_proba).")
                st.stop()

            prob_percent = prob_risk * 100

            # Phân tầng theo chuẩn y khoa
            if prob_percent >= 25.0:
                risk_label = "Nguy cơ Cao (Cần can thiệp/theo dõi sát)"
                risk_color = "#DC2626"
                risk_advice = "Bệnh nhân có nguy cơ tái nhập viện dưới 30 ngày rất cao. Khuyến nghị hội chẩn lại phác đồ điều trị, hẹn lịch tái khám trong vòng 7 ngày và tư vấn quản lý đường huyết tại nhà."
            elif prob_percent >= 12.0:
                risk_label = "Nguy cơ Trung bình (Cần lưu ý khi xuất viện)"
                risk_color = "#D97706"
                risk_advice = "Bệnh nhân có nguy cơ ở mức cảnh báo. Cần hướng dẫn kỹ đơn thuốc xuất viện và dặn dò các dấu hiệu cảnh báo hạ/tăng đường huyết."
            else:
                risk_label = "Nguy cơ Thấp (An toàn)"
                risk_color = "#16A34A"
                risk_advice = "Tình trạng ổn định. Tiếp tục duy trì chế độ dinh dưỡng và phác đồ điều trị ngoại trú định kỳ."

            # Hiển thị kết quả
            st.divider()
            st.markdown(f"### 📊 Báo Cáo Lâm Sàng: **{patient_name}** ({patient_id})")
            
            res_col1, res_col2 = st.columns([1.2, 1.8])
            with res_col1:
                st.metric("Xác suất Tái nhập viện (< 30 ngày)", f"{prob_percent:.1f}%")
                st.markdown(f"<div style='font-size: 18px; font-weight: bold; color: {risk_color}; margin-top: 8px;'>{risk_label}</div>", unsafe_allow_html=True)
                st.info(f"**Khuyến nghị chuyên môn:**\n\n{risk_advice}")

                st.markdown("#### 🚨 Các chỉ số cần lưu ý:")
                warnings_list = []
                if safe_number_inpatient > 0:
                    warnings_list.append(f"Có **{safe_number_inpatient}** lần điều trị nội trú trong năm qua.")
                if safe_number_emergency > 0:
                    warnings_list.append(f"Có **{safe_number_emergency}** lần nhập viện cấp cứu trong năm qua.")
                if safe_num_active_meds >= 3:
                    warnings_list.append(f"Sử dụng nhiều loại thuốc ĐTĐ kết hợp (**{safe_num_active_meds}** loại).")
                if safe_time_in_hospital > 7:
                    warnings_list.append(f"Thời gian nằm viện kéo dài (**{safe_time_in_hospital}** ngày).")
                if safe_discharge_disposition_id not in ["1", "6", "8", "Missing"]:
                     warnings_list.append(f"Tình trạng xuất viện tiềm ẩn rủi ro (Mã: {safe_discharge_disposition_id}).")

                if warnings_list:
                    for warning in warnings_list:
                        st.warning(warning)
                else:
                    st.success("Không phát hiện chỉ số lâm sàng bất thường nghiêm trọng.")

            with res_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_percent,
                    number={'suffix': "%"},
                    title={'text': "Thước Đo Mức Độ Rủi Ro Tái Nhập Viện", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 12], 'color': "#DCFCE7"},
                            {'range': [12, 25], 'color': "#FEF3C7"},
                            {'range': [25, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.8,
                            'value': prob_percent
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # GIẢI THÍCH MÔ HÌNH BẰNG SHAP LOCAL EXPLANATION
            if explainer is not None:
                st.divider()
                st.markdown("#### 🔬 Giải thích Khuyến nghị (Explainable AI bằng SHAP)")
                st.markdown("Biểu đồ thể hiện mức độ tác động của từng đặc trưng lâm sàng đẩy xác suất rủi ro **Lên cao (Đỏ)** hoặc **Xuống thấp (Xanh)** đối với bệnh nhân này.")
                
                try:
                    safe_cols = []
                    for c in shap_input.columns:
                        safe_c = re.sub(r'[\[\]<>]', '_', c)
                        safe_c = safe_c.replace("cat__", "").replace("num__", "")
                        safe_cols.append(safe_c)
                    shap_input.columns = safe_cols

                    shap_values = explainer.shap_values(shap_input)
                    
                    if isinstance(shap_values, list):
                        shap_values_to_plot = shap_values[1][0] 
                    elif len(shap_values.shape) == 3: 
                         shap_values_to_plot = shap_values[0, :, 1]
                    else:
                        shap_values_to_plot = shap_values[0] 

                    if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                        expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                    else:
                         expected_value = explainer.expected_value

                    explanation = shap.Explanation(values=shap_values_to_plot, 
                                                   base_values=expected_value, 
                                                   data=shap_input.iloc[0], 
                                                   feature_names=safe_cols)
                    
                    # [ĐÃ SỬA LỖI HIỂN THỊ WATERFALL]
                    # Khởi tạo plt figure và gọi biểu đồ SHAP vào, sau đó gửi cho Streamlit
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Sử dụng matplotlib mode của SHAP waterfall, không hiển thị trực tiếp (show=False)
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    # Thiết lập thông số điều chỉnh layout cho hình ảnh để không bị cắt chữ
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    
                except Exception as e:
                    st.info(f"Không thể vẽ biểu đồ SHAP cho hồ sơ này. Chi tiết lỗi hệ thống: {e}")

            # Lấy giờ Việt Nam
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            current_time = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M")

            st.session_state.patient_history.insert(0, {
                "Thời gian": current_time,
                "Mã BN": patient_id,
                "Họ và Tên": patient_name,
                "Xác suất": f"{prob_percent:.1f}%",
                "Phân tầng": risk_label.split(" (")[0]
            })

        except Exception as e:
            st.error(f"Lỗi xử lý dự báo: {e}")

# --- 6. BẢNG THEO DÕI HỒ SƠ ĐÃ ĐÁNH GIÁ ---
if st.session_state.patient_history:
    st.divider()
    st.subheader("📑 Lịch sử Đánh giá trong Phiên làm việc")
    st.dataframe(pd.DataFrame(st.session_state.patient_history), use_container_width=True, hide_index=True)
    if st.button("🗑️ Xóa danh sách lịch sử"):
        st.session_state.patient_history = []
        st.rerun()

st.caption("Công cụ hỗ trợ nghiên cứu và tham khảo lâm sàng, không thay thế chẩn đoán chuyên môn của bác sĩ.")