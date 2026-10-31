import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

# =========================================================
# 1. CẤU HÌNH TRANG & PHONG CÁCH HỌC THUẬT
# =========================================================
st.set_page_config(
    page_title="CDSS - Dự báo Tái nhập viện Đái tháo đường",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {font-size: 26px; font-weight: 700; color: #1E3A8A; margin-bottom: 4px;}
    .sub-header {font-size: 15px; color: #475569; margin-bottom: 20px;}
    .form-title {font-size: 17px; font-weight: 600; color: #0F172A; border-left: 4px solid #2563EB; padding-left: 10px; margin-top: 10px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. TỪ ĐIỂN ÁNH XẠ THUẬT NGỮ Y KHOA (VIỆT -> MÃ DATASET)
# =========================================================
MAP_GENDER = {"Nữ (Female)": "Female", "Nam (Male)": "Male"}

MAP_RACE = {
    "Người da trắng (Caucasian)": "Caucasian",
    "Người Mỹ gốc Phi (AfricanAmerican)": "AfricanAmerican",
    "Người gốc Tây Ban Nha (Hispanic)": "Hispanic",
    "Người châu Á (Asian)": "Asian",
    "Khác / Chưa ghi nhận (Other/Missing)": "Other"
}

MAP_ADMISSION_TYPE = {
    "1 - Cấp cứu (Emergency)": "1",
    "2 - Khẩn cấp (Urgent)": "2",
    "3 - Nhập viện có kế hoạch (Elective)": "3",
    "5 - Chưa xác định (Not Available)": "5",
    "6 - Khác (NULL)": "6"
}

MAP_ADMISSION_SOURCE = {
    "7 - Từ khoa Cấp cứu (Emergency Room)": "7",
    "1 - Bác sĩ giới thiệu (Physician Referral)": "1",
    "4 - Chuyển từ bệnh viện khác (Transfer from hospital)": "4",
    "6 - Chuyển từ cơ sở y tế khác (Transfer from another facility)": "6",
    "2 - Phòng khám giới thiệu (Clinic Referral)": "2",
    "5 - Chuyển từ cơ sở điều dưỡng (SNF)": "5",
    "17 - Không rõ thông tin": "17"
}

MAP_DISCHARGE = {
    "1 - Xuất viện về nhà tự chăm sóc (Discharged to home)": "1",
    "3 - Chuyển đến cơ sở điều dưỡng chuyên môn (SNF)": "3",
    "6 - Xuất viện về nhà có dịch vụ y tế theo dõi (Home with health service)": "6",
    "22 - Chuyển đến cơ sở phục hồi chức năng (Rehab facility)": "22",
    "2 - Chuyển sang bệnh viện ngắn hạn khác": "2",
    "25 - Chưa phân loại": "25"
}

MAP_ICD9 = {
    "Bệnh hệ tuần hoàn / Tim mạch (Circulatory: 390-459, 785)": "Circulatory",
    "Đái tháo đường (Diabetes: 250.xx)": "Diabetes",
    "Bệnh hệ hô hấp (Respiratory: 460-519, 786)": "Respiratory",
    "Bệnh hệ tiêu hóa (Digestive: 520-579, 787)": "Digestive",
    "Chấn thương & Ngộ độc (Injury: 800-999)": "Injury",
    "Bệnh hệ tiết niệu - sinh dục (Genitourinary: 580-629, 788)": "Genitourinary",
    "Bệnh cơ xương khớp (Musculoskeletal: 710-739)": "Musculoskeletal",
    "Khối u / Ung bướu (Neoplasms: 140-239)": "Neoplasms",
    "Bệnh lý khác (Other)": "Other"
}

MAP_GLU = {
    "Chưa thực hiện xét nghiệm (None)": "None",
    "Bình thường (Norm)": "Norm",
    "Cao > 200 mg/dL (>200)": ">200",
    "Rất cao > 300 mg/dL (>300)": ">300"
}

MAP_A1C = {
    "Chưa thực hiện xét nghiệm (None)": "None",
    "Bình thường < 7% (Norm)": "Norm",
    "Cao > 7% (>7)": ">7",
    "Rất cao > 8% - Kiểm soát kém (>8)": ">8"
}

MAP_MED_STATUS = {
    "Không sử dụng (No)": "No",
    "Giữ nguyên liều (Steady)": "Steady",
    "Tăng liều (Up)": "Up",
    "Giảm liều (Down)": "Down"
}

MAP_CHANGE = {"Không thay đổi (No)": "No", "Có thay đổi liều/thuốc (Ch)": "Ch"}
MAP_DIABETES_MED = {"Có chỉ định (Yes)": "Yes", "Không chỉ định (No)": "No"}

FEATURE_VI_NAMES = {
    "num__number_inpatient": "Số lần điều trị nội trú (1 năm qua)",
    "num__number_emergency": "Số lần cấp cứu (1 năm qua)",
    "num__number_outpatient": "Số lần khám ngoại trú (1 năm qua)",
    "num__time_in_hospital": "Số ngày nằm viện hiện tại",
    "num__num_medications": "Tổng số loại thuốc điều trị",
    "num__num_lab_procedures": "Số lượng xét nghiệm cận lâm sàng",
    "num__num_procedures": "Số lượng thủ thuật can thiệp",
    "num__number_diagnoses": "Tổng số bệnh lý chẩn đoán kèm theo"
}

def translate_feature_name(raw_name):
    if raw_name in FEATURE_VI_NAMES:
        return FEATURE_VI_NAMES[raw_name]
    clean = raw_name.replace("cat__", "").replace("num__", "")
    clean = clean.replace("discharge_disposition_id_", "Hình thức xuất viện mã ")
    clean = clean.replace("admission_type_id_", "Loại nhập viện mã ")
    clean = clean.replace("admission_source_id_", "Nguồn nhập viện mã ")
    clean = clean.replace("diag_1_", "Chẩn đoán chính: ")
    clean = clean.replace("diag_2_", "Bệnh kèm theo 1: ")
    clean = clean.replace("diag_3_", "Bệnh kèm theo 2: ")
    clean = clean.replace("age_", "Nhóm tuổi ")
    clean = clean.replace("insulin_", "Insulin: ")
    clean = clean.replace("metformin_", "Metformin: ")
    clean = clean.replace("A1Cresult_", "Xét nghiệm HbA1c: ")
    clean = clean.replace("max_glu_serum_", "Đường huyết tối đa: ")
    clean = clean.replace("diabetesMed_", "Thuốc tiểu đường: ")
    clean = clean.replace("change_", "Thay đổi phác đồ: ")
    return clean

# =========================================================
# 3. TẢI MÔ HÌNH ĐÃ ĐÓNG GÓI
# =========================================================
@st.cache_resource
def load_model_bundle():
    return joblib.load("kltn_diabetes_model.joblib")

try:
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    calibrator = bundle["calibrator"]
    selected_cols = bundle["selected_columns"]
except Exception as e:
    st.error(
        f"⚠️ Không tìm thấy file `kltn_diabetes_model.joblib` trong thư mục hiện tại. "
        f"Bạn hãy copy file `kltn_diabetes_model.joblib` vào đúng thư mục `KLTN-App` rồi tải lại trang. Chi tiết: {e}"
    )
    st.stop()

# =========================================================
# 4. THANH BÊN (THÔNG TIN HỌC THUẬT CỦA ĐỀ TÀI)
# =========================================================
with st.sidebar:
    st.markdown("### 🎓 KHÓA LUẬN TỐT NGHIỆP")
    st.info(
        "**Đề tài:** Dự báo nguy cơ tái nhập viện trong 30 ngày của bệnh nhân đái tháo đường bằng Học máy có giải thích (Explainable AI)."
    )
    st.markdown("**Cấu hình Mô hình Sản phẩm:**")
    st.markdown("- **Thuật toán lõi:** `XGBoost (Weighted)`")
    st.markdown("- **Hiệu chuẩn xác suất:** `Isotonic Calibration`")
    st.markdown("- **Chỉ số ROC-AUC:** `0.6711`")
    st.markdown("- **Recall (Nhóm nguy cơ):** `58.74%`")
    st.markdown("- **Tỷ lệ nền toàn viện:** `11.16%`")
    st.markdown("---")
    st.caption("Dữ liệu huấn luyện: UCI Diabetes 130-US Hospitals (101,766 hồ sơ bệnh án, tách độc lập theo mã bệnh nhân).")

# =========================================================
# 5. GIAO DIỆN CHÍNH: ĐA TAB CHUYÊN NGHIỆP
# =========================================================
st.markdown('<div class="main-header">🏥 HỆ THỐNG ĐÁNH GIÁ & DỰ BÁO NGUY CƠ TÁI NHẬP VIỆN (30 NGÀY)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Công cụ lượng giá xác suất tái nhập viện và giải thích cá nhân hóa bằng SHAP dành cho nhân viên y tế</div>', unsafe_allow_html=True)

tab_predict, tab_academic = st.tabs([
    "🩺 1. Biểu mẫu Lâm sàng & Dự báo Xác suất (%)",
    "📊 2. Hồ sơ Học thuật & Thực nghiệm Mô hình"
])

with tab_predict:
    with st.form("clinical_multi_form"):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown('<div class="form-title">📋 Form 1: Hồ sơ Nhân khẩu học & Nhập viện</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.selectbox("Nhóm tuổi", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
            with c2:
                gender_label = st.selectbox("Giới tính", list(MAP_GENDER.keys()))
            with c3:
                race_label = st.selectbox("Chủng tộc", list(MAP_RACE.keys()))

            admission_type_label = st.selectbox(" Phân loại lúc nhập viện", list(MAP_ADMISSION_TYPE.keys()))
            admission_source_label = st.selectbox("Nguồn tiếp nhận bệnh nhân", list(MAP_ADMISSION_SOURCE.keys()))
            discharge_label = st.selectbox("Dự kiến hướng xử trí xuất viện", list(MAP_DISCHARGE.keys()))
            time_in_hospital = st.slider("Thời gian điều trị nội trú hiện tại (Ngày)", 1, 14, 4)

            st.markdown('<div class="form-title">🕒 Form 2: Tiền sử Sử dụng Dịch vụ Y tế (1 năm qua)</div>', unsafe_allow_html=True)
            h1, h2, h3 = st.columns(3)
            with h1:
                number_inpatient = st.number_input("Số lần nằm Nội trú", min_value=0, max_value=30, value=0)
            with h2:
                number_emergency = st.number_input("Số lần Cấp cứu", min_value=0, max_value=30, value=0)
            with h3:
                number_outpatient = st.number_input("Số lần khám Ngoại trú", min_value=0, max_value=40, value=0)

        with col_f2:
            st.markdown('<div class="form-title">🔬 Form 3: Tình trạng Bệnh lý & Cận lâm sàng hiện tại</div>', unsafe_allow_html=True)
            diag_1_label = st.selectbox("Chẩn đoán chính (Bệnh lý nguyên nhân)", list(MAP_ICD9.keys()), index=0)
            d2_col, d3_col = st.columns(2)
            with d2_col:
                diag_2_label = st.selectbox("Bệnh lý đi kèm thứ 1", list(MAP_ICD9.keys()), index=1)
            with d3_col:
                diag_3_label = st.selectbox("Bệnh lý đi kèm thứ 2", list(MAP_ICD9.keys()), index=2)

            l1, l2 = st.columns(2)
            with l1:
                a1c_label = st.selectbox("Chỉ số đường huyết HbA1c (A1Cresult)", list(MAP_A1C.keys()))
            with l2:
                glu_label = st.selectbox("Đường huyết thanh tối đa (Max Glucose)", list(MAP_GLU.keys()))

            p1, p2, p3 = st.columns(3)
            with p1:
                number_diagnoses = st.number_input("Tổng số mã bệnh", min_value=1, max_value=16, value=7)
            with p2:
                num_lab_procedures = st.number_input("Số xét nghiệm Lab", min_value=1, max_value=135, value=45)
            with p3:
                num_procedures = st.number_input("Số thủ thuật Y khoa", min_value=0, max_value=6, value=1)

            st.markdown('<div class="form-title">💊 Form 4: Phác đồ Thuốc & Điều chỉnh Liều lượng</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                insulin_label = st.selectbox("Phác đồ Insulin hiện tại", list(MAP_MED_STATUS.keys()), index=1)
                change_label = st.selectbox("Thay đổi liều lượng thuốc", list(MAP_CHANGE.keys()))
            with m2:
                metformin_label = st.selectbox("Phác đồ Metformin hiện tại", list(MAP_MED_STATUS.keys()), index=0)
                diabetes_med_label = st.selectbox("Sử dụng thuốc đặc trị tiểu đường", list(MAP_DIABETES_MED.keys()))

            num_medications = st.slider("Tổng số loại thuốc được kê trong đợt điều trị", 1, 80, 15)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔍 PHÂN TÍCH HỒ SƠ & DỰ BÁO XÁC SUẤT TÁI NHẬP VIỆN (%)", use_container_width=True)

    # =========================================================
    # 6. HIỂN THỊ KẾT QUẢ PHÂN TÍCH & GIẢI THÍCH SHAP
    # =========================================================
    if submit_btn:
        input_df = pd.DataFrame([{
            "age": age,
            "gender": MAP_GENDER[gender_label],
            "race": MAP_RACE[race_label],
            "admission_type_id": MAP_ADMISSION_TYPE[admission_type_label],
            "admission_source_id": MAP_ADMISSION_SOURCE[admission_source_label],
            "discharge_disposition_id": MAP_DISCHARGE[discharge_label],
            "time_in_hospital": time_in_hospital,
            "number_inpatient": number_inpatient,
            "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
            "diag_1": MAP_ICD9[diag_1_label],
            "diag_2": MAP_ICD9[diag_2_label],
            "diag_3": MAP_ICD9[diag_3_label],
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": MAP_GLU[glu_label],
            "A1Cresult": MAP_A1C[a1c_label],
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "insulin": MAP_MED_STATUS[insulin_label],
            "metformin": MAP_MED_STATUS[metformin_label],
            "change": MAP_CHANGE[change_label],
            "diabetesMed": MAP_DIABETES_MED[diabetes_med_label],
            "num_medications": num_medications
        }])[selected_cols]

        # 1. Tính xác suất thực tế đã hiệu chuẩn và xác suất thô
        prob_cal = float(calibrator.predict_proba(input_df)[0, 1]) * 100
        prob_raw = float(pipeline.predict_proba(input_df)[0, 1]) * 100
        baseline_rate = 11.16
        relative_ratio = prob_cal / baseline_rate

        if prob_cal >= 22.0:
            risk_badge = "🔴 NGUY CƠ CAO"
            recommendation = (
                "Bệnh nhân có xác suất tái nhập viện sớm cao gấp nhiều lần mức trung bình của bệnh viện. "
                "Khuyến nghị hội chẩn kỹ trước xuất viện, rà soát lại phác đồ Insulin/đường huyết và hẹn tái khám sớm trong 7 ngày."
            )
            gauge_color = "#DC2626"
        elif prob_cal >= 12.0:
            risk_badge = "🟡 NGUY CƠ TRUNG BÌNH"
            recommendation = (
                "Nguy cơ tái nhập viện cao hơn mức nền chung (11.16%). "
                "Cần hướng dẫn kỹ chế độ dùng thuốc tại nhà và theo dõi các triệu chứng bệnh lý đi kèm."
            )
            gauge_color = "#D97706"
        else:
            risk_badge = "🟢 NGUY CƠ THẤP (AN TOÀN)"
            recommendation = (
                "Tình trạng hiện tại ổn định, xác suất tái nhập viện trong 30 ngày thấp hơn mức trung bình toàn viện. "
                "Có thể xuất viện và tái khám định kỳ theo lịch hẹn chuẩn."
            )
            gauge_color = "#16A34A"

        st.markdown("---")
        st.subheader("📌 KẾT QUẢ LƯỢNG GIÁ NGUY CƠ TÁI NHẬP VIỆN (< 30 NGÀY)")

        res_col1, res_col2 = st.columns([1.1, 1.3])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_cal,
                number={"suffix": "%", "font": {"size": 42}},
                delta={"reference": baseline_rate, "increasing": {"color": "#DC2626"}, "decreasing": {"color": "#16A34A"}, "suffix": "% so với mức nền"},
                title={"text": "Xác suất tái nhập viện (Đã hiệu chuẩn Y tế)", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 60], "ticksuffix": "%"},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 12], "color": "#DCFCE7"},
                        {"range": [12, 22], "color": "#FEF3C7"},
                        {"range": [22, 60], "color": "#FEE2E2"}
                    ],
                    "threshold": {"line": {"color": "#1E293B", "width": 3}, "thickness": 0.8, "value": baseline_rate}
                }
            ))
            fig_gauge.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Xác suất Thực tế", f"{prob_cal:.2f}%")
            m_col2.metric("So với Tỷ lệ nền (11.16%)", f"Gấp {relative_ratio:.2f} lần")
            m_col3.metric("Điểm rủi ro mô hình (Raw)", f"{prob_raw:.1f}%")

            st.markdown(f"#### Phân tầng cảnh báo: {risk_badge}")
            st.info(f"**Khuyến nghị lâm sàng:** {recommendation}")

        # 2. Giải thích cá nhân hóa bằng SHAP (Trả lời câu hỏi: Vì triệu chứng/tình trạng nào?)
        st.markdown("---")
        st.subheader("🧬 GIẢI THÍCH NGUYÊN NHÂN CÁ NHÂN HÓA (SHAP EXPLAINABILITY)")
        st.caption("Phân tích định lượng mức độ đóng góp của từng yếu tố lâm sàng hiện tại vào kết quả dự báo của riêng bệnh nhân này:")

        preprocessor = pipeline.named_steps["preprocessor"]
        selector = pipeline.named_steps["selector"]
        xgb_model = pipeline.named_steps["model"]

        X_proc = selector.transform(preprocessor.transform(input_df))
        if hasattr(X_proc, "toarray"):
            X_proc_dense = X_proc.toarray()
        else:
            X_proc_dense = np.asarray(X_proc)

        raw_feature_names = selector.get_feature_names_out(preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X_proc_dense)[0]

        # Chỉ lọc các đặc trưng đang kích hoạt của bệnh nhân (giá trị khác 0) hoặc các biến định lượng
        active_indices = [
            i for i, (fname, val) in enumerate(zip(raw_feature_names, X_proc_dense[0]))
            if fname.startswith("num__") or val > 0
        ]

        shap_df = pd.DataFrame({
            "Yếu tố lâm sàng": [translate_feature_name(raw_feature_names[i]) for i in active_indices],
            "Mức tác động (SHAP)": [shap_vals[i] for i in active_indices],
            "Abs_SHAP": [abs(shap_vals[i]) for i in active_indices]
        }).sort_values(by="Abs_SHAP", ascending=False).head(10).sort_values(by="Mức tác động (SHAP)", ascending=True)

        shap_col1, shap_col2 = st.columns([1.3, 1.0])

        with shap_col1:
            colors = ["#DC2626" if v > 0 else "#2563EB" for v in shap_df["Mức tác động (SHAP)"]]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["Mức tác động (SHAP)"],
                y=shap_df["Yếu tố lâm sàng"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["Mức tác động (SHAP)"]],
                textposition="auto"
            ))
            fig_shap.update_layout(
                title="Top 10 yếu tố làm TĂNG (Đỏ) hoặc GIẢM (Xanh) nguy cơ tái nhập viện",
                xaxis_title="Giá trị tác động SHAP (Log-odds)",
                yaxis_title="",
                height=380,
                margin=dict(l=10, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with shap_col2:
            shap_table = shap_df.sort_values(by="Abs_SHAP", ascending=False)[["Yếu tố lâm sàng", "Mức tác động (SHAP)"]].copy()
            shap_table["Chiều hướng tác động"] = shap_table["Mức tác động (SHAP)"].apply(
                lambda x: "⬆️ Làm tăng nguy cơ" if x > 0 else "⬇️ Làm giảm nguy cơ"
            )
            shap_table["Mức tác động (SHAP)"] = shap_table["Mức tác động (SHAP)"].round(4)
            st.dataframe(shap_table, use_container_width=True, hide_index=True)

with tab_academic:
    st.subheader("📚 Tổng quan Thực nghiệm & Phương pháp Học thuật trong Khóa luận")
    st.markdown("""
    1. **Kiểm soát Rò rỉ Dữ liệu (Data Leakage Prevention):** Sử dụng `StratifiedGroupKFold` phân tách theo mã định danh bệnh nhân (`patient_nbr`), đảm bảo 57,231 bệnh nhân tập Huấn luyện và 14,287 bệnh nhân tập Kiểm thử hoàn toàn độc lập.
    2. **Gom nhóm Chẩn đoán Lâm sàng ICD-9:** Chuẩn hóa hơn 700 mã bệnh rời rạc về 9 nhóm bệnh lý y khoa chuẩn, giảm số chiều sau mã hóa One-Hot từ 2,425 xuống còn 277 đặc trưng.
    3. **Xử lý Mất cân bằng & Hiệu chuẩn Xác suất (`Isotonic Calibration`):** Kết hợp trọng số phạt lớp thiểu số (`scale_pos_weight ≈ 8.02`) trên XGBoost giúp phát hiện gần **59%** số ca tái nhập viện thực tế, đồng thời hiệu chuẩn xác suất đầu ra về đúng phân phối thực tế của bệnh viện (~11.16%).
    """)

    st.markdown("#### Bảng đối sánh hiệu năng trên Tập Kiểm thử Độc lập (20,353 ca)")
    benchmark_df = pd.DataFrame([
        {"Mô hình": "XGBoost (Weighted + Calibrated) - Mô hình Sản phẩm", "ROC-AUC": 0.6711, "Recall (Class 1)": "58.74%", "Precision (Class 1)": "18.25%", "F1-score (Class 1)": 0.2785},
        {"Mô hình": "Logistic Regression (Balanced)", "ROC-AUC": 0.6680, "Recall (Class 1)": "54.37%", "Precision (Class 1)": "18.61%", "F1-score (Class 1)": 0.2773},
        {"Mô hình": "Random Forest (Balanced Subsample)", "ROC-AUC": 0.6680, "Recall (Class 1)": "56.86%", "Precision (Class 1)": "18.19%", "F1-score (Class 1)": 0.2756},
        {"Mô hình": "KNN Classifier (Baseline)", "ROC-AUC": 0.5537, "Recall (Class 1)": "3.26%", "Precision (Class 1)": "21.90%", "F1-score (Class 1)": 0.0567},
        {"Mô hình": "Decision Tree Classifier", "ROC-AUC": 0.5212, "Recall (Class 1)": "15.77%", "Precision (Class 1)": "15.05%", "F1-score (Class 1)": 0.1540}
    ])
    st.table(benchmark_df)