import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ----------------------------------------
st.set_page_config(
    page_title="E-Commerce Delivery Prediction",
    page_icon="🚚",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# 2. ตกแต่งแถบด้านข้าง (Sidebar)
# ----------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2769/2769339.png", width=120)
    st.title("เกี่ยวกับโปรเจค")
    st.info("🎯 **เป้าหมาย:**\nนำ Machine Learning มาวิเคราะห์ปัจจัยต่างๆ เพื่อคาดการณ์ความเสี่ยงที่สินค้าจะจัดส่งล่าช้า")
    st.markdown("---")
    st.markdown("💡 **Model:** Gradient Boosting (Tuned Pipeline)")
    st.markdown("📊 **Focus Metric:** Recall & Probability")
    
# ----------------------------------------
# 3. ส่วนหัวของหน้าหลัก (Header)
# ----------------------------------------
st.title("🚚 AI Predictor: ตรวจสอบสถานะการจัดส่งสินค้า")
st.markdown("กรอกข้อมูลรายละเอียดออเดอร์ด้านล่าง เพื่อให้ AI ประเมินความเสี่ยงในการจัดส่ง")
st.markdown("---")

# ----------------------------------------
# 4. โหลดไฟล์สมอง AI (เวอร์ชันกัน Error 100%)
# ----------------------------------------
@st.cache_resource 
def load_models():
    try:
        pipeline = joblib.load('shipping_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าอัปโหลดไฟล์ .pkl ครบถ้วนแล้ว ({e})")
        return None

pipeline = load_models()

# ----------------------------------------
# 5. สร้างส่วนรับข้อมูล (UI)
# ----------------------------------------
if pipeline is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("📦 1. ข้อมูลสินค้า (Product Details)")
            weight = st.number_input("⚖️ น้ำหนักสินค้า (กรัม)", min_value=0, value=2000, step=100)
            cost = st.number_input("💵 ราคาสินค้า (USD)", min_value=0, value=150, step=10)
            discount = st.number_input("🏷️ ส่วนลดที่ได้รับ (%)", min_value=0, value=10, step=1)
            importance = st.selectbox("⭐ ความสำคัญของสินค้า", ['low', 'medium', 'high'])
    
    with col2:
        with st.container(border=True):
            st.subheader("🏢 2. ข้อมูลการจัดส่งและลูกค้า")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                warehouse = st.selectbox("🏭 คลังสินค้า", ['A', 'B', 'C', 'D', 'F'])
            with col2_2:
                shipment_mode = st.selectbox("✈️ วิธีจัดส่ง", ['Flight', 'Ship', 'Road'])
                
            care_calls = st.slider("📞 ลูกค้าโทรติดตามของ (ครั้ง)", 0, 10, 3)
            prior_purchases = st.slider("🛍️ ลูกค้าเคยสั่งของ (ครั้ง)", 0, 15, 3)
            
            col2_3, col2_4 = st.columns(2)
            with col2_3:
                rating = st.radio("🌟 คะแนนรีวิวเดิม", [1, 2, 3, 4, 5], horizontal=True)
            with col2_4:
                gender = st.radio("👤 เพศลูกค้า", ['Female', 'Male'], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True) 
    
    # ----------------------------------------
    # 6. ปุ่มทำนายผล (แก้ไขระบบกัน Error ชื่อคอลัมน์)
    # ----------------------------------------
    if st.button("🚀 เริ่มวิเคราะห์ความเสี่ยงการจัดส่ง (Predict)", type="primary", use_container_width=True):
        st.markdown("---")
        
        gender_map = {'Female': 0, 'Male': 1}
        importance_map = {'low': 0, 'medium': 1, 'high': 2}
        
        # 🟢 ดึงรายชื่อตัวแปรที่ AI ต้องการจริงๆ จากโมเดลโดยตรง
        expected_features = pipeline.feature_names_in_
        input_data = {col: 0 for col in expected_features}
        
        # 🟢 จับคู่ข้อมูลเข้าด้วยกันแบบยืดหยุ่น (แก้บัค File Out of Sync)
        for col in expected_features:
            name = col.lower().strip()
            if 'care_calls' in name:
                input_data[col] = care_calls
            elif 'rating' in name:
                input_data[col] = rating
            elif 'cost' in name:
                input_data[col] = cost
            elif 'prior_purchases' in name:
                input_data[col] = prior_purchases
            elif 'importance' in name:
                input_data[col] = importance_map[importance]
            elif 'gender' in name:
                input_data[col] = gender_map[gender]
            elif 'discount' in name:
                input_data[col] = discount
            elif 'weight' in name:
                input_data[col] = weight
            elif f'warehouse_block_{warehouse}'.lower() == name:
                input_data[col] = 1
            elif f'mode_of_shipment_{shipment_mode}'.lower() == name:
                input_data[col] = 1
                
        # แปลงเป็น DataFrame และบังคับเรียงคอลัมน์ให้เป๊ะ
        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_features]
        
        # ใช้ Pipeline ทำนาย
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0] 
        
        # ----------------------------------------
        # 7. แสดงผลลัพธ์
        # ----------------------------------------
        st.subheader("🎯 ผลการทำนายจากระบบ AI:")
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div style='background-color: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
                <h3 style='color: #ff4b4b; margin:0;'>⚠️ มีแนวโน้มว่าสินค้านี้จะจัดส่ง "ล่าช้า" (Late Delivery)</h3>
                <p style='color: #333; margin-top:10px;'><b>💡 คำแนะนำเชิงรุก:</b> ออเดอร์นี้มีความเสี่ยงสูง แนะนำให้ฝ่ายปฏิบัติการจัดคิวพิเศษ หรือแจ้งเตือนลูกค้าล่วงหน้าเพื่อรักษาความพึงพอใจ</p>
                <hr style='border-top: 1px solid #ffb3b3;'>
                <h4 style='color: #cc0000; margin:0;'>📊 ความมั่นใจของ AI: {probability[1]*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #e6ffe6; padding: 20px; border-radius: 10px; border-left: 5px solid #00cc66;'>
                <h3 style='color: #00cc66; margin:0;'>✅ มีแนวโน้มว่าสินค้านี้จะจัดส่ง "ตรงเวลา" (On Time)</h3>
                <p style='color: #333; margin-top:10px;'><b>💡 สภาพปกติ:</b> สามารถดำเนินการจัดส่งตามรอบปกติได้เลย</p>
                <hr style='border-top: 1px solid #b3ffcc;'>
                <h4 style='color: #00994d; margin:0;'>📊 ความมั่นใจของ AI: {probability[0]*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

        # ----------------------------------------
        # 8. กราฟ Feature Importance 
        # ----------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 ปัจจัยหลักที่มีผลต่อออเดอร์นี้ (Feature Importance)")
        st.info("กราฟแสดงข้อมูลว่า AI ให้น้ำหนักกับปัจจัยใดมากที่สุดในการตัดสินใจครั้งนี้")
        
        gb_model = pipeline.named_steps['gb']
        importances = gb_model.feature_importances_
        
        feat_df = pd.DataFrame({'Feature': expected_features, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
        ax.set_title("Top 5 Key Factors for this Prediction", fontsize=12)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        st.pyplot(fig)
