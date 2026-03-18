import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ----------------------------------------
st.set_page_config(
    page_title="E-Commerce Delivery Prediction",
    page_icon="📦",
    layout="centered"
)

st.title("📦 E-Commerce Delivery Prediction")
st.markdown("แอปพลิเคชันสำหรับทำนายสถานะการจัดส่งสินค้า ว่าจะ **ตรงเวลา** หรือ **ล่าช้า**")
st.markdown("---")

# ----------------------------------------
# 2. โหลดไฟล์สมอง AI (Model, Scaler, Features)
# ----------------------------------------
@st.cache_resource # ให้เว็บโหลดแค่ครั้งเดียว จะได้ไม่ช้า
def load_models():
    try:
        model = joblib.load('shipping_model.pkl')
        scaler = joblib.load('shipping_scaler.pkl')
        features = joblib.load('shipping_features.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าอัปโหลดไฟล์ .pkl ครบถ้วนแล้ว ({e})")
        return None, None, None

model, scaler, expected_features = load_models()

# ----------------------------------------
# 3. สร้างส่วนรับข้อมูลจากผู้ใช้งาน (User Inputs)
# ----------------------------------------
if model is not None:
    st.header("📝 กรอกข้อมูลออเดอร์สินค้า")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ข้อมูลสินค้า")
        weight = st.number_input("น้ำหนักสินค้า (กรัม)", min_value=0, value=2000, step=100)
        cost = st.number_input("ราคาสินค้า (USD)", min_value=0, value=150, step=10)
        discount = st.number_input("ส่วนลดที่ได้รับ (%)", min_value=0, value=10, step=1)
        importance = st.selectbox("ความสำคัญของสินค้า", ['low', 'medium', 'high'])
    
    with col2:
        st.subheader("ข้อมูลการจัดส่งและลูกค้า")
        warehouse = st.selectbox("คลังสินค้าที่ส่งออก", ['A', 'B', 'C', 'D', 'F'])
        shipment_mode = st.selectbox("วิธีการจัดส่ง", ['Flight', 'Ship', 'Road'])
        care_calls = st.slider("จำนวนครั้งที่ลูกค้าโทรติดตาม", min_value=0, max_value=10, value=3)
        prior_purchases = st.slider("จำนวนครั้งที่ลูกค้าเคยสั่งซื้อ", min_value=0, max_value=15, value=3)
        rating = st.radio("คะแนนรีวิวของลูกค้า (1=แย่, 5=ดี)", [1, 2, 3, 4, 5], horizontal=True)
        gender = st.radio("เพศของลูกค้า", ['Female', 'Male'], horizontal=True)

    st.markdown("---")
    
    # ----------------------------------------
    # 4. ปุ่มกดเพื่อทำนาย (Prediction Button)
    # ----------------------------------------
    if st.button("🚀 ทำนายสถานะการจัดส่ง", use_container_width=True):
        
        # ก. แปลงข้อมูลตัวหนังสือให้เป็นตัวเลขตามที่ AI เข้าใจ
        gender_map = {'Female': 0, 'Male': 1}
        importance_map = {'low': 0, 'medium': 1, 'high': 2}
        
        # ข. สร้างโครงตารางข้อมูลเปล่าๆ ให้ตรงกับที่โมเดลต้องการ (เติม 0 ไว้ก่อน)
        input_data = {col: 0 for col in expected_features}
        
        # ค. ใส่ข้อมูลที่ผู้ใช้กรอกลงไปในตาราง
        input_data['Customer_care_calls'] = care_calls
        input_data['Customer_rating'] = rating
        input_data['Cost_of_the_Product'] = cost
        input_data['Prior_purchases'] = prior_purchases
        input_data['Product_importance'] = importance_map[importance]
        input_data['Gender'] = gender_map[gender]
        input_data['Discount_offered'] = discount
        input_data['Weight_in_gms'] = weight
        
        # ง. จัดการข้อมูล One-Hot Encoding (คลังสินค้า และ วิธีส่ง)
        # ถ้าระบุว่าเป็น B, C, D, F หรือ Ship, Road ค่อยเปลี่ยนเป็น 1 (A และ Flight ถูกซ่อนเป็นค่าฐาน)
        if f'Warehouse_block_{warehouse}' in expected_features:
            input_data[f'Warehouse_block_{warehouse}'] = 1
            
        if f'Mode_of_Shipment_{shipment_mode}' in expected_features:
            input_data[f'Mode_of_Shipment_{shipment_mode}'] = 1
            
        # สร้างเป็น DataFrame 1 แถว
        input_df = pd.DataFrame([input_data])
        
        # จ. ปรับสเกลข้อมูล (Scaling) ด้วยไม้บรรทัดเดิมที่เซฟไว้
        input_scaled = scaler.transform(input_df)
        
        # ฉ. สั่งให้ AI ทำนาย
        prediction = model.predict(input_scaled)
        
        # ----------------------------------------
        # 5. แสดงผลลัพธ์ (Show Result)
        # ----------------------------------------
        st.subheader("🎯 ผลการทำนาย:")
        
        if prediction[0] == 1:
            st.error("⚠️ มีแนวโน้มว่าสินค้านี้จะจัดส่ง **'ล่าช้า'** (Late Delivery)")
            st.info("💡 ข้อเสนอแนะ: ออเดอร์นี้อาจมีน้ำหนักมาก หรือได้รับส่วนลดสูงจนคิวจัดส่งแน่น ควรเฝ้าระวังเป็นพิเศษ!")
        else:
            st.success("✅ มีแนวโน้มว่าสินค้านี้จะจัดส่ง **'ตรงเวลา'** (On Time)")
            st.balloons() # ใส่เอฟเฟกต์ลูกโป่งให้ดูว้าว
