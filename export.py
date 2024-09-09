import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# 페이지 설정
st.set_page_config(page_title="손꾸락 판별기", page_icon="👋", layout="wide")

# CSS를 사용한 스타일링
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #1E88E5;
}
.result-font {
    font-size:24px !important;
    font-weight: bold;
    color: #4CAF50;
}
.stProgress > div > div > div > div {
    background-color: #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# 사용자 정의 객체 정의
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# 사용자 정의 객체를 사용하여 모델 로드
@st.cache_resource
def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

# Load the model
model_path = os.path.join(current_dir, "keras_Model.h5")
if not os.path.exists(model_path):
    st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
    st.stop()

try:
    model = load_model_with_custom_objects(model_path)
except Exception as e:
    st.error(f"모델 로딩 중 오류 발생: {str(e)}")
    st.stop()

# Load the labels
labels_path = os.path.join(current_dir, "labels.txt")
if not os.path.exists(labels_path):
    st.error(f"레이블 파일을 찾을 수 없습니다: {labels_path}")
    st.stop()

class_names = open(labels_path, "r").readlines()

# 이미지 전처리
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# 예측
def predict(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:].strip(), confidence_score

# Streamlit 앱
def main():
    st.markdown('<p class="big-font">👋 손꾸락 판별기</p>', unsafe_allow_html=True)
    st.write("AI가 당신의 손 제스처를 인식합니다. 이미지를 업로드해보세요!")

    uploaded_file = st.file_uploader("이미지를 선택하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner('이미지 처리 중...'):
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='업로드된 이미지', use_column_width=True)
            
            with col2:
                # 예측
                result, confidence = predict(image)
                st.markdown(f'<p class="result-font">인식된 손가락 숫자: {result}</p>', unsafe_allow_html=True)
                st.write(f"신뢰도: {confidence:.2f}")
                
                # 프로그레스 바로 신뢰도 표시
                st.progress(float(confidence))
                
                # 결과에 따른 이모지 표시
                if confidence > 0.8:
                    st.success("높은 신뢰도로 인식되었습니다! 👍")
                elif confidence > 0.5:
                    st.warning("중간 정도의 신뢰도입니다. 다시 시도해보세요. 🤔")
                else:
                    st.error("낮은 신뢰도입니다. 다른 이미지로 시도해보세요. 😕")

    st.markdown("---")
    st.write("© 2024 손꾸락 판별기 | 제작: 지상하")

if __name__ == "__main__":
    main()