from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
from roboflow import Roboflow
from PIL import Image
import numpy as np
import faiss
import PIL
import os
import cv2

def get_file_paths():
    # 모든 파일을 저장할 버퍼
    file_path = {}
    img_dir = os.path.join(os.getcwd(), "data_storage/images")
    for root, _, files in os.walk(img_dir):
        category = root.replace(img_dir, '')
        if category:
            file_path[category[1:]] = files
    return file_path        

def predict_yolo(model, img_path, confidenct=40, overlap = 30):
    '''
    return : x, y 값 좌표 ( 예측된 사각형의 젤 왼쪽 위 끝점 임 )
            width , height
            class : 카테고리 이름
    '''

    predict = model.predict(img_path, confidence = confidenct, overlap = overlap)
    image = cv2.imread(img_path)

    # 예측 정보 출력
    predict_info = predict.json()
    
    # 예측 사진 다운로드 하기
    result = list()
    for info in predict_info["predictions"]:
        if info["class"] not in ["hat", "sunglass", "bag", "shoe"]:
            start_x, start_y, width, height, category = info['x']-(info["width"] // 2), info['y']-(info["height"] // 2), info['width'], info['height'], info['class']
            end_x, end_y = (start_x + width), (start_y + height)
            result.append(image[start_y:end_y, start_x:end_x])

    return result

def model_create():
    # EfficientNetV2S 모델 객체 생성
    base_model = EfficientNetV2S(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    # EfficientNetV2S 모델의 데이터 추출 모드 사용 
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    rf = Roboflow(api_key="Q78HnDQgOoukAA6rXrsG")
    project = rf.workspace().project("fashion-hkjfr")
    rf_model = project.version(5).model
    return model, rf_model

def extract_features(img, model):
    #img_path = "/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/" + img_path
    # img = image.load_img(img_path, target_size=(384, 384))
    img = Image.fromarray(img).resize((384, 384))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def search_similar_images(query_img_path, model, index, dataset_paths, n_results=3):
    """
    Search for similar images in the FAISS index.

    :param query_img_path: Path to the query image.
    :param model: Pretrained Keras model for feature extraction.
    :param index: FAISS index containing features of the dataset.
    :param dataset_paths: List of paths to images in the dataset.
    :param n_results: Number of similar images to retrieve.
    :return: List of paths to similar images.
    """
    result = []
    # Extract features from the query image
    pred_img = predict_yolo(rf_model, query_img_path)
    for img in pred_img:
        query_features = extract_features(img, model)

        # Search the FAISS index
        distances, indices = index.search(np.array([query_features]), n_results)
        print(distances, indices)
        # Retrieve the similar images
        similar_images = [dataset_paths[i] for i in indices[0]]
        print(similar_images)
        result += similar_images
    return result

# 벡터 디비 호출
index = faiss.read_index("/Users/mingi/Desktop/Sesac_Project/Fashion_Project/index.faiss")

# 모델 생성
model, rf_model = model_create()

img_name = get_file_paths()

dataset_paths = list()
for key in img_name: # ["top", "skirt", "outer", "onepiece", "pants"]:
    for img_path in img_name[key]:
        dataset_paths.append(img_path)
        
# 테스트 파일 
query_img_path = '/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/k1.png'
similar_images = search_similar_images(query_img_path, model, index, dataset_paths, n_results = 3)

# 유사한 이미지 출력
for img_path in similar_images:
    for category in img_name[key]:
        path = "/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/" + category + '/' + img_path
        if os.path.exists(path):
            image = cv2.imread(path)
            cv2.imshow("이미지", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            