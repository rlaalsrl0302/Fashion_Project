# from sklearn.metrics.pairwise import cosine_similarity
# import h5py # 모델이 추출한 무신사 데이터셋의 특징 저장
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import faiss
import PIL
import os
import cv2

# 이미지 데이터 불러오기
def get_file_paths():
    # 모든 파일을 저장할 버퍼
    file_path = {}
    img_dir = os.path.join(os.getcwd(), "data_storage/images")
    for root, _, files in os.walk(img_dir):
        category = root.replace(img_dir, '')
        if category:
            file_path[category[1:]] = files
    return file_path        

# 모델 생성 함수
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
    return model

# 해당 폴더 안의 모든 이미지를 가져옴
dataset_paths = get_file_paths()

def extract_features(img_path, model):
    img_path = "/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/" + img_path
    img = image.load_img(img_path, target_size=(384, 384))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# 벡터디비 생성
def db_create(model):
    # Initialize FAISS index
    dimension = model.output.shape[1]  # This should match the feature vector size
    index = faiss.IndexFlatL2(dimension)
    return index

# 모델 생성
model = model_create()
# DB 생성
index = db_create(model)

# DB 담을 버퍼?
dataset_features = {}
for category in dataset_paths:  # dataset_paths is a list of image file paths
    for img_path in dataset_paths[category]:
        features = extract_features(category + '/' + img_path, model)
        dataset_features[category] = features
        index.add(np.array([features]))  # Add features to FAISS index

# Now you can use the 'index' for similarity searches

## ----------- 이미지 서치하는 함수 
def search_similar_images(query_img_path, model, index, dataset_paths, n_results=5):
    """
    Search for similar images in the FAISS index.

    :param query_img_path: Path to the query image.
    :param model: Pretrained Keras model for feature extraction.
    :param index: FAISS index containing features of the dataset.
    :param dataset_paths: List of paths to images in the dataset.
    :param n_results: Number of similar images to retrieve.
    :return: List of paths to similar images.
    """
    # Extract features from the query image
    query_features = extract_features(query_img_path, model)

    # Search the FAISS index
    distances, indices = index.search(np.array([query_features]), n_results)
    
    # Retrieve the similar images
    similar_images = [dataset_paths[i] for i in indices[0]]

    return similar_images

# 추출한 파일 저장하기 
faiss.write_index(index, "index.faiss")

# 테스트 파일 
query_img_path = 'co.png'
similar_images = search_similar_images(query_img_path, model, index, dataset_paths, n_results=5)

# 유사한 이미지 출력
for img_path in similar_images:
    print(img_path)
    image = cv2.imread("img_path")
    cv2.imshow("이미지",image)