def predict_yolo ( model , img_path , save_path=None ,confidenct=40, overlap = 30 ) :
#     '''
#     return : x, y 값 좌표 ( 예측된 사각형의 젤 왼쪽 위 끝점 임 )
#             width , height
#             class : 카테고리 이름
#     '''
#     if not save_path :
#     #save_path = os.getcwd() + '/' + img_path.split('/')[-1] # 현재 폴더에 저장하는것보다
#         save_path = img_path[:-(len(img_path.split('/')[-1]))] + 'predict_' + img_path.split('/')[-1]


#     predict = model.predict(img_path, confidence = confidenct, overlap = overlap)

#     # 예측 정보 출력
#     predict_info = predict.json()
    
#     # 예측 사진 다운로드 하기
#     predict.save(save_path)
#     result = list()
#     for info in predict_info["predictions"]:
#         if info["class"] not in ["hat", "sunglass", "bag", "shoe"]:
#             start_x, start_y, width, height, category = info['x']-(info["width"] // 2), info['y']-(info["height"] // 2), info['width'], info['height'], info['class']
#             end_x, end_y = (start_x + width), (start_y + height)
#             result.append(image[start_y:end_y, start_x:end_x])

#     return result