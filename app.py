from flask import Flask, request, jsonify
import requests
import json
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import os

app = Flask(__name__)

# -- 설정 파일 로드 및 모델 초기화 --
CONFIG_FILE = './models/models_configs.json'
loaded_models = {} #로드된 YOLO 모델 인스턴스를 저장할 딕셔너리
models_config = {} #models_config.json 파일 내용을 저장할 딕셔너리

try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        models_config = json.load(f)
    print(f"'{CONFIG_FILE}' 파일 로드 성공.")

    #config 파일에 정의된 모든 모델을 미리 로드
    for challenge_id_str, config in models_config.items():
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                loaded_models[challenge_id_str] = YOLO(model_path)
                print(f"Challenge ID '{challenge_id_str}' 모델 '{model_path}' 로드 성공.")
            except Exception as e:
                print(f"ERROR : Challenge ID '{challenge_id_str}' 모델 로드 실패 : {e}")
        else:
            print(f"경고: Challenge ID '{challenge_id_str}'의 모델 경로가 없거나 파일이 존재하지 않습니다.")
except FileNotFoundError:
    print(f"ERROR : '{CONFIG_FILE}' 파일을 찾을 수 없습니다. API 정상 작동 불가능")
except json.JSONDecodeError:
    print(f"ERROR : '{CONFIG_FILE}' 파일의 형식이 옳바르지 않습니다. JSON 구문을 확인해주세요.")
except Exception as e:
    print(f"ERROR : 애플리케이션 초기화 중 오류 발생 : {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 요청으로부터 데이터 받기
        data = request.get_json()

        challenge_id_str = str(data.get('challengeId'))
        mission_id_str = str(data.get('missionId'))
        user_id = data.get('userId')
        temp_image_id = data.get('tempImageId')
        image_url = data.get('imageUrl')

        if not all([challenge_id_str, mission_id_str, user_id, temp_image_id, image_url]):
            return jsonify({'error' : 'Missing data in request.'}), 400
        
        # 2. challengeId에 해당하는 모델 및 설정 정보 가져오기
        selected_model = loaded_models.get(challenge_id_str)
        model_config_data = models_config.get(challenge_id_str)

        if not selected_model:
            return jsonify({'error': f'Model for challengeId "{challenge_id_str}" is not loaded or available.'}), 500
        if not model_config_data:
            return jsonify({'error': f'Configuration for challengeId "{challenge_id_str}" not found in {CONFIG_FILE}.'}), 500
        
        #missionId에 맞는 목표 클래스명 선택
        mission_map = model_config_data.get('mission_map', {})
        target_class_name = mission_map.get(mission_id_str)

        if not target_class_name:
            return jsonify({'error': f'Invalid missionId "{mission_id_str}" for challengeId "{challenge_id_str}". No corresponding class mapping found.'}), 400

        # 3. 이미지 다운로드
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': f'Could not download image from the provided URL. Status Code: {response.status_code}'}), 400
        
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes).convert("RGB")

        # 4. 모델 추론 수행
        results = selected_model.predict(source=image, conf=0.25)

        predicted_confidence = 0.0
        class_detected = False
        bbox_coords = None

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detected_class_name = selected_model.names.get(class_id)

                if detected_class_name == target_class_name:
                    predicted_confidence = confidence
                    class_detected = True
                    bbox_coords = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
                    break
        
        response_data = {
            'success' : True,
            'challengeId' : challenge_id_str,
            'missionId' : mission_id_str,
            'userId' : user_id,
            'tempImageId' : temp_image_id,
            'imageUrl' : image_url,
            'Confidence' : round(predicted_confidence, 4),
            'classDetected' : class_detected,
            'bbox' : bbox_coords
        }

        # --- 이 부분에 응답 데이터 로그를 추가합니다 ---
        print("생성된 응답 데이터:")
        print(response_data)

        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"서버 처리 중 예상치 못한 오류 발생 : {e}")
        return jsonify({'success' : False, 'error' : f'error occurred :{e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
