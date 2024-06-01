from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import sys
import os
import shutil
import json
from predict.PillMain import PillMain
import logging
import traceback

app = FastAPI()

# 로그 설정
logging.basicConfig(level=logging.INFO)

@app.post("/predict")
async def predict(frontImage: UploadFile = File(...), backImage: UploadFile = File(...), pillCsv: UploadFile = File(...)):
    
    # 파일 저장 경로 설정
    data_dir = "./data/"
    log_dir = "./data/pred_log/"

    # 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    file1_path = os.path.join(data_dir, frontImage.filename)
    file2_path = os.path.join(data_dir, backImage.filename)
    text_file_path = os.path.join(data_dir, pillCsv.filename)

    try:
        # 파일 저장
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(frontImage.file, buffer)
        logging.info(f"Saved file1 to {file1_path}")

        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(backImage.file, buffer)
        logging.info(f"Saved file2 to {file2_path}")

        with open(text_file_path, "wb") as buffer:
            shutil.copyfileobj(pillCsv.file, buffer)
        logging.info(f"Saved text_file to {text_file_path}")

        # 저장된 파일 크기 로그 출력
        logging.info(f"Saved file1 to {file1_path}, size: {os.path.getsize(file1_path)} bytes")
        logging.info(f"Saved file2 to {file2_path}, size: {os.path.getsize(file2_path)} bytes")
        logging.info(f"Saved text_file to {text_file_path}, size: {os.path.getsize(text_file_path)} bytes")

        # PillMain 클래스 인스턴스 생성 및 실행
        pill_main = PillMain()
        sys.argv = ["PillMain.py", file1_path, file2_path, text_file_path]
        pill_main.main(sys.argv)

        # 예측 결과를 로드
        latest_log_file = max([os.path.join(log_dir, f) for f in os.listdir(log_dir)], key=os.path.getctime)

        with open(latest_log_file, "r") as log_file:
            result = log_file.read()

        # 예측 결과를 JSON으로 파싱
        result_data = json.loads(result)

        # 예측 결과 반환
        return JSONResponse(content={"result": result_data})

    except Exception as e:
        # 에러 로그 반환
        error_message = str(e)
        logging.error(f"Error during processing: {error_message}")
        logging.error(traceback.format_exc())
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        # 사용한 파일 삭제
        os.remove(file1_path)
        os.remove(file2_path)
        os.remove(text_file_path)
        temp_image1_path = "./data/temp_processed_image1.jpeg"
        temp_image2_path = "./data/temp_processed_image2.jpeg"
        os.remove(temp_image1_path)
        os.remove(temp_image2_path)
        if 'latest_log_file' in locals() and os.path.exists(latest_log_file):
            os.remove(latest_log_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
