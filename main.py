from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import sys
import os
import shutil
import json
from predict.PillMain import PillMain

app = FastAPI()

@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...), text_file: UploadFile = File(...)):
    # 파일 저장 경로 설정
    file1_path = f"./data/{file1.filename}"
    file2_path = f"./data/{file2.filename}"
    text_file_path = f"./data/{text_file.filename}"

    try:
        # 파일 저장
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)

        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        with open(text_file_path, "wb") as buffer:
            shutil.copyfileobj(text_file.file, buffer)

        # PillMain 클래스 인스턴스 생성 및 실행
        pill_main = PillMain()
        sys.argv = ["main.py", file1_path, file2_path, text_file_path]
        pill_main.main(sys.argv)

        # 예측 결과를 로드
        log_path = './data/pred_log/'
        latest_log_file = max([os.path.join(log_path, f) for f in os.listdir(log_path)], key=os.path.getctime)

        with open(latest_log_file, "r") as log_file:
            result = log_file.read()

        # 예측 결과를 JSON으로 파싱
        result_data = json.loads(result)

        # 예측 결과 반환
        return JSONResponse(content={"result": result_data})

    finally:
        # 사용한 파일 삭제
        os.remove(file1_path)
        os.remove(file2_path)
        os.remove(text_file_path)
        if os.path.exists(latest_log_file):
            os.remove(latest_log_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
