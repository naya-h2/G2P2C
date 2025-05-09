from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def example_history():
    # 12개 step, [[혈당, 인슐린]] (mg/dL, U/h), 가장 간단한 정상 예시
    return [
        [110.5, 0.15], [112.1, 0.18], [113.4, 0.10], [115.2, 0.20],
        [118.3, 0.14], [120.0, 0.22], [121.8, 0.30], [123.4, 0.24],
        [122.7, 0.18], [124.9, 0.20], [125.0, 0.25], [126.1, 0.27]
    ]

def test_predict_action_success():
    payload = {"history": example_history()}
    response = client.post("/predict_action", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    # 행동값이 범위 내인지 체크 (0.0 ~ 5.0)
    assert "insulin_action_U_per_h" in data
    assert 0.0 <= data["insulin_action_U_per_h"] <= 5.0

def test_predict_action_failure_shape():
    # history의 shape가 이상할 때(11개)
    wrong_history = example_history()[:-1]
    payload = {"history": wrong_history}
    response = client.post("/predict_action", json=payload)
    assert response.status_code == 500
    assert "history 형상 오류" in response.json()["detail"]

def test_predict_action_nan():
    nan_history = example_history()
    nan_history[2][0] = float("nan")
    payload = {"history": nan_history}
    response = client.post("/predict_action", json=payload)
    assert response.status_code == 500
    assert "입력 NaN/Inf" in response.json()["detail"]