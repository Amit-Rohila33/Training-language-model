import threading
import requests


def make_request():
    url = "http://localhost:8000/generate"
    data = {"text": "Input text"}
    response = requests.post(url, json=data)
    print(response.json())


def stress_test():
    threads = []
    num_threads = 10

    for _ in range(num_threads):
        t = threading.Thread(target=make_request)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    stress_test()
