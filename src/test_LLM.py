import requests
import json
import time

def test_simple_connection():
    # --- 1. 配置信息 (直接使用你之前的配置) ---
    api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    api_key = "cc03e248-3c76-4216-838e-2944190cdb3a"
    model_id = "doubao-seed-1-6-250615"
    
    # --- 2. 构造最简单的请求 ---
    # 我们只发一句"你好"，强制模型快速回复，排除推理耗时的干扰
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "请回复数字1。"}
        ],
        "temperature": 0.1
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    print("--- 开始测试 LLM 连接 ---")
    print(f"目标 URL: {api_url}")
    print("正在发送请求 (Timeout 设置为 30秒)...")

    start_time = time.time()

    try:
        # 发送请求
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # 计算耗时
        duration = time.time() - start_time
        print(f"✅ 请求返回! 耗时: {duration:.2f} 秒")
        print(f"状态码: {response.status_code}")

        # 检查是否成功
        if response.status_code == 200:
            res_json = response.json()
            print("\n⬇️ 返回内容:")
            print(json.dumps(res_json, indent=2, ensure_ascii=False))
            
            content = res_json['choices'][0]['message']['content']
            print(f"\n💬 模型回复: {content}")
        else:
            print("\n❌ API 报错:")
            print(response.text)

    except requests.exceptions.ReadTimeout:
        duration = time.time() - start_time
        print(f"\n❌ 超时错误 (ReadTimeout)!")
        print(f"耗时 {duration:.2f} 秒后连接断开。")
        print("这说明服务器收到了请求，但在规定时间内没发回数据，或者网络链路阻塞。")

    except requests.exceptions.ConnectTimeout:
        print("\n❌ 连接超时 (ConnectTimeout)!")
        print("无法连接到服务器。请检查你的网络、DNS 或防火墙设置。")
        
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

if __name__ == "__main__":
    test_simple_connection()