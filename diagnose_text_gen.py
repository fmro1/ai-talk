
"""
Diagnostic script for text-generation-webui API connectivity
"""

import requests
import json
from typing import Dict, Any

def test_endpoint(base_url: str, endpoint: str, payload: Dict[str, Any]) -> None:
    """Test a specific endpoint with given payload."""
    url = f"{base_url}{endpoint}"
    print(f"\n🔍 Testing: {endpoint}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
        else:
            print("❌ FAILED")
            print(f"Response text: {response.text[:300]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")

def main():
    base_url = "http://localhost:5000"  # New default port
    
    print("🔧 Text-Generation-WebUI OpenAI API Diagnostic")
    print("=" * 60)
    
    # Test basic connection
    print("\n📡 Testing basic connection...")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ OpenAI API is working!")
            models = response.json()
            if 'data' in models and models['data']:
                print(f"📝 Available models: {[m.get('id', 'Unknown') for m in models['data']]}")
            else:
                print("⚠️ No models found")
        else:
            print(f"⚠️ Models endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to {base_url}: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure text-generation-webui is running with:")
        print("   ./start_linux.sh --api")
        print("2. The default port changed from 43031 to 5000")
        print("3. Try the old port too: http://localhost:43031")
        
        # Try old port
        print("\n📡 Trying old port (43031)...")
        try:
            old_base_url = "http://localhost:43031"
            response = requests.get(f"{old_base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✅ Found API on old port!")
                base_url = old_base_url
            else:
                print("❌ Old port also not working")
                return
        except:
            print("❌ Old port also not accessible")
            return
    
    # Test OpenAI-compatible endpoints
    endpoints_to_test = [
        {
            "name": "Chat Completions (Recommended)",
            "endpoint": "/v1/chat/completions",
            "payload": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello!"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
        },
        {
            "name": "Text Completions",
            "endpoint": "/v1/completions",
            "payload": {
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7
            }
        }
    ]
    
    print("\n🧪 Testing OpenAI-compatible endpoints...")
    working_endpoints = []
    
    for test_config in endpoints_to_test:
        print(f"\n🔍 Testing: {test_config['name']}")
        print(f"Endpoint: {test_config['endpoint']}")
        
        try:
            response = requests.post(
                f"{base_url}{test_config['endpoint']}", 
                json=test_config["payload"], 
                timeout=15
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                result = response.json()
                
                # Extract and show response
                if "choices" in result:
                    if "message" in result["choices"][0]:
                        # Chat format
                        content = result["choices"][0]["message"]["content"]
                    else:
                        # Completions format
                        content = result["choices"][0]["text"]
                    
                    print(f"Response: {content.strip()[:100]}...")
                    working_endpoints.append(test_config["endpoint"])
                else:
                    print("⚠️ Unexpected response format")
                    
            else:
                print("❌ FAILED")
                print(f"Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📋 Summary:")
    print(f"Base URL: {base_url}")
    
    if working_endpoints:
        print("✅ Working endpoints:")
        for endpoint in working_endpoints:
            print(f"  - {endpoint}")
            
        print(f"\n🔧 Your config.json should use:")
        print(f'  "base_url": "{base_url}"')
        if "/v1/chat/completions" in working_endpoints:
            print('  "use_chat_format": true  # Recommended')
        else:
            print('  "use_chat_format": false')
    else:
        print("❌ No working endpoints found!")
        print("\n🔧 Setup instructions:")
        print("1. Start text-generation-webui with API enabled:")
        print("   cd ~/dev-ai/text-generation-webui")
        print("   ./start_linux.sh --api")
        print("2. Load a model in the web interface")
        print("3. The new API uses OpenAI-compatible endpoints")
    
    # Test web interface
    print(f"\n🌐 Web interface test...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Web interface accessible")
        else:
            print(f"⚠️ Web interface status: {response.status_code}")
    except:
        print("❌ Web interface not accessible")

if __name__ == "__main__":
    main()