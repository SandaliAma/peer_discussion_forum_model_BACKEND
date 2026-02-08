"""
Quick test script to verify Gemini API is working
"""

import google.generativeai as genai
import config

print("\n" + "="*70)
print("TESTING GEMINI API")
print("="*70)

# Configure Gemini
print(f"\n1. API Key: {'✓ Set' if config.GEMINI_API_KEY else '✗ Not set'}")
print(f"2. Model: {config.GEMINI_MODEL}")

if not config.GEMINI_API_KEY:
    print("\n❌ ERROR: No API key found in config.py")
    print("Please add your Gemini API key to config.py")
    exit(1)

genai.configure(api_key=config.GEMINI_API_KEY)

print("\n3. Testing model connection...")

try:
    # List available models
    print("\n4. Available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - {m.name}")

    # Test generation
    print(f"\n5. Testing {config.GEMINI_MODEL}...")
    model = genai.GenerativeModel(config.GEMINI_MODEL)

    response = model.generate_content("2 + 2 හි පිළිතුර කීයද?")

    print("\n6. Test Response:")
    print(f"   {response.text}")

    print("\n" + "="*70)
    print("✅ GEMINI API WORKING CORRECTLY!")
    print("="*70)

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR")
    print("="*70)
    print(f"\nError: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Try a different model name:")
    print("   - gemini-1.5-pro-latest")
    print("   - gemini-1.5-flash-latest")
    print("   - gemini-pro")
    print("3. Check your internet connection")
    print("4. Verify API is enabled in Google Cloud Console")
    print("="*70)
