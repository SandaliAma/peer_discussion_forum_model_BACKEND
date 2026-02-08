"""
Quick test script to verify Groq API is working
"""

from groq import Groq
import config

print("\n" + "="*70)
print("TESTING GROQ API")
print("="*70)

# Check configuration
print(f"\n1. API Key: {'✓ Set' if config.GROQ_API_KEY else '✗ Not set'}")
print(f"2. Model: {config.GROQ_MODEL}")

if not config.GROQ_API_KEY:
    print("\n❌ ERROR: No API key found in config.py")
    print("Please add your Groq API key to config.py")
    exit(1)

print("\n3. Testing model connection...")

try:
    # Initialize client
    client = Groq(api_key=config.GROQ_API_KEY)

    # Test with a simple math question
    print(f"\n4. Testing {config.GROQ_MODEL}...")

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a math teacher. Answer in Sinhala."
            },
            {
                "role": "user",
                "content": "2 + 2 හි පිළිතුර කීයද?"
            }
        ],
        temperature=0.3,
        max_tokens=256
    )

    answer = response.choices[0].message.content

    print("\n5. Test Response:")
    print(f"   {answer}")

    print("\n" + "="*70)
    print("✅ GROQ API WORKING CORRECTLY!")
    print("="*70)

    # Test with a more complex math question
    print("\n6. Testing complex math question...")

    response2 = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "ඔබ ගණිත ගුරුවරයෙකි. සිංහල භාෂාවෙන් පියවරෙන් පියවර ගණිත ගැටළු විසඳන්න."
            },
            {
                "role": "user",
                "content": "x = √3/2 නම් x² + 1/4 හි අගය සොයන්න. පියවරෙන් පියවර විසඳන්න."
            }
        ],
        temperature=0.3,
        max_tokens=1024
    )

    answer2 = response2.choices[0].message.content

    print("\n7. Complex Math Response:")
    print("-"*70)
    print(answer2)
    print("-"*70)

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYou can now restart the server:")
    print("   python fast_api_server.py")

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR")
    print("="*70)
    print(f"\nError: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Install groq package: pip install groq")
    print("3. Check your internet connection")
    print("4. Verify API key at https://console.groq.com/keys")
    print("="*70)

    import traceback
    traceback.print_exc()
