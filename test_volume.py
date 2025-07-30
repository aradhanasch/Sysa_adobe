import os

print("Current working directory:", os.getcwd())
print("Contents of /app/input:")
try:
    files = os.listdir('/app/input')
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"Error: {e}")

print("Contents of current directory:")
try:
    files = os.listdir('.')
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"Error: {e}") 