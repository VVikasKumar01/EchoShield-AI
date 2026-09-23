import sounddevice as sd

print("Default device:", sd.default.device)
print()
devs = sd.query_devices()
print("=" * 65)
print("ALL INPUT DEVICES:")
print("=" * 65)
for i, d in enumerate(devs):
    if d['max_input_channels'] > 0:
        print(f"[{i:2d}] channels={d['max_input_channels']}  name={d['name']}")
print("=" * 65)
