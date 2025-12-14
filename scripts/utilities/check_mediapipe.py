import mediapipe as mp
import os

mp_path = os.path.dirname(mp.__file__)
print(f'MediaPipe path: {mp_path}')

modules_path = os.path.join(mp_path, 'modules')
print(f'Modules dir exists: {os.path.exists(modules_path)}')

if os.path.exists(modules_path):
    print(f'\nModules directory contents:')
    for root, dirs, files in os.walk(modules_path):
        level = root.replace(modules_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:  # Only show 2 levels deep
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
else:
    print('\n⚠️ Modules directory not found!')
    print('\nMediaPipe directory contents:')
    for item in os.listdir(mp_path)[:20]:
        print(f'  {item}')
